package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// Forwarding requests for models kinfer cannot load.
//
// Ollama lists cloud models alongside local ones; they carry no weights, and
// the name resolves to hosted inference. kinfer used to drop them, which meant
// a caller asking for one was told the model did not exist and went to Ollama
// directly — leaving kinfer out of that request's path entirely.
//
// That is the path that failed. On 2026-07-23 `kimi-k2.5:cloud` stopped
// answering, without erroring, and about 150 agents waited on it for half a day.
// Every protection this runtime has since grown — a wall clock on a reply,
// errors carried in-band through a stream, a bounded queue that refuses rather
// than hangs — applies to local models only, and the model that actually caused
// the outage is exactly the one none of them can reach.
//
// So a request for a cloud model is forwarded rather than refused, and the
// forwarding is where the clock goes.

// DefaultUpstream is where cloud models are served from. Ollama holds the
// credentials and the routing; kinfer is in front of it for the timeout.
const DefaultUpstream = "http://localhost:11434"

// Upstream returns the address cloud requests are forwarded to, honouring
// OLLAMA_HOST the way Ollama's own clients do so a machine configured once
// works here too.
func Upstream() string {
	h := os.Getenv("OLLAMA_HOST")
	if h == "" {
		return DefaultUpstream
	}
	if !strings.Contains(h, "://") {
		h = "http://" + h
	}
	if _, err := url.Parse(h); err != nil {
		return DefaultUpstream
	}
	return strings.TrimSuffix(h, "/")
}

// SetUpstream points cloud requests somewhere other than the default. Empty is
// ignored, so a caller passing an unset flag does not silently disable routing.
func (s *Server) SetUpstream(addr string) {
	if addr != "" {
		s.upstream = strings.TrimSuffix(addr, "/")
	}
}

// remoteModel reports whether this name belongs to something served elsewhere.
//
// The test is the manifest's own — a cloud entry has no weights layer — and not
// the ":cloud" suffix. Guessing from a name would forward a typo to a hosted
// endpoint and turn a fast "no such model" into a slow one.
func (s *Server) remoteModel(name string) bool {
	m, err := s.store.Lookup(name)
	return err == nil && m.Remote()
}

// forward proxies one request upstream, unchanged, and streams the reply back
// as it arrives.
//
// Same path, same body, same dialect: /api/chat goes to /api/chat and
// /v1/chat/completions to /v1/chat/completions. Ollama speaks both, so there is
// nothing to translate, and translating would be a second place for the shape
// of a request to drift.
//
// What kinfer adds is the deadline. That is the entire point of being in this
// path: the upstream may stop answering, and a caller that cannot tell a slow
// model from a dead one is the failure this runtime exists to prevent.
// retryableUpstream is a failure that says "not now" rather than "not this
// request": the work is fine, the upstream cannot take it. Only these fall back.
//
// A 400 or a 401 must not: the request is wrong or unauthorised, and running it
// on a different model produces a confident answer to a question the caller
// already got wrong. A 404 must not either — that name does not exist upstream,
// and substituting a local model would hide a typo behind a plausible reply.
func retryableUpstream(status int) bool {
	switch status {
	case http.StatusTooManyRequests, // rate limited, which is the case this exists for
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	}
	return false
}

// forward proxies one request upstream. It reports whether the caller should
// fall back to a local model instead.
//
// True is only ever returned when nothing has been written: no header, no byte
// of body. Once a reply has begun streaming there is no falling back, for the
// same reason a request already mid-batch is not re-run when an engine dies —
// the client has seen part of an answer, and a second attempt would repeat it.
func (s *Server) forward(w http.ResponseWriter, r *http.Request, body []byte, model string) (fallback bool) {
	deadline := s.opts.MaxGenerate
	if deadline == 0 {
		deadline = engine.DefaultMaxGenerate
	}

	ctx := r.Context()
	var cancel context.CancelFunc
	if deadline > 0 {
		ctx, cancel = context.WithTimeout(ctx, deadline)
		defer cancel()
	}

	target := s.upstream + r.URL.Path
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	if a := r.Header.Get("Authorization"); a != "" {
		req.Header.Set("Authorization", a)
	}

	s.stats.forwarded.Add(1)
	log.Printf("→ %s for %s, forwarded to %s", r.URL.Path, model, s.upstream)
	resp, err := s.upstreamClient.Do(req)
	if err != nil {
		// An upstream that is unreachable or silent is the same kind of "not
		// now" as a 429, and nothing has been written yet.
		if s.fallback != "" {
			return true
		}
		// An upstream that is not there must fail immediately: turning "Ollama
		// is not running" into a five-minute wait would move the original
		// outage rather than fix it.
		writeUpstreamError(w, model, s.upstream, err)
		return false
	}
	defer resp.Body.Close()

	if s.fallback != "" && retryableUpstream(resp.StatusCode) {
		return true
	}

	for _, h := range []string{"Content-Type", "Retry-After"} {
		if v := resp.Header.Get(h); v != "" {
			w.Header().Set(h, v)
		}
	}
	w.WriteHeader(resp.StatusCode)

	// Copy through unbuffered. A streamed reply that arrives in one lump at the
	// end has lost the only thing streaming was for.
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 4<<10)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return false
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if rerr != nil {
			if !errors.Is(rerr, io.EOF) && ctx.Err() != nil {
				// The deadline landed mid-reply. The header is long gone, so
				// this travels in the body, the same way a local generation
				// failure does.
				fmt.Fprintf(w, "\n{\"error\":%q,\"done\":true,\"done_reason\":\"timeout\"}\n",
					fmt.Sprintf("%s stopped answering after %s", model, deadline))
				if flusher != nil {
					flusher.Flush()
				}
			}
			return false
		}
	}
}

// writeUpstreamError says which failure this was, because the three need
// different things from whoever reads it.
func writeUpstreamError(w http.ResponseWriter, model, upstream string, err error) {
	status := http.StatusBadGateway
	msg := fmt.Sprintf("%s is served by %s, which did not answer: %v", model, upstream, err)

	switch {
	case errors.Is(err, context.DeadlineExceeded):
		status = http.StatusGatewayTimeout
		msg = fmt.Sprintf("%s is served by %s, which stopped answering", model, upstream)
	case isConnRefused(err):
		status = http.StatusServiceUnavailable
		msg = fmt.Sprintf("%s is a cloud model served by %s, and nothing is listening there — "+
			"start Ollama, or set OLLAMA_HOST", model, upstream)
	}
	writeJSON(w, status, map[string]string{"error": msg})
}

func isConnRefused(err error) bool {
	var oe *net.OpError
	return errors.As(err, &oe) && strings.Contains(oe.Error(), "connection refused")
}
