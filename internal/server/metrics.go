package server

import (
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
)

// What a fleet operator needs to see.
//
// Everything kinfer knew about itself was in a debug log behind an environment
// variable. That is enough to tune a server once, with a script, which is how
// every number in this repository was arrived at — and no use at all at three
// in the morning when 150 agents are slow and the question is whether to raise
// -slots, raise -queue, or buy a machine.
//
// The three that answer it: how many requests are waiting, how many slots are
// busy, and what share of replies are being refused. A queue that is deep while
// slots sit idle is a different problem from slots that are always full.
//
// Prometheus text format because it is the one every scraper reads, and because
// it is plain enough to be useful with curl alone.

type counters struct {
	// By HTTP status class, recorded by middleware rather than by each handler,
	// so a new endpoint cannot forget to count itself.
	ok, refused, rejected, failed atomic.Int64
}

// countingWriter remembers the status so the middleware can classify a request
// after the handler has finished with it.
type countingWriter struct {
	http.ResponseWriter
	status int
}

func (w *countingWriter) WriteHeader(code int) {
	if w.status == 0 {
		w.status = code
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *countingWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK // Go's implicit header on first write
	}
	return w.ResponseWriter.Write(b)
}

// Flush keeps streaming working: the wrapper must not hide the Flusher the
// handlers reach for, or every streamed reply arrives in one lump at the end.
func (w *countingWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (s *Server) count(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cw := &countingWriter{ResponseWriter: w}
		next.ServeHTTP(cw, r)
		// Refusals first: 503 is a 5xx, and the general case would swallow it —
		// it did, and "refused" read zero through an afternoon of refusals.
		switch {
		case cw.status == http.StatusServiceUnavailable || cw.status == http.StatusTooManyRequests:
			s.stats.refused.Add(1)
		case cw.status >= 500:
			s.stats.failed.Add(1)
		// A 4xx is the caller's request being wrong for this server — a prompt
		// larger than a slot, an unknown model. It went uncounted, and a client
		// whose every request was 413 read as "no requests" here while its
		// operator asked why nothing answered.
		case cw.status >= 400:
			s.stats.rejected.Add(1)
		case cw.status >= 200 && cw.status < 400:
			s.stats.ok.Add(1)
		}
	})
}

// handleMetrics reports what the server is doing now and what it has done.
//
// The live numbers come from the resident engine and are a snapshot: by the
// time they are read they have moved. That is what makes them useful — a
// scraper takes them repeatedly, and the shape over time is the answer.
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	var b strings.Builder

	metric := func(name, help, typ string, value any, labels string) {
		fmt.Fprintf(&b, "# HELP kinfer_%s %s\n# TYPE kinfer_%s %s\nkinfer_%s%s %v\n",
			name, help, name, typ, name, labels, value)
	}

	s.mu.Lock()
	cur := s.cur
	s.mu.Unlock()

	if cur != nil {
		waiting, busy, slots := cur.eng.Load()
		metric("queue_depth", "Requests waiting for a slot.", "gauge", waiting, "")
		metric("slots_busy", "Slots currently generating.", "gauge", busy, "")
		metric("slots_total", "Slots this model was loaded with.", "gauge", slots, "")
		metric("prompt_tokens_total", "Prompt tokens processed since this model loaded.",
			"counter", cur.eng.PromptTokens(), "")
		metric("eval_tokens_total", "Tokens generated since this model loaded.",
			"counter", cur.eng.EvalTokens(), "")
		metric("model_loaded", "1 for the model currently resident.", "gauge", 1,
			fmt.Sprintf("{model=%q}", cur.path))
	} else {
		// Distinguishable from a busy server with an empty queue: no model is
		// resident at all, so these are not zero, they are absent.
		metric("model_loaded", "1 for the model currently resident.", "gauge", 0, `{model=""}`)
	}

	metric("requests_total", "Requests answered.", "counter", s.stats.ok.Load(), `{outcome="ok"}`)
	fmt.Fprintf(&b, "kinfer_requests_total{outcome=\"refused\"} %d\n", s.stats.refused.Load())
	fmt.Fprintf(&b, "kinfer_requests_total{outcome=\"rejected\"} %d\n", s.stats.rejected.Load())
	fmt.Fprintf(&b, "kinfer_requests_total{outcome=\"failed\"} %d\n", s.stats.failed.Load())

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = w.Write([]byte(b.String()))
}
