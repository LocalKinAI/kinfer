package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/mlxasr"
	"github.com/LocalKinAI/kinfer/internal/mlxembed"
	"github.com/LocalKinAI/kinfer/internal/mlximage"
	"github.com/LocalKinAI/kinfer/internal/mlxtts"
)

// ─── every model in a process of its own ─────────────────────────────────────
//
// With chat runners (the default for `kinfer serve`), this server loads no
// model itself: a chat or embedding request goes, by the model it names, to a
// runner of that model — this binary's serve on a port of the loopback, with
// this server's own sizing flags — as speech and pictures do (runners.go).
// A model that crashes takes only its runner down, and the next request
// starts it again; a model unloaded is a process ended, so all its memory goes
// back; and this server knows what every runner holds, so before one starts it
// makes room on the GPU: it stops runners nothing is using, the least
// recently used first, or waits for one in use to finish, until the new model
// fits — two fit side by side when they do. A request's keep_alive is this
// server's to keep: the runner holds its model until told to stop.

// SetChatRunners has chat and embedding models served in runners, each started
// with args — the serve flags that size a model. Call before serving.
func (s *Server) SetChatRunners(args []string) {
	s.chatRunners, s.chatArgs = true, args
	s.runners.room = s.makeRoomFor
}

// The chat dialects' endpoints, each a JSON body naming its model.
var chatEndpoints = []string{"/api/chat", "/api/generate", "/api/embed", "/api/embeddings",
	"/v1/chat/completions", "/v1/embeddings", "/v1/messages", "/v1/messages/count_tokens", "/v1/responses"}

// maxChatBody bounds a chat request read whole: pictures travel in it.
const maxChatBody = 256 << 20

// storeName is the store's name for the model at path.
func (s *Server) storeName(path string) string {
	if models, err := s.store.All(); err == nil {
		for _, m := range models {
			if m.Path == path {
				return m.Name
			}
		}
	}
	return modelName(path)
}

// load has a runner load its model, as Ollama's load request does — so that
// the memory it takes is taken before the next runner is sized against what
// is left, and the first request finds it loaded.
func (s *Server) load(p *proc, path string) {
	t0 := time.Now()
	body, _ := json.Marshal(map[string]any{"model": s.storeName(path), "keep_alive": -1})
	resp, err := (&http.Client{Timeout: 30 * time.Minute}).Post(p.base+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		log.Printf("runner: %s did not load: %v", modelName(path), err)
		return
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		log.Printf("runner: %s did not load: %s", modelName(path), strings.TrimSpace(string(out)))
		return
	}
	log.Printf("runner: %s loaded in %v", modelName(path), time.Since(t0).Round(time.Millisecond))
}

// ─── making room ─────────────────────────────────────────────────────────────

// gpuMemory reads what the GPU has free and its budget, the whole GPU's
// (GGML_METAL_FREE_DEVICE, set by cmd/kinfer). Tests replace it.
var gpuMemory = func() (free, total uint64, ok bool) {
	d, ok := llama.Accelerator()
	if !ok {
		return 0, 0, false
	}
	free, total = d.Free, d.Total
	// KINFER_RUNNER_BUDGET (GB) pretends the GPU's budget is smaller, what
	// everything holds now taken from it: a test of making room, with models
	// a shared machine has room for.
	if b, err := strconv.ParseFloat(os.Getenv("KINFER_RUNNER_BUDGET"), 64); err == nil && b > 0 {
		budget := uint64(b * 1e9)
		used := total - min(free, total)
		free, total = 0, min(budget, total)
		if used < total {
			free = total - used
		}
	}
	return free, total, true
}

// need is about what the model at path takes on the GPU loaded: its weights,
// and room beside them — a context and its passes for a chat model, a
// picture's activations (18.4 GB at the peak of a 1024² one against 10.8 of
// weights), a speech model's little.
func (s *Server) need(path string) uint64 {
	var weights int64
	if st, err := os.Stat(path); err == nil && st.IsDir() {
		filepath.Walk(path, func(p string, fi os.FileInfo, err error) error {
			if err == nil && strings.HasSuffix(p, ".safetensors") {
				if st, err := os.Stat(p); err == nil { // a link's target
					weights += st.Size()
				}
			}
			return nil
		})
	} else if files, err := s.store.Shards(path); err == nil {
		for _, f := range files {
			if st, err := os.Stat(f); err == nil {
				weights += st.Size()
			}
		}
	}
	extra := max(int64(2<<30), weights/10)
	switch {
	case mlximage.IsModelDir(path):
		extra = 8 << 30
	case mlxtts.IsModelDir(path), mlxasr.IsModelDir(path), mlxembed.IsModelDir(path):
		extra = 1 << 30
	}
	return uint64(weights + extra)
}

// makeRoomFor makes room on the GPU for the model at path before its runner
// starts: while the GPU has less free than the model needs (all of it, for
// one larger than the GPU), it stops the runner unused longest that nothing is
// using and is not kept, and once none is left, waits up to waitForRoom for one
// in use to finish. Whatever it cannot free, the model's own sizing makes do
// with: its context, and where its experts go.
func (s *Server) makeRoomFor(path string) {
	rs := s.runners
	need := s.need(path)
	deadline := time.Now().Add(waitForRoom)
	waited := false
	for {
		free, total, ok := gpuMemory()
		if !ok {
			return
		}
		if need > total {
			need = total
		}
		if free >= need {
			return
		}
		// The runner or the service unused longest goes (services.go).
		if s.makeRoom(modelName(path), need, path, "") {
			return
		}
		if rs.busy(path) && time.Now().Before(deadline) {
			if !waited {
				log.Printf("runner: %s needs about %s of the GPU and %s is free: waiting for a model in use to finish", modelName(path), gb(need), gb(free))
				waited = true
			}
			time.Sleep(250 * time.Millisecond)
			continue
		}
		log.Printf("runner: %s needs about %s of the GPU, %s is free and nothing more can be stopped: it sizes itself to what there is", modelName(path), gb(need), gb(free))
		return
	}
}

// waitForRoom is how long a model waits for others in use to finish before it
// starts in what is left.
var waitForRoom = 5 * time.Minute

// waitFreed waits, a few seconds at most, for the GPU to report more free
// than before: an ended process's memory comes back to the count a moment
// after it ends.
var waitFreed = func(before uint64) {
	for i := 0; i < 30; i++ {
		if free, _, ok := gpuMemory(); !ok || free > before {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func gb(n uint64) string { return fmt.Sprintf("%.1f GB", float64(n)/1e9) }

// idlest is the runner unused longest that nothing uses, is up and is not
// kept, other than path's; nil if none is.
func (rs *runnerSet) idlest(path string) *runner {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	var best *runner
	for _, r := range rs.byPath {
		if r.path == path || r.inflight > 0 || rs.keep && rs.kept[r.path] {
			continue
		}
		select {
		case <-r.ready:
		default:
			continue
		}
		if r.err != nil || r.p.exited() {
			continue
		}
		if best == nil || r.used.Before(best.used) {
			best = r
		}
	}
	return best
}

// busy reports a runner in use that could be stopped once done, other than
// path's.
func (rs *runnerSet) busy(path string) bool {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	for _, r := range rs.byPath {
		if r.path != path && r.inflight > 0 && !(rs.keep && rs.kept[r.path]) {
			return true
		}
	}
	return false
}

// stopNow stops r, which nothing uses, and waits for it to end.
func (rs *runnerSet) stopNow(r *runner) {
	rs.mu.Lock()
	if rs.byPath[r.path] != r || r.inflight > 0 {
		rs.mu.Unlock()
		return
	}
	delete(rs.byPath, r.path)
	if r.idle != nil {
		r.idle.Stop()
	}
	rs.mu.Unlock()
	r.p.stop()
}

// stopPath stops path's runner once nothing uses it — a keep_alive of 0
// with nothing to say — and reports whether there was one.
func (rs *runnerSet) stopPath(path string) bool {
	rs.mu.Lock()
	path = rs.same(path)
	r := rs.byPath[path]
	rs.mu.Unlock()
	if r == nil {
		return false
	}
	<-r.ready
	if r.err != nil {
		return false
	}
	for {
		rs.mu.Lock()
		if rs.byPath[path] != r {
			rs.mu.Unlock()
			return true
		}
		if r.inflight == 0 {
			delete(rs.byPath, path)
			if r.idle != nil {
				r.idle.Stop()
			}
			rs.mu.Unlock()
			r.p.stop()
			return true
		}
		rs.mu.Unlock()
		time.Sleep(100 * time.Millisecond)
	}
}

// up is the runner of path if it is up, or nil.
func (rs *runnerSet) upRunner(path string) *runner {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	r := rs.byPath[rs.same(path)]
	if r == nil {
		return nil
	}
	select {
	case <-r.ready:
		if r.err == nil && !r.p.exited() {
			return r
		}
	default:
	}
	return nil
}

// ─── the chat endpoints, in runners ──────────────────────────────────────────

// dialectError answers an error as the endpoint's dialect writes one.
func dialectError(w http.ResponseWriter, endpoint string, status int, msg string) {
	switch {
	case strings.HasPrefix(endpoint, "/v1/messages"):
		kind := "api_error"
		switch {
		case status == http.StatusNotFound:
			kind = "not_found_error"
		case status < 500:
			kind = "invalid_request_error"
		case status == http.StatusServiceUnavailable:
			kind = "overloaded_error"
		}
		writeAnthropicError(w, status, kind, msg)
	case strings.HasPrefix(endpoint, "/v1/"):
		writeJSON(w, status, map[string]any{"error": map[string]string{"message": msg}})
	default:
		writeJSON(w, status, map[string]string{"error": msg})
	}
}

// proxyChat answers a chat dialect's endpoint in the runner of the model the
// request names. The request's keep_alive stays here: the runner holds its
// model until this server stops it.
func (s *Server) proxyChat(endpoint string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(io.LimitReader(r.Body, maxChatBody+1))
		if err != nil {
			dialectError(w, endpoint, http.StatusBadRequest, "reading the request: "+err.Error())
			return
		}
		if len(body) > maxChatBody {
			dialectError(w, endpoint, http.StatusRequestEntityTooLarge, fmt.Sprintf("the request is larger than %d MiB", maxChatBody>>20))
			return
		}
		var top map[string]json.RawMessage
		if err := json.Unmarshal(body, &top); err != nil {
			dialectError(w, endpoint, http.StatusBadRequest, err.Error())
			return
		}
		var name string
		json.Unmarshal(top["model"], &name)
		ask, err := parseKeepAlive(top["keep_alive"])
		if err != nil {
			dialectError(w, endpoint, http.StatusBadRequest, err.Error())
			return
		}
		if _, ok := top["keep_alive"]; ok {
			delete(top, "keep_alive")
			body, _ = json.Marshal(top)
		}
		path, err := s.store.Resolve(name)
		if err != nil {
			dialectError(w, endpoint, http.StatusNotFound, err.Error())
			return
		}
		// Ollama's way of loading a model, and with keep_alive 0 of
		// unloading it: a request with nothing in it.
		loadOnly := endpoint == "/api/generate" && empty(top["prompt"]) || endpoint == "/api/chat" && empty(top["messages"])
		if loadOnly && ask != nil && *ask == 0 {
			if s.runners.stopPath(path) {
				log.Printf("runner: %s stopped as asked (keep_alive 0)", modelName(path))
			}
			unloaded(w, endpoint, name)
			return
		}
		rn, release, err := s.runners.acquireAs(path, true)
		if err != nil {
			dialectError(w, endpoint, http.StatusServiceUnavailable, fmt.Sprintf("%s could not be started: %v", name, err))
			return
		}
		defer release(ask)
		s.forward(w, r, rn, endpoint, body)
	}
}

// empty reports a JSON field absent, null, "" or [].
func empty(raw json.RawMessage) bool {
	t := strings.TrimSpace(string(raw))
	return t == "" || t == "null" || t == `""` || t == "[]"
}

// unloaded answers an unload as the endpoint answers one.
func unloaded(w http.ResponseWriter, endpoint, name string) {
	if endpoint == "/api/chat" {
		writeJSON(w, http.StatusOK, ollamaChatResponse{Model: name, CreatedAt: time.Now(),
			Message: ollamaMsg{Role: "assistant"}, Done: true, DoneReason: "unload"})
		return
	}
	writeJSON(w, http.StatusOK, ollamaGenerateResponse{Model: name, CreatedAt: time.Now(), Done: true, DoneReason: "unload"})
}

// forward sends the request, body read, to rn and its answer back as it comes.
func (s *Server) forward(w http.ResponseWriter, r *http.Request, rn *runner, endpoint string, body []byte) {
	target, _ := url.Parse(rn.p.base)
	answered := false
	proxy := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(target)
			pr.Out.Header.Set(runnerHeader, rn.path)
			if body != nil {
				pr.Out.Body = io.NopCloser(bytes.NewReader(body))
				pr.Out.ContentLength = int64(len(body))
			}
		},
		ModifyResponse: func(resp *http.Response) error {
			answered = true
			resp.Body = &cutShort{ReadCloser: resp.Body, kind: resp.Header.Get("Content-Type"), model: modelName(rn.path)}
			return nil
		},
		FlushInterval: -1, // a streamed reply goes out as it is written
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if errors.Is(err, context.Canceled) {
				return
			}
			log.Printf("runner: %s: %v", modelName(rn.path), err)
			if !answered {
				dialectError(w, endpoint, http.StatusBadGateway, fmt.Sprintf("the model %s stopped answering: %v; the next request starts it again", modelName(rn.path), err))
			}
		},
	}
	proxy.ServeHTTP(w, r)
}

// cutShort is a runner's streamed answer, which says so in its own format
// when the runner ends before it does — a reply cut short by a crash, which
// in this server's own process would have ended with an error in-band.
type cutShort struct {
	io.ReadCloser
	kind, model string
	said        bool
	rest        []byte
}

func (c *cutShort) Read(p []byte) (int, error) {
	if c.rest != nil {
		n := copy(p, c.rest)
		c.rest = c.rest[n:]
		if len(c.rest) == 0 {
			return n, io.EOF
		}
		return n, nil
	}
	n, err := c.ReadCloser.Read(p)
	if err == nil || err == io.EOF || c.said {
		return n, err
	}
	c.said = true
	msg := fmt.Sprintf("the model %s stopped answering mid-reply (%v); the next request starts it again", c.model, err)
	b, _ := json.Marshal(msg)
	switch {
	case strings.HasPrefix(c.kind, "text/event-stream"):
		c.rest = []byte(fmt.Sprintf("\n\ndata: {\"error\":{\"message\":%s}}\n\n", b))
	case strings.HasPrefix(c.kind, "application/x-ndjson"):
		c.rest = []byte(fmt.Sprintf("\n{\"error\":%s,\"done\":true}\n", b))
	default:
		return n, err
	}
	log.Printf("runner: %s", msg)
	return n, nil
}

// handleShowRunners answers /api/show here, or in the model's runner while
// one is up, which knows the context it gave each conversation.
func (s *Server) handleShowRunners(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	var req struct{ Model, Name string }
	json.Unmarshal(body, &req)
	name := req.Model
	if name == "" {
		name = req.Name
	}
	if path, err := s.store.Resolve(name); err == nil {
		if rn := s.runners.upRunner(path); rn != nil && rn.chat {
			s.forward(w, r, rn, "/api/show", body)
			return
		}
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	s.handleShow(w, r)
}

// lastChat is the chat runner used last, or nil.
func (s *Server) lastChat() *runner {
	rs := s.runners
	rs.mu.Lock()
	defer rs.mu.Unlock()
	var best *runner
	for _, r := range rs.byPath {
		if !r.chat {
			continue
		}
		select {
		case <-r.ready:
		default:
			continue
		}
		if r.err == nil && !r.p.exited() && (best == nil || r.used.After(best.used) || r.inflight > 0) {
			best = r
		}
	}
	return best
}

// handlePropsRunners answers /props for the chat model used last.
func (s *Server) handlePropsRunners(w http.ResponseWriter, r *http.Request) {
	if rn := s.lastChat(); rn != nil {
		s.forward(w, r, rn, "/props", nil)
		return
	}
	s.handleProps(w, r)
}

// runnerGet reads one of a runner's endpoints.
func runnerGet(rn *runner, path string) ([]byte, error) {
	resp, err := (&http.Client{Timeout: 5 * time.Second}).Get(rn.p.base + path)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

// chatRunning are the chat runners up.
func (s *Server) chatRunning() []*runner {
	rs := s.runners
	rs.mu.Lock()
	defer rs.mu.Unlock()
	var out []*runner
	for _, r := range rs.byPath {
		if !r.chat {
			continue
		}
		select {
		case <-r.ready:
			if r.err == nil && !r.p.exited() {
				out = append(out, r)
			}
		default:
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].path < out[j].path })
	return out
}

// runnerPS is what /api/ps says of the chat runners: each one's own entry,
// expiring when this server will stop it.
func (s *Server) runnerPS() []map[string]any {
	var out []map[string]any
	for _, rn := range s.chatRunning() {
		b, err := runnerGet(rn, "/api/ps")
		if err != nil {
			continue
		}
		var ps struct {
			Models []map[string]any `json:"models"`
		}
		json.Unmarshal(b, &ps)
		s.runners.mu.Lock()
		expiry := rn.expiry
		if rn.inflight > 0 || rn.idle == nil {
			expiry = time.Time{}
		}
		s.runners.mu.Unlock()
		for _, m := range ps.Models {
			delete(m, "expires_at")
			if !expiry.IsZero() {
				m["expires_at"] = expiry
			}
			out = append(out, m)
		}
	}
	return out
}

// runnerMetrics are the chat runners' metrics, each line labelled with its
// model, each metric's HELP and TYPE once; the requests each answered are
// counted here, as this server answered them.
func (s *Server) runnerMetrics() string {
	var b strings.Builder
	seen := map[string]bool{}
	for _, rn := range s.chatRunning() {
		text, err := runnerGet(rn, "/metrics")
		if err != nil {
			continue
		}
		label := fmt.Sprintf("model=%q", s.storeName(rn.path))
		sc := bufio.NewScanner(bytes.NewReader(text))
		for sc.Scan() {
			line := sc.Text()
			if strings.HasPrefix(line, "kinfer_requests_total") || strings.HasPrefix(line, "kinfer_model_loaded") {
				continue
			}
			if strings.HasPrefix(line, "# ") {
				f := strings.Fields(line)
				if len(f) >= 3 && (strings.HasSuffix(f[2], "requests_total") || strings.HasSuffix(f[2], "model_loaded")) {
					continue
				}
				if !seen[line] {
					seen[line] = true
					b.WriteString(line + "\n")
				}
				continue
			}
			name, rest, ok := strings.Cut(line, " ")
			if !ok {
				continue
			}
			if i := strings.Index(name, "{"); i >= 0 {
				name = name[:i+1] + label + "," + name[i+1:]
			} else {
				name += "{" + label + "}"
			}
			b.WriteString(name + " " + rest + "\n")
		}
		fmt.Fprintf(&b, "kinfer_model_loaded{%s} 1\n", label)
	}
	return b.String()
}
