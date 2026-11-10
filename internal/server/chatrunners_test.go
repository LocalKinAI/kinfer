package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// chatFakes stand in for chat models' runners: each answers what it was sent,
// and blocks a reply while hold is set for its model.
type chatFakes struct {
	mu      sync.Mutex
	started map[string]int
	live    map[string]*proc
	bodies  []map[string]any
	hold    map[string]chan struct{}
	drop    map[string]bool   // close the connection without an answer
	cut     map[string]string // the Content-Type of a reply cut short
}

func newChatFakes() *chatFakes {
	return &chatFakes{started: map[string]int{}, live: map[string]*proc{}, hold: map[string]chan struct{}{}, drop: map[string]bool{}, cut: map[string]string{}}
}

func (f *chatFakes) start(path string, chat bool) (*proc, error) {
	name := strings.TrimSuffix(filepath.Base(path), ".gguf")
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/ps":
			writeJSON(w, 200, map[string]any{"models": []any{map[string]any{"name": name, "model": name, "size": 123,
				"context_length": 4096, "expires_at": "2099-01-01T00:00:00Z"}}})
			return
		case "/metrics":
			io.WriteString(w, "# HELP kinfer_queue_depth q.\n# TYPE kinfer_queue_depth gauge\nkinfer_queue_depth 0\nkinfer_requests_total{outcome=\"ok\"} 9\n")
			return
		}
		f.mu.Lock()
		hold, drop := f.hold[name], f.drop[name]
		f.mu.Unlock()
		if drop {
			c, _, _ := w.(http.Hijacker).Hijack()
			c.Close()
			return
		}
		f.mu.Lock()
		cut := f.cut[name]
		f.mu.Unlock()
		if cut != "" { // two pieces of a streamed reply, then the process dies
			w.Header().Set("Content-Type", cut)
			io.WriteString(w, `{"message":{"content":"1"}}`+"\n"+`{"message":{"content":"2"}}`+"\n")
			w.(http.Flusher).Flush()
			c, _, _ := w.(http.Hijacker).Hijack()
			c.Close()
			return
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		f.mu.Lock()
		f.bodies = append(f.bodies, body)
		f.mu.Unlock()
		if hold != nil {
			<-hold
		}
		writeJSON(w, 200, map[string]any{"runner": name, "path": r.URL.Path, "header": r.Header.Get(runnerHeader)})
	}))
	done := make(chan struct{})
	var once sync.Once
	p := &proc{base: ts.URL, done: done}
	p.stop = func() { once.Do(func() { ts.CloseClientConnections(); ts.Close(); close(done) }) }
	f.mu.Lock()
	f.started[name]++
	f.live[name] = p
	f.mu.Unlock()
	return p, nil
}

func (f *chatFakes) count(name string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.started[name]
}

func (f *chatFakes) alive(name string) bool {
	f.mu.Lock()
	p := f.live[name]
	f.mu.Unlock()
	return p != nil && !p.exited()
}

// chatServer is a front server with chat runners, its models alpha, beta and
// gamma, its runners fakes, and a GPU of room GB of which each runner takes
// what need says (2 GiB, the least, for these one-byte models).
func chatServer(t *testing.T, room float64) (*Server, *chatFakes) {
	t.Helper()
	srv, _ := newTestServer(t, "alpha", "beta", "gamma")
	f := newChatFakes()
	srv.SetChatRunners(nil)
	srv.runners.start = f.start
	old, oldFreed, oldWait := gpuMemory, waitFreed, waitForRoom
	gpuMemory = func() (uint64, uint64, bool) {
		used := 0
		for _, n := range []string{"alpha", "beta", "gamma"} {
			if f.alive(n) {
				used++
			}
		}
		total := uint64(room * (1 << 30))
		u := uint64(used) * (2 << 30)
		if u > total {
			return 0, total, true
		}
		return total - u, total, true
	}
	waitFreed = func(uint64) {}
	waitForRoom = 2 * time.Second
	t.Cleanup(func() { gpuMemory, waitFreed, waitForRoom = old, oldFreed, oldWait; srv.Close() })
	return srv, f
}

// Each dialect's request goes to the runner of the model it names, the
// keep_alive kept here; a model the store has not is refused in the
// dialect's own words, and nothing starts.
func TestChatGoesToItsRunner(t *testing.T) {
	srv, f := chatServer(t, 100)
	for path, body := range map[string]string{
		"/api/chat":            `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"keep_alive":"1h"}`,
		"/v1/chat/completions": `{"model":"alpha","messages":[{"role":"user","content":"hi"}]}`,
		"/v1/messages":         `{"model":"alpha","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}`,
		"/v1/responses":        `{"model":"alpha","input":"hi"}`,
		"/api/embed":           `{"model":"beta","input":"hi"}`,
	} {
		code, out, _ := do(t, srv, "POST", path, "application/json", body)
		want := "alpha"
		if path == "/api/embed" {
			want = "beta"
		}
		if code != 200 || out["runner"] != want || out["path"] != path || !strings.HasSuffix(out["header"].(string), want+".gguf") {
			t.Errorf("%s: %d %v", path, code, out)
		}
	}
	if f.count("alpha") != 1 || f.count("beta") != 1 {
		t.Errorf("started %v", f.started)
	}
	for _, b := range f.bodies {
		if _, ok := b["keep_alive"]; ok {
			t.Errorf("keep_alive reached the runner: %v", b)
		}
	}
	// Not in the store: refused as each dialect refuses.
	_, out, _ := do(t, srv, "POST", "/api/chat", "application/json", `{"model":"nope","messages":[]}`)
	if _, ok := out["error"].(string); !ok {
		t.Errorf("ollama: %v", out)
	}
	code, out, _ := do(t, srv, "POST", "/v1/chat/completions", "application/json", `{"model":"nope"}`)
	if e, _ := out["error"].(map[string]any); code != 404 || e["message"] == nil {
		t.Errorf("openai: %d %v", code, out)
	}
	if code, out, _ = do(t, srv, "POST", "/v1/messages", "application/json", `{"model":"nope"}`); code != 404 || out["type"] != "error" {
		t.Errorf("anthropic: %d %v", code, out)
	}
	if f.count("nope") != 0 {
		t.Error("a runner started for a model that is not here")
	}

	// /api/ps: each runner's own entry, expiring when this server stops it.
	_, ps, _ := do(t, srv, "GET", "/api/ps", "", "")
	var names []string
	for _, m := range ps["models"].([]any) {
		e := m.(map[string]any)
		names = append(names, e["name"].(string))
		if e["name"] == "alpha" && e["context_length"] != float64(4096) {
			t.Errorf("ps lost the runner's context: %v", e)
		}
	}
	if strings.Join(names, ",") != "alpha,beta" {
		t.Errorf("ps: %v", ps)
	}
	// /metrics: the runners', each labelled; requests counted here.
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	m := w.Body.String()
	if !strings.Contains(m, `kinfer_queue_depth{model="alpha"} 0`) || strings.Count(m, "# TYPE kinfer_queue_depth") != 1 || strings.Contains(m, "} 9\n") {
		t.Errorf("metrics:\n%s", m)
	}
}

// keep_alive 0 stops the runner once the request is done; a request with
// nothing in it and keep_alive 0 stops it at once, answered as an unload.
func TestRunnerKeepAlive(t *testing.T) {
	srv, f := chatServer(t, 100)
	do(t, srv, "POST", "/api/chat", "application/json", `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"keep_alive":0}`)
	for i := 0; i < 50 && f.alive("alpha"); i++ {
		time.Sleep(20 * time.Millisecond)
	}
	if f.alive("alpha") {
		t.Error("keep_alive 0 left the runner up")
	}
	do(t, srv, "POST", "/api/chat", "application/json", `{"model":"beta","messages":[{"role":"user","content":"hi"}]}`)
	code, out, _ := do(t, srv, "POST", "/api/generate", "application/json", `{"model":"beta","keep_alive":0}`)
	if code != 200 || out["done_reason"] != "unload" || f.alive("beta") {
		t.Errorf("unload: %d %v alive=%v", code, out, f.alive("beta"))
	}
	// A request with nothing in it otherwise loads: the runner starts.
	if code, _, _ = do(t, srv, "POST", "/api/generate", "application/json", `{"model":"gamma"}`); code != 200 || !f.alive("gamma") {
		t.Errorf("load: %d", code)
	}
}

// With room for two, a third model stops the one unused longest that nothing
// is using; one in use is waited for, then stopped.
func TestRunnersMakeRoom(t *testing.T) {
	srv, f := chatServer(t, 5) // two of 2 GiB fit, not three
	chat := func(model string) int {
		code, _, _ := do(t, srv, "POST", "/api/chat", "application/json", `{"model":"`+model+`","messages":[{"role":"user","content":"hi"}]}`)
		return code
	}
	chat("alpha")
	time.Sleep(10 * time.Millisecond)
	chat("beta")
	if !f.alive("alpha") || !f.alive("beta") {
		t.Fatal("two that fit did not both stay")
	}
	chat("gamma")
	if f.alive("alpha") || !f.alive("beta") || !f.alive("gamma") {
		t.Errorf("after gamma: alpha %v beta %v gamma %v; alpha, unused longest, should have gone",
			f.alive("alpha"), f.alive("beta"), f.alive("gamma"))
	}

	// beta and gamma both busy: alpha waits for one to finish.
	hb, hg := make(chan struct{}), make(chan struct{})
	f.mu.Lock()
	f.hold["beta"], f.hold["gamma"] = hb, hg
	f.mu.Unlock()
	var wg sync.WaitGroup
	for _, m := range []string{"beta", "gamma"} {
		wg.Add(1)
		go func() { defer wg.Done(); chat(m) }()
	}
	time.Sleep(100 * time.Millisecond)
	got := make(chan int)
	go func() { got <- chat("alpha") }()
	time.Sleep(300 * time.Millisecond)
	if f.alive("alpha") {
		t.Fatal("alpha started while both others were in use and there was no room")
	}
	close(hb) // beta finishes, and goes
	if code := <-got; code != 200 || f.alive("beta") || !f.alive("alpha") {
		t.Errorf("alpha %d; beta alive %v, alpha alive %v", code, f.alive("beta"), f.alive("alpha"))
	}
	close(hg)
	wg.Wait()
}

// A runner that dies mid-request: the request is answered 502 in its dialect,
// and the next one starts the model again.
func TestRunnerThatDies(t *testing.T) {
	srv, f := chatServer(t, 100)
	do(t, srv, "POST", "/api/chat", "application/json", `{"model":"alpha","messages":[{"role":"user","content":"hi"}]}`)
	f.mu.Lock()
	f.drop["alpha"] = true
	f.mu.Unlock()
	code, out, _ := do(t, srv, "POST", "/v1/chat/completions", "application/json", `{"model":"alpha","messages":[{"role":"user","content":"hi"}]}`)
	if e, _ := out["error"].(map[string]any); code != 502 || !strings.Contains(e["message"].(string), "stopped answering") {
		t.Errorf("%d %v", code, out)
	}
	f.mu.Lock()
	f.drop["alpha"] = false
	p := f.live["alpha"]
	f.mu.Unlock()
	p.stop() // it crashed
	time.Sleep(20 * time.Millisecond)
	if code, _, _ := do(t, srv, "POST", "/api/chat", "application/json", `{"model":"alpha","messages":[{"role":"user","content":"hi"}]}`); code != 200 || f.count("alpha") != 2 {
		t.Errorf("after it died: %d, started %d", code, f.count("alpha"))
	}
}

// A streamed reply cut short by its runner's end ends with an error in its
// own format, as one that failed in this process would.
func TestStreamCutShort(t *testing.T) {
	srv, f := chatServer(t, 100)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	for kind, want := range map[string]string{
		"application/x-ndjson": `{"error":"the model alpha stopped answering mid-reply`,
		"text/event-stream":    `data: {"error":{"message":"the model alpha stopped answering mid-reply`,
	} {
		f.mu.Lock()
		f.cut["alpha"] = kind
		f.mu.Unlock()
		resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"model":"alpha","messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if !strings.Contains(string(b), `{"message":{"content":"2"}}`) || !strings.Contains(string(b), want) {
			t.Errorf("%s: %q", kind, b)
		}
	}
}
