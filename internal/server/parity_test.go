package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// Parity with Ollama on three of the four places where kinfer answered
// differently and a caller paid for it (2026-09-24, a night of deep-study runs
// on the box): thinking, the default token budget, and unloading a model on
// request. The fourth, -prefix, is tested in cmd/kinfer and engine.

func call(t *testing.T, srv *Server, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
	return rec
}

func (f *fakeEngine) lastParams() engine.GenParams {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.params
}

func waitFor(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !ok() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// A reasoning model's reply: working out in a think block, then the answer.
func thinkingEngine(path string) *fakeEngine {
	return &fakeEngine{path: path, frags: []string{"<think>", "carry the one", "</think>", "391"}}
}

// ─── thinking ────────────────────────────────────────────────────────────────

// Absent think is on, and the thinking comes back in its own field — Ollama's
// behaviour, measured. Before, kinfer let the model think and dropped the
// thinking, and a reply that ran out of time came back empty with nothing to
// say why.
func TestThinkingComesBackUnlessTurnedOff(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()
	var eng *fakeEngine
	srv.open = func(p string, _ engine.Options) (generator, error) {
		eng = thinkingEngine(p)
		return eng, nil
	}

	for _, c := range []struct {
		think, wantThinking string
		wantNoThink         bool
	}{
		{"", "carry the one", false},
		{`,"think":true`, "carry the one", false},
		{`,"think":"high"`, "carry the one", false},
		{`,"think":false`, "", true},
	} {
		rec := call(t, srv, "/api/chat",
			`{"model":"alpha","stream":false,"messages":[{"role":"user","content":"17*23?"}]`+c.think+`}`)
		var r ollamaChatResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &r); err != nil {
			t.Fatalf("think%q: %v: %s", c.think, err, rec.Body.String())
		}
		if got := strings.TrimSpace(r.Message.Thinking); got != c.wantThinking {
			t.Errorf("think%q: message.thinking = %q, want %q", c.think, got, c.wantThinking)
		}
		if got := strings.TrimSpace(r.Message.Content); got != "391" {
			t.Errorf("think%q: content = %q, want only the answer", c.think, got)
		}
		if got := eng.lastParams().NoThink; got != c.wantNoThink {
			t.Errorf("think%q: engine NoThink = %v, want %v", c.think, got, c.wantNoThink)
		}
	}
}

// Streamed, the thinking travels chunk by chunk in message.thinking, never in
// content — the same shape Ollama streams.
func TestStreamedThinkingTravelsInItsOwnField(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()
	srv.open = func(p string, _ engine.Options) (generator, error) { return thinkingEngine(p), nil }

	rec := call(t, srv, "/api/chat", `{"model":"alpha","messages":[{"role":"user","content":"17*23?"}]}`)
	var thinking, content strings.Builder
	for _, line := range strings.Split(strings.TrimSpace(rec.Body.String()), "\n") {
		var r ollamaChatResponse
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("bad chunk %q: %v", line, err)
		}
		if strings.Contains(r.Message.Content, "carry") {
			t.Errorf("thinking leaked into content: %q", r.Message.Content)
		}
		thinking.WriteString(r.Message.Thinking)
		content.WriteString(r.Message.Content)
	}
	if strings.TrimSpace(thinking.String()) != "carry the one" || strings.TrimSpace(content.String()) != "391" {
		t.Errorf("streamed thinking %q, content %q; want the working out and the answer apart",
			thinking.String(), content.String())
	}
}

// The OpenAI dialect has no standard field: Ollama's /v1 writes "reasoning",
// DeepSeek's API "reasoning_content". Both, and by default.
func TestOpenAIDialectReturnsReasoningUnderBothNames(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()
	srv.open = func(p string, _ engine.Options) (generator, error) { return thinkingEngine(p), nil }

	for _, c := range []struct {
		think string
		want  string
	}{{"", "carry the one"}, {`,"think":false`, ""}} {
		rec := call(t, srv, "/v1/chat/completions",
			`{"model":"alpha","messages":[{"role":"user","content":"17*23?"}]`+c.think+`}`)
		var out struct {
			Choices []struct {
				Message map[string]any `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil || len(out.Choices) != 1 {
			t.Fatalf("think%q: %v: %s", c.think, err, rec.Body.String())
		}
		msg := out.Choices[0].Message
		for _, key := range []string{"reasoning", "reasoning_content"} {
			got, _ := msg[key].(string)
			if strings.TrimSpace(got) != c.want {
				t.Errorf("think%q: %s = %q, want %q", c.think, key, got, c.want)
			}
		}
		if got, _ := msg["content"].(string); strings.TrimSpace(got) != "391" {
			t.Errorf("think%q: content = %q, want only the answer", c.think, got)
		}
	}

	rec := call(t, srv, "/v1/chat/completions",
		`{"model":"alpha","stream":true,"messages":[{"role":"user","content":"17*23?"}]}`)
	if !strings.Contains(rec.Body.String(), `"reasoning":"carry the one"`) ||
		!strings.Contains(rec.Body.String(), `"reasoning_content":"carry the one"`) {
		t.Errorf("the stream carried no reasoning delta under both names:\n%s", rec.Body.String())
	}
}

// ─── token budget ────────────────────────────────────────────────────────────

// A caller that names no budget gets none: until the model stops or its slot
// is full, as in Ollama. The 512 default cut card-writing replies off
// mid-JSON, which then parsed as zero cards.
func TestNoBudgetMeansNoCeiling(t *testing.T) {
	srv, reg := newTestServer(t, "alpha")
	defer srv.Close()

	for _, c := range []struct {
		path, body string
		want       int
	}{
		{"/api/chat", `{"model":"alpha","stream":false,"messages":[{"role":"user","content":"hi"}]}`, 0},
		{"/api/chat", `{"model":"alpha","stream":false,"messages":[{"role":"user","content":"hi"}],"options":{"num_predict":64}}`, 64},
		{"/v1/chat/completions", `{"model":"alpha","messages":[{"role":"user","content":"hi"}]}`, 0},
		{"/v1/chat/completions", `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"max_tokens":32}`, 32},
	} {
		if rec := call(t, srv, c.path, c.body); rec.Code != http.StatusOK {
			t.Fatalf("%s: status %d: %s", c.path, rec.Code, rec.Body.String())
		}
		if got := reg.get("alpha").lastParams().MaxTokens; got != c.want {
			t.Errorf("%s %s: engine MaxTokens = %d, want %d", c.path, c.body, got, c.want)
		}
	}
}

// ─── keep_alive ──────────────────────────────────────────────────────────────

func TestParseKeepAliveReadsWhatOllamaSends(t *testing.T) {
	forever := time.Duration(-1)
	for raw, want := range map[string]*time.Duration{
		``:        nil,
		`null`:    nil,
		`0`:       ptr(0),
		`"0"`:     ptr(0),
		`"10s"`:   ptr(10 * time.Second),
		`"5m"`:    ptr(5 * time.Minute),
		`300`:     ptr(300 * time.Second),
		`"300"`:   ptr(300 * time.Second),
		`1.5`:     ptr(1500 * time.Millisecond),
		`-1`:      &forever,
		`"-1m"`:   &forever,
		`"-1"`:    &forever,
		`1e300`:   &forever,
		`"never"`: nil, // an error, below
	} {
		got, err := parseKeepAlive(json.RawMessage(raw))
		if raw == `"never"` {
			if err == nil {
				t.Errorf("parseKeepAlive(%s) accepted it", raw)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseKeepAlive(%s): %v", raw, err)
			continue
		}
		if (got == nil) != (want == nil) || (got != nil && *got != *want) {
			t.Errorf("parseKeepAlive(%s) = %v, want %v", raw, show(got), show(want))
		}
	}
	if _, err := parseKeepAlive(json.RawMessage(`true`)); err == nil {
		t.Error("parseKeepAlive(true) accepted a boolean")
	}
}

func ptr(d time.Duration) *time.Duration { return &d }

func show(d *time.Duration) string {
	if d == nil {
		return "nil"
	}
	return d.String()
}

// `ollama stop`'s request: /api/generate, no prompt, keep_alive 0. The model
// leaves memory before the answer comes back, so a caller can load something
// else the moment it hears.
func TestGenerateWithKeepAliveZeroUnloads(t *testing.T) {
	srv, reg := newTestServer(t, "alpha")
	defer srv.Close()

	call(t, srv, "/api/chat", `{"model":"alpha","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	f := reg.get("alpha")

	rec := call(t, srv, "/api/generate", `{"model":"alpha","keep_alive":0}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var r ollamaGenerateResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if !r.Done || r.DoneReason != "unload" || r.Response != "" {
		t.Errorf("got %+v, want Ollama's done/unload with an empty response", r)
	}
	if !f.isClosed() {
		t.Error("the model was still open after the unload answered")
	}

	ps := httptest.NewRecorder()
	srv.Handler().ServeHTTP(ps, httptest.NewRequest(http.MethodGet, "/api/ps", nil))
	if !strings.Contains(ps.Body.String(), `"models":[]`) {
		t.Errorf("/api/ps after unloading: %s", ps.Body.String())
	}
}

// No prompt and no keep_alive loads ahead of use, and /api/chat with no
// messages does the same two things — Ollama answers both.
func TestLoadAndUnloadWithoutAPrompt(t *testing.T) {
	srv, reg := newTestServer(t, "alpha")
	defer srv.Close()

	for _, c := range []struct{ path, body, reason string }{
		{"/api/generate", `{"model":"alpha"}`, "load"},
		{"/api/chat", `{"model":"alpha","messages":[],"keep_alive":0}`, "unload"},
		{"/api/chat", `{"model":"alpha","messages":[]}`, "load"},
		{"/api/generate", `{"model":"alpha","keep_alive":"0"}`, "unload"},
	} {
		rec := call(t, srv, c.path, c.body)
		if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"done_reason":"`+c.reason+`"`) {
			t.Errorf("%s %s: status %d, want done_reason %s: %s", c.path, c.body, rec.Code, c.reason, rec.Body.String())
		}
	}
	if n := reg.loadCount(); n != 2 {
		t.Errorf("%d loads, want 2: one per load request", n)
	}
	srv.mu.Lock()
	resident := srv.cur != nil
	srv.mu.Unlock()
	if resident {
		t.Error("a model is still resident after the last unload")
	}
}

// An unload waits for a request still generating on the model — freeing the
// engine under it crashes the process — and while it waits, a request for the
// same model must not load a second copy beside it: memory for one copy is
// exactly what the caller is freeing.
func TestUnloadWaitsForTheRequestStillGenerating(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()

	// Opened however the test ends. A failure that returns before opening it
	// left the first request generating forever, and the deferred srv.Close
	// waiting on it: one failed assertion became a ten-minute hang.
	gate := make(chan struct{})
	openGate := sync.OnceFunc(func() { close(gate) })
	defer openGate()
	var mu sync.Mutex
	var engines []*fakeEngine
	srv.open = func(p string, _ engine.Options) (generator, error) {
		mu.Lock()
		defer mu.Unlock()
		for _, e := range engines {
			if !e.isClosed() {
				t.Error("a second copy loaded while the first was still open")
			}
		}
		f := &fakeEngine{path: p, frags: []string{"ok"}}
		if len(engines) == 0 {
			f.release = gate
		}
		engines = append(engines, f)
		return f, nil
	}
	first := func() *fakeEngine {
		mu.Lock()
		defer mu.Unlock()
		if len(engines) == 0 {
			return nil
		}
		return engines[0]
	}

	chatDone := make(chan struct{})
	go func() {
		call(t, srv, "/api/chat", `{"model":"alpha","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
		close(chatDone)
	}()
	waitFor(t, "the first request to start generating", func() bool {
		f := first()
		return f != nil && f.generating()
	})

	unloaded := make(chan *httptest.ResponseRecorder, 1)
	go func() { unloaded <- call(t, srv, "/api/generate", `{"model":"alpha","keep_alive":0}`) }()
	waitFor(t, "the unload to start waiting", func() bool {
		srv.mu.Lock()
		defer srv.mu.Unlock()
		return srv.swapping && srv.cur == nil
	})

	second := make(chan struct{})
	go func() {
		call(t, srv, "/api/chat", `{"model":"alpha","stream":false,"messages":[{"role":"user","content":"again"}]}`)
		close(second)
	}()

	select {
	case <-unloaded:
		t.Fatal("the unload answered while a request was still generating on the model")
	case <-time.After(100 * time.Millisecond):
	}

	openGate()
	<-chatDone
	if rec := <-unloaded; !strings.Contains(rec.Body.String(), `"done_reason":"unload"`) {
		t.Errorf("unload answered %s", rec.Body.String())
	}
	<-second
	if !first().isClosed() {
		t.Error("the unloaded engine was never closed")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(engines) != 2 {
		t.Errorf("%d loads, want 2: the unloaded copy and a fresh one for the later request", len(engines))
	}
}

// keep_alive on an ordinary request governs what follows it, and 0 means
// unload once it is done — even on a server whose -keepalive keeps models
// forever. The flag's zero and the field's zero mean opposite things; each
// keeps its own.
func TestRequestKeepAliveZeroUnloadsAfterIt(t *testing.T) {
	srv, reg := newTestServer(t, "alpha") // -keepalive 0: forever
	defer srv.Close()

	rec := call(t, srv, "/api/chat",
		`{"model":"alpha","stream":false,"keep_alive":0,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	waitFor(t, "the model to unload after a keep_alive 0 request", func() bool { return reg.get("alpha").isClosed() })
}

func TestRequestKeepAliveOverridesTheFlag(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()
	srv.SetKeepAlive(time.Hour)

	expires := func() *time.Time {
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/ps", nil))
		var out struct {
			Models []struct {
				ExpiresAt *time.Time `json:"expires_at"`
			} `json:"models"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil || len(out.Models) != 1 {
			t.Fatalf("/api/ps: %v: %s", err, rec.Body.String())
		}
		return out.Models[0].ExpiresAt
	}
	chat := func(keepAlive string) {
		call(t, srv, "/api/chat",
			`{"model":"alpha","stream":false`+keepAlive+`,"messages":[{"role":"user","content":"hi"}]}`)
	}

	chat(`,"keep_alive":-1`)
	if e := expires(); e != nil {
		t.Errorf("keep_alive -1: expires at %v, want never", e)
	}
	chat(`,"keep_alive":"10s"`)
	if e := expires(); e == nil || time.Until(*e) > 11*time.Second || time.Until(*e) < 5*time.Second {
		t.Errorf("keep_alive 10s: expires at %v, want about ten seconds from now", e)
	}
	chat(``)
	if e := expires(); e == nil || time.Until(*e) < 59*time.Minute {
		t.Errorf("no keep_alive: expires at %v, want the server's hour", e)
	}
}

func TestGenerateRefusesWhatItDoesNotServe(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()

	for _, c := range []struct {
		body string
		code int
	}{
		{`{"model":"alpha","prompt":"hello"}`, http.StatusNotImplemented},
		{`{"model":"nope","keep_alive":0}`, http.StatusNotFound},
		{`{"model":"alpha","keep_alive":"soon"}`, http.StatusBadRequest},
	} {
		if rec := call(t, srv, "/api/generate", c.body); rec.Code != c.code {
			t.Errorf("%s: status %d, want %d: %s", c.body, rec.Code, c.code, rec.Body.String())
		}
	}
}
