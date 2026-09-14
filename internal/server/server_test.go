package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/store"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// fakeEngine stands in for a loaded model. Chat blocks until release is closed,
// which is what lets a test hold a model "mid-generation" while another request
// asks for a different one.
type fakeEngine struct {
	path    string
	err     error
	frags   []string
	release chan struct{}

	tools     bool
	truncated bool
	broken    bool

	// What ChatFull reports as measurements. The eval count is not among them:
	// it is counted from the fragments actually emitted, so a handler that
	// invents a number or drops the real one fails here instead of agreeing
	// with a constant.
	promptTokens int
	promptEval   time.Duration
	evalDur      time.Duration

	mu     sync.Mutex
	closed bool
	inChat bool
}

func (f *fakeEngine) Chat(ctx context.Context, _ []chat.Message, _ engine.GenParams, onToken func(string)) (string, error) {
	f.mu.Lock()
	if f.closed {
		f.mu.Unlock()
		// The real engine segfaults here. Returning an error makes the failure
		// observable instead of fatal, so a broken test reports rather than
		// takes down the test binary.
		return "", errors.New("USE AFTER CLOSE")
	}
	f.inChat = true
	f.mu.Unlock()

	defer func() {
		f.mu.Lock()
		f.inChat = false
		f.mu.Unlock()
	}()

	if f.release != nil {
		select {
		case <-f.release:
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}

	f.mu.Lock()
	closed := f.closed
	f.mu.Unlock()
	if closed {
		return "", errors.New("USE AFTER CLOSE")
	}

	var out strings.Builder
	for _, frag := range f.frags {
		if onToken != nil {
			onToken(frag)
		}
		out.WriteString(frag)
	}
	return out.String(), f.err
}

func (f *fakeEngine) Broken() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.broken
}

func (f *fakeEngine) ChatFull(ctx context.Context, m []chat.Message, p engine.GenParams, onToken func(string)) (string, engine.Stats, error) {
	generated := 0
	text, err := f.Chat(ctx, m, p, func(frag string) {
		generated++
		if onToken != nil {
			onToken(frag)
		}
	})
	return text, engine.Stats{
		Truncated:          f.truncated,
		PromptTokens:       f.promptTokens,
		EvalTokens:         generated,
		PromptEvalDuration: f.promptEval,
		EvalDuration:       f.evalDur,
	}, err
}

func (f *fakeEngine) ToolFormat() tools.Format {
	if f.tools {
		return tools.Hermes
	}
	return tools.None
}

func (f *fakeEngine) Close() {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.inChat {
		panic("engine closed while a generation was in flight: " + f.path)
	}
	f.closed = true
}

func (f *fakeEngine) generating() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.inChat
}

func (f *fakeEngine) isClosed() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.closed
}

// registry records the fake engines a server handed out. The server loads from
// its own goroutines, so every access is guarded.
type registry struct {
	mu     sync.Mutex
	byName map[string]*fakeEngine
	loads  int
}

func (r *registry) put(f *fakeEngine) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.byName[modelName(f.path)] = f
	r.loads++
}

func (r *registry) get(name string) *fakeEngine {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.byName[name]
}

func (r *registry) loadCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.loads
}

// newTestServer returns a server over a store containing the named models,
// plus the registry of fake engines it will hand out.
func newTestServer(t *testing.T, names ...string) (*Server, *registry) {
	t.Helper()

	dir := t.TempDir()
	for _, n := range names {
		if err := os.WriteFile(filepath.Join(dir, n+".gguf"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	st, err := store.OpenAt(dir)
	if err != nil {
		t.Fatal(err)
	}

	reg := &registry{byName: map[string]*fakeEngine{}}
	srv := New(st, engine.Options{})
	srv.open = func(path string, _ engine.Options) (generator, error) {
		f := &fakeEngine{path: path, frags: []string{"ok"}}
		reg.put(f)
		return f, nil
	}
	return srv, reg
}

// TestSwapWaitsForInFlightGeneration is the regression test for the crash that
// killed the whole process: request A held a model, request B asked for a
// different one, and B freed A's llama.cpp context out from under it. The next
// llama_decode ran against a NULL context.
func TestSwapWaitsForInFlightGeneration(t *testing.T) {
	srv, reg := newTestServer(t, "alpha", "beta")
	defer srv.Close()

	hold := make(chan struct{})
	srv.open = func(path string, _ engine.Options) (generator, error) {
		f := &fakeEngine{path: path, frags: []string{"ok"}}
		if modelName(path) == "alpha" {
			f.release = hold // alpha stays mid-generation until the test says so
		}
		reg.put(f)
		return f, nil
	}

	// A starts generating with alpha and does not finish.
	aDone := make(chan error, 1)
	go func() {
		eng, release, err := srv.acquire("alpha")
		if err != nil {
			aDone <- err
			return
		}
		defer release()
		_, err = eng.Chat(context.Background(), nil, engine.DefaultGenParams(), nil)
		aDone <- err
	}()

	// Wait for alpha to be resident and generating.
	deadline := time.Now().Add(2 * time.Second)
	for {
		if e := reg.get("alpha"); e != nil && e.generating() {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("alpha never started generating")
		}
		time.Sleep(5 * time.Millisecond)
	}

	// B asks for a different model. It must not free alpha while A is running.
	bDone := make(chan error, 1)
	go func() {
		eng, release, err := srv.acquire("beta")
		if err != nil {
			bDone <- err
			return
		}
		defer release()
		_, err = eng.Chat(context.Background(), nil, engine.DefaultGenParams(), nil)
		bDone <- err
	}()

	// Give B a chance to do the wrong thing.
	time.Sleep(100 * time.Millisecond)
	if reg.get("alpha").isClosed() {
		t.Fatal("alpha was freed while a generation was still in flight (this is the segfault)")
	}
	if reg.get("beta") != nil {
		t.Fatal("beta was loaded before alpha was released — two models resident at once")
	}

	close(hold) // let A finish

	if err := <-aDone; err != nil {
		t.Fatalf("request A failed: %v", err)
	}
	select {
	case err := <-bDone:
		if err != nil {
			t.Fatalf("request B failed: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("request B never completed — the swap deadlocked")
	}
	if !reg.get("alpha").isClosed() {
		t.Error("alpha should have been freed once its last reference went away")
	}
}

// TestConcurrentSameModelSharesOneLoad checks that N requests for one model do
// not each load their own copy.
func TestConcurrentSameModelSharesOneLoad(t *testing.T) {
	srv, reg := newTestServer(t, "alpha")
	defer srv.Close()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			eng, release, err := srv.acquire("alpha")
			if err != nil {
				t.Error(err)
				return
			}
			defer release()
			if _, err := eng.Chat(context.Background(), nil, engine.DefaultGenParams(), nil); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()

	if n := reg.loadCount(); n != 1 {
		t.Errorf("loaded the same model %d times, want 1", n)
	}
}

// TestStreamingErrorIsReported is the regression test for the silent empty
// reply: a generation failure used to arrive as 200 OK with an empty message
// and no error anywhere, which is indistinguishable from a model that chose to
// say nothing.
func TestStreamingErrorIsReported(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()
	srv.open = func(path string, _ engine.Options) (generator, error) {
		return &fakeEngine{path: path, err: errors.New("prompt is 2415 tokens but the context holds 512")}, nil
	}

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (headers are already sent when generation fails)", rec.Code)
	}

	var last map[string]any
	for _, line := range strings.Split(strings.TrimSpace(rec.Body.String()), "\n") {
		var frame map[string]any
		if err := json.Unmarshal([]byte(line), &frame); err != nil {
			t.Fatalf("frame %q: %v", line, err)
		}
		last = frame
	}
	if last["done"] != true {
		t.Fatalf("last frame is not done: %v", last)
	}
	errText, _ := last["error"].(string)
	if !strings.Contains(errText, "context holds 512") {
		t.Errorf("final frame carries no error; got %v", last)
	}
	if last["done_reason"] != "error" {
		t.Errorf("done_reason = %v, want \"error\"", last["done_reason"])
	}
}

// TestOpenAIStreamingErrorIsNotReportedAsStop guards the worse half of the same
// bug: the OpenAI dialect claimed finish_reason "stop", actively asserting that
// the model had finished normally.
func TestOpenAIStreamingErrorIsNotReportedAsStop(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()
	srv.open = func(path string, _ engine.Options) (generator, error) {
		return &fakeEngine{path: path, err: errors.New("decode failed")}, nil
	}

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	out := rec.Body.String()
	if strings.Contains(out, `"finish_reason":"stop"`) {
		t.Error("a failed generation reported finish_reason \"stop\"")
	}
	if !strings.Contains(out, "decode failed") {
		t.Errorf("the error never reached the client:\n%s", out)
	}
}

// TestPSReportsWhatIsLoaded covers the endpoint you reach for when a request
// was slower than expected and you want to know whether it paid for a load.
func TestPSReportsWhatIsLoaded(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()
	srv.SetKeepAlive(time.Minute)

	ps := func() []any {
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/ps", nil))
		var out struct {
			Models []any `json:"models"`
		}
		if err := json.NewDecoder(rec.Body).Decode(&out); err != nil {
			t.Fatal(err)
		}
		return out.Models
	}

	if got := ps(); len(got) != 0 {
		t.Errorf("nothing is loaded yet, but ps reports %v", got)
	}

	rec := httptest.NewRecorder()
	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("chat failed: %d %s", rec.Code, rec.Body.String())
	}

	if got := ps(); len(got) != 1 {
		t.Errorf("after a chat, ps reports %d models, want 1", len(got))
	}
}

// TestKeepAliveUnloadsIdleModel checks that an idle model is eventually
// released — a 7B left resident forever would own most of a 16 GB machine.
func TestKeepAliveUnloadsIdleModel(t *testing.T) {
	srv, reg := newTestServer(t, "alpha")
	defer srv.Close()
	srv.SetKeepAlive(50 * time.Millisecond)

	eng, release, err := srv.acquire("alpha")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Chat(context.Background(), nil, engine.DefaultGenParams(), nil); err != nil {
		t.Fatal(err)
	}
	release()

	deadline := time.Now().Add(2 * time.Second)
	for !reg.get("alpha").isClosed() {
		if time.Now().After(deadline) {
			t.Fatal("idle model was never unloaded")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestVersionParsesAsAVersion guards a compatibility trap: Ollama clients
// compare this string to decide which features to use, and a value that is not
// a version fails the comparison silently.
func TestVersionParsesAsAVersion(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/version", nil))

	var out struct {
		Version string `json:"version"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	for _, part := range strings.Split(out.Version, ".") {
		if part == "" || strings.TrimLeft(part, "0123456789") != "" {
			t.Fatalf("version %q is not numeric-dotted; clients that semver-compare it will silently disable features", out.Version)
		}
	}
}

// TestTruncatedReplyReportsLength guards the third member of a family of bugs
// this server has had: reporting that something completed normally when it did
// not. A reasoning model can spend an entire token budget thinking and return
// an empty answer, and "stop" tells the caller the model had nothing to say.
func TestTruncatedReplyReportsLength(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()
	srv.open = func(path string, _ engine.Options) (generator, error) {
		return &fakeEngine{path: path, frags: []string{"cut off here"}, truncated: true}, nil
	}

	for _, c := range []struct{ path, field string }{
		{"/api/chat", "done_reason"},
		{"/v1/chat/completions", "finish_reason"},
	} {
		body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, c.path, strings.NewReader(body)))

		if rec.Code != http.StatusOK {
			t.Fatalf("%s: status %d", c.path, rec.Code)
		}
		if !strings.Contains(rec.Body.String(), `"length"`) {
			t.Errorf("%s: a truncated reply did not report %s \"length\":\n%s", c.path, c.field, rec.Body.String())
		}
	}
}

func TestCompleteReplyReportsStop(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))

	if !strings.Contains(rec.Body.String(), `"done_reason":"stop"`) {
		t.Errorf("a complete reply did not report stop:\n%s", rec.Body.String())
	}
}

// TestBrokenEngineIsReplaced is the regression test for a GPU that ran out of
// memory. llama.cpp's Metal backend enters an error state that no later decode
// recovers from — "recreate the backend to recover" — so a server that keeps
// the context answers every request from then on with the same error. For a
// runtime whose job is to be the fallback, that is being down without saying so.
func TestBrokenEngineIsReplaced(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()

	var loads int
	var mu sync.Mutex
	first := &fakeEngine{path: "alpha", frags: []string{"ok"}, broken: true}

	srv.open = func(path string, _ engine.Options) (generator, error) {
		mu.Lock()
		defer mu.Unlock()
		loads++
		if loads == 1 {
			return first, nil
		}
		return &fakeEngine{path: path, frags: []string{"ok"}}, nil
	}

	// The first acquire loads the engine that is about to break.
	eng, release, err := srv.acquire("alpha")
	if err != nil {
		t.Fatal(err)
	}
	release()
	if !eng.Broken() {
		t.Fatal("the test engine did not report itself broken")
	}

	// The next one must not be handed the same dead engine.
	eng2, release2, err := srv.acquire("alpha")
	if err != nil {
		t.Fatal(err)
	}
	defer release2()

	if eng2 == generator(first) {
		t.Error("a broken engine was handed out again; every request would fail identically")
	}
	if eng2.Broken() {
		t.Error("the replacement is broken too")
	}
	mu.Lock()
	got := loads
	mu.Unlock()
	if got != 2 {
		t.Errorf("the model was loaded %d times, want 2 — the broken one should have been replaced", got)
	}
	if !first.isClosed() {
		t.Error("the broken engine was retired without being closed")
	}
}

// timedFake is the fake engine with measurements attached: four fragments of
// reply, an eleven-token prompt, and durations far enough apart that a swapped
// pair of fields shows up as a wrong number rather than a plausible one.
func timedFake(path string) *fakeEngine {
	return &fakeEngine{
		path:         path,
		frags:        []string{"four", " tokens", " of", " reply"},
		promptTokens: 11,
		promptEval:   2 * time.Millisecond,
		evalDur:      5 * time.Millisecond,
	}
}

// TestOllamaChatReportsTimings guards the fields every measurement of a local
// model reads: `ollama run --verbose`, benchmark scripts, dashboards. All of
// them divide eval_count by eval_duration, and kinfer answered both as zero —
// which does not read as fast, it reads as unmeasurable, leaving a stopwatch as
// the only way to compare kinfer to the runtime it stands in for.
func TestOllamaChatReportsTimings(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()

	fake := timedFake("alpha")
	srv.open = func(path string, _ engine.Options) (generator, error) { return fake, nil }

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	got := decodeTimings(t, rec.Body.Bytes())

	if got.EvalCount == 0 || got.EvalDuration == 0 {
		t.Fatalf("reply reports %d tokens in %d ns: a caller computing tokens/s gets nothing\n%s",
			got.EvalCount, got.EvalDuration, rec.Body)
	}
	if want := len(fake.frags); got.EvalCount != want {
		t.Errorf("eval_count = %d but the model generated %d tokens", got.EvalCount, want)
	}
	// Nanoseconds, not milliseconds. Every consumer assumes the unit rather
	// than asking, so being off by a thousand is a wrong benchmark rather than
	// a broken one.
	if want := (5 * time.Millisecond).Nanoseconds(); got.EvalDuration != want {
		t.Errorf("eval_duration = %d, want %d — Ollama reports integer nanoseconds", got.EvalDuration, want)
	}
	if got.PromptEvalCount != 11 {
		t.Errorf("prompt_eval_count = %d, want 11", got.PromptEvalCount)
	}
	if want := (2 * time.Millisecond).Nanoseconds(); got.PromptEvalDuration != want {
		t.Errorf("prompt_eval_duration = %d, want %d ns", got.PromptEvalDuration, want)
	}
	if got.TotalDuration <= 0 {
		t.Errorf("total_duration = %d; the request took longer than no time at all", got.TotalDuration)
	}
}

// TestOllamaStreamReportsTimingsOnDoneChunk covers the path `ollama run`
// actually uses. Ollama puts the counts on the terminal object only, so a
// streaming client that never sees them there never sees them at all.
func TestOllamaStreamReportsTimingsOnDoneChunk(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()

	fake := timedFake("alpha")
	srv.open = func(path string, _ engine.Options) (generator, error) { return fake, nil }

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))

	lines := strings.Split(strings.TrimSpace(rec.Body.String()), "\n")
	final := decodeTimings(t, []byte(lines[len(lines)-1]))
	if !final.Done {
		t.Fatalf("last streamed object is not the done object:\n%s", rec.Body)
	}
	if final.EvalCount != len(fake.frags) || final.EvalDuration == 0 {
		t.Errorf("done chunk reports %d tokens in %d ns, want %d tokens and a duration:\n%s",
			final.EvalCount, final.EvalDuration, len(fake.frags), rec.Body)
	}

	// The per-token objects stay slim, as Ollama's do: counts belong to the end
	// of the reply, and repeating a running total on every token would be a
	// different protocol.
	for i, line := range lines[:len(lines)-1] {
		if strings.Contains(line, "eval_count") {
			t.Errorf("streamed object %d carries counts before the reply ended: %s", i, line)
		}
	}
}

// TestOpenAIChatReportsUsage covers the other dialect's name for the same
// thing. Clients bill, budget and trim context against usage; all-zero tells
// them the conversation cost nothing, which they believe.
func TestOpenAIChatReportsUsage(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	defer srv.Close()

	fake := timedFake("alpha")
	srv.open = func(path string, _ engine.Options) (generator, error) { return fake, nil }

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var out struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Usage.PromptTokens != 11 || out.Usage.CompletionTokens != len(fake.frags) {
		t.Errorf("usage = %+v, want 11 prompt and %d completion tokens:\n%s",
			out.Usage, len(fake.frags), rec.Body)
	}
	if want := out.Usage.PromptTokens + out.Usage.CompletionTokens; out.Usage.TotalTokens != want {
		t.Errorf("total_tokens = %d but prompt + completion is %d", out.Usage.TotalTokens, want)
	}
}

type wireTimings struct {
	Done               bool  `json:"done"`
	TotalDuration      int64 `json:"total_duration"`
	LoadDuration       int64 `json:"load_duration"`
	PromptEvalCount    int   `json:"prompt_eval_count"`
	PromptEvalDuration int64 `json:"prompt_eval_duration"`
	EvalCount          int   `json:"eval_count"`
	EvalDuration       int64 `json:"eval_duration"`
}

func decodeTimings(t *testing.T, body []byte) wireTimings {
	t.Helper()
	var w wireTimings
	if err := json.Unmarshal(body, &w); err != nil {
		t.Fatalf("decode %s: %v", body, err)
	}
	return w
}

// A request that was still queued when the backend died must come back as a
// complete reply, not as an error.
//
// This is the recoverable half of a fatal failure. The engine that died had
// never touched the request — no tokens reached the client, no KV state exists
// — so the only correct outcome is to run it on the replacement. Before this,
// every queued request died alongside the one batch that actually failed; on a
// load test that was 26 at once, most of which had never started.
func TestQueuedRequestSurvivesAFatalBackendFailure(t *testing.T) {
	srv, reg := newTestServer(t, "alpha")

	var opened int
	srv.open = func(path string, _ engine.Options) (generator, error) {
		opened++
		f := &fakeEngine{path: path, frags: []string{"real", " answer"}}
		if opened == 1 {
			// The first engine dies before this request is picked up.
			f.err = engine.ErrNotStarted
			f.frags = nil
			f.broken = true
		}
		reg.put(f)
		return f, nil
	}

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/api/chat", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("status %d, want 200 — a request that never started was failed instead of retried:\n%s",
			w.Code, w.Body.String())
	}
	var got ollamaChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v (%s)", err, w.Body.String())
	}
	if got.Error != "" {
		t.Fatalf("reply carries error %q", got.Error)
	}
	if got.Message.Content != "real answer" {
		t.Errorf("content = %q, want the reply from the replacement engine", got.Message.Content)
	}
	if opened != 2 {
		t.Errorf("opened %d engines, want 2 — the dead one and its replacement", opened)
	}
}

// A request that had already started must NOT be retried: its client has seen
// part of the answer, and running it again would repeat that text.
func TestInFlightRequestIsNotRetried(t *testing.T) {
	srv, reg := newTestServer(t, "alpha")

	var opened int
	srv.open = func(path string, _ engine.Options) (generator, error) {
		opened++
		f := &fakeEngine{path: path, frags: []string{"half"}, err: errors.New("engine is closing")}
		reg.put(f)
		return f, nil
	}

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/api/chat", strings.NewReader(body)))

	if opened != 1 {
		t.Errorf("opened %d engines — a request that had already streamed text was re-run", opened)
	}
	if w.Code == http.StatusOK {
		t.Errorf("a failed in-flight request reported 200; the caller cannot tell it was cut short")
	}
}

// The retry must not loop: a replacement that is itself down has to surface,
// not spin.
func TestRetryHappensOnceOnly(t *testing.T) {
	srv, reg := newTestServer(t, "alpha")

	var opened int
	srv.open = func(path string, _ engine.Options) (generator, error) {
		opened++
		f := &fakeEngine{path: path, err: engine.ErrNotStarted, broken: true}
		reg.put(f)
		return f, nil
	}

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/api/chat", strings.NewReader(body)))

	if opened > 2 {
		t.Errorf("opened %d engines — the retry is looping", opened)
	}
	if w.Code == http.StatusOK {
		t.Error("a request that never ran twice reported success")
	}
}

// A request that paid for a reload must say so in load_duration.
//
// The trap is where the measurement is taken: the handler times acquire, which
// happens before generation, while a mid-request reload happens during it. Left
// unaccounted, a request that spent forty seconds loading a replacement reports
// every component in milliseconds and only total_duration knows — which is
// worse than a missing figure, because a plausible small number does not look
// wrong.
func TestReloadTimeReachesLoadDuration(t *testing.T) {
	srv, reg := newTestServer(t, "alpha")

	const pause = 60 * time.Millisecond
	var opened int
	srv.open = func(path string, _ engine.Options) (generator, error) {
		opened++
		if opened == 1 {
			f := &fakeEngine{path: path, err: engine.ErrNotStarted, broken: true}
			reg.put(f)
			return f, nil
		}
		time.Sleep(pause) // the replacement is expensive to load, as a real one is
		f := &fakeEngine{path: path, frags: []string{"ok"}}
		reg.put(f)
		return f, nil
	}

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/api/chat", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var got ollamaChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.LoadDuration < pause.Nanoseconds() {
		t.Errorf("load_duration = %v, want at least the %v the reload took — "+
			"the reload happened inside ChatFull and was not counted",
			time.Duration(got.LoadDuration), pause)
	}
	// And the parts must still add up to no more than the whole.
	parts := got.LoadDuration + got.PromptEvalDuration + got.EvalDuration
	if parts > got.TotalDuration {
		t.Errorf("components sum to %v, more than the total %v — the reload was double counted",
			time.Duration(parts), time.Duration(got.TotalDuration))
	}
}

// And a request that never needed a reload must not acquire a phantom one.
func TestLoadDurationUnchangedWithoutARetry(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/api/chat", strings.NewReader(body)))

	var got ollamaChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.LoadDuration > got.TotalDuration {
		t.Errorf("load_duration %v exceeds total %v on a request that loaded once",
			time.Duration(got.LoadDuration), time.Duration(got.TotalDuration))
	}
}

// busyEngine refuses everything, the way a full queue does.
type busyEngine struct {
	fakeEngine
	retryAfter time.Duration
}

func (b *busyEngine) ChatFull(context.Context, []chat.Message, engine.GenParams, func(string)) (string, engine.Stats, error) {
	return "", engine.Stats{}, &engine.BusyError{Waiting: 128, Capacity: 128, RetryAfter: b.retryAfter}
}
func (b *busyEngine) Chat(context.Context, []chat.Message, engine.GenParams, func(string)) (string, error) {
	return "", &engine.BusyError{Waiting: 128, Capacity: 128, RetryAfter: b.retryAfter}
}

func serveBusy(t *testing.T, path, body string, retryAfter time.Duration) *httptest.ResponseRecorder {
	t.Helper()
	srv, _ := newTestServer(t, "alpha")
	srv.open = func(p string, _ engine.Options) (generator, error) {
		return &busyEngine{fakeEngine: fakeEngine{path: p}, retryAfter: retryAfter}, nil
	}
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", path, strings.NewReader(body)))
	return w
}

// A refused request must come back as 503, not 500. They mean different things
// to a fleet: retrying a 500 hammers a broken server, giving up on a 503
// abandons work that would have succeeded.
func TestFullQueueAnswers503(t *testing.T) {
	for _, c := range []struct{ name, path, body string }{
		{"ollama non-streaming", "/api/chat",
			`{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`},
		{"ollama streaming", "/api/chat",
			`{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true}`},
		{"openai non-streaming", "/v1/chat/completions",
			`{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`},
		{"openai streaming", "/v1/chat/completions",
			`{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true}`},
	} {
		w := serveBusy(t, c.path, c.body, 0)
		if w.Code != http.StatusServiceUnavailable {
			t.Errorf("%s: status %d, want 503 — a busy server is not a broken one:\n%s",
				c.name, w.Code, w.Body.String())
		}
	}
}

// A streaming request that is refused must not have opened a stream first. A
// 200 whose body says "busy" is the failure mode this project exists to avoid.
func TestRefusedStreamNeverOpens(t *testing.T) {
	w := serveBusy(t, "/api/chat",
		`{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true}`, 0)
	if ct := w.Header().Get("Content-Type"); strings.Contains(ct, "ndjson") {
		t.Errorf("Content-Type = %q — the stream header went out before the refusal", ct)
	}
	if strings.Contains(w.Body.String(), `"done":true`) {
		t.Error("a terminal stream object was emitted for a request that never ran")
	}
}

// Retry-After is sent only when there is a measurement behind it. A guessed
// interval brings every refused caller back at the same invented moment.
func TestRetryAfterOnlyWhenMeasured(t *testing.T) {
	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`

	w := serveBusy(t, "/api/chat", body, 0)
	if got := w.Header().Get("Retry-After"); got != "" {
		t.Errorf("Retry-After = %q with no estimate available, want it absent", got)
	}

	w = serveBusy(t, "/api/chat", body, 4500*time.Millisecond)
	if got := w.Header().Get("Retry-After"); got != "5" {
		t.Errorf("Retry-After = %q for a 4.5s estimate, want \"5\" — whole seconds, rounded up "+
			"so the caller is not invited back before the work can have finished", got)
	}
}

// A reply cut off by the clock must arrive as a reply, with the text it did
// produce and a reason saying it was cut off. Reporting it as an error would
// throw away work the caller can use; reporting it as "stop" would be the lie
// this runtime exists to not tell.
func TestTimeoutIsAFinishReasonNotAFailure(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	srv.open = func(p string, _ engine.Options) (generator, error) {
		return &timedOutEngine{fakeEngine{path: p, frags: []string{"half an ", "answer"}}}, nil
	}

	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/api/chat",
		strings.NewReader(`{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`)))

	if w.Code != http.StatusOK {
		t.Fatalf("status %d, want 200 — the partial reply is worth more than an error", w.Code)
	}
	var got ollamaChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Message.Content != "half an answer" {
		t.Errorf("content = %q, want the text produced before the deadline", got.Message.Content)
	}
	if got.DoneReason != "timeout" {
		t.Errorf("done_reason = %q, want \"timeout\" — \"stop\" would claim the model finished", got.DoneReason)
	}
}

// OpenAI's finish_reason is a closed set its clients switch on, so a timeout
// travels as "length" there. Still not "stop": the caller must be able to tell
// an incomplete reply from a complete one.
func TestOpenAITimeoutUsesAKnownFinishReason(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	srv.open = func(p string, _ engine.Options) (generator, error) {
		return &timedOutEngine{fakeEngine{path: p, frags: []string{"partial"}}}, nil
	}

	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":false}`)))

	var got struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v (%s)", err, w.Body.String())
	}
	if len(got.Choices) == 0 {
		t.Fatalf("no choices: %s", w.Body.String())
	}
	switch got.Choices[0].FinishReason {
	case "length":
	case "stop":
		t.Error("finish_reason = \"stop\" for a reply the server cut off")
	default:
		t.Errorf("finish_reason = %q, which is outside OpenAI's closed set",
			got.Choices[0].FinishReason)
	}
}

type timedOutEngine struct{ fakeEngine }

func (e *timedOutEngine) ChatFull(ctx context.Context, m []chat.Message, p engine.GenParams, onToken func(string)) (string, engine.Stats, error) {
	text, st, err := e.fakeEngine.ChatFull(ctx, m, p, onToken)
	st.Timeout = true
	return text, st, err
}

// A request that aged out of the queue never ran, so it must be refused — not
// answered with an empty body and a reason, which would say the model had
// nothing to say.
func TestQueueExpiryIsRefusedNotAnsweredEmpty(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	srv.open = func(p string, _ engine.Options) (generator, error) {
		return &staleEngine{fakeEngine{path: p}}, nil
	}

	for _, path := range []string{"/api/chat", "/v1/chat/completions"} {
		for _, stream := range []string{"false", "true"} {
			w := httptest.NewRecorder()
			srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", path, strings.NewReader(
				`{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":`+stream+`}`)))
			if w.Code != http.StatusServiceUnavailable {
				t.Errorf("%s stream=%s: status %d, want 503 — nothing of this request ran:\n%s",
					path, stream, w.Code, w.Body.String())
			}
			if strings.Contains(w.Body.String(), `"done_reason":"timeout"`) {
				t.Errorf("%s stream=%s: reported a finish reason for a reply that never started", path, stream)
			}
		}
	}
}

type staleEngine struct{ fakeEngine }

func (e *staleEngine) ChatFull(context.Context, []chat.Message, engine.GenParams, func(string)) (string, engine.Stats, error) {
	return "", engine.Stats{}, &engine.TimeoutError{After: 90 * time.Second, Stage: engine.StageQueued}
}
func (e *staleEngine) Chat(context.Context, []chat.Message, engine.GenParams, func(string)) (string, error) {
	return "", &engine.TimeoutError{After: 90 * time.Second, Stage: engine.StageQueued}
}

func (f *fakeEngine) Load() (waiting, busy, slots int) { return 0, 0, 1 }
func (f *fakeEngine) PromptTokens() int64              { return 0 }
func (f *fakeEngine) EvalTokens() int64                { return 0 }

// think:false must reach the engine as NoThink; anything else must not. The
// asymmetry is deliberate — absent means the model's own default, as it does
// in Ollama — and getting it wrong in either direction is visible: off when it
// should be on loses the reasoning a caller asked for, on when it should be
// off spends the whole budget on text nobody sees.
func TestThinkOffOnlyForAnExplicitFalse(t *testing.T) {
	for raw, want := range map[string]bool{
		`false`:   true,
		`"false"`: true,
		`true`:    false,
		`"high"`:  false,
		`"low"`:   false,
		``:        false,
	} {
		if got := thinkOff(json.RawMessage(raw)); got != want {
			t.Errorf("thinkOff(%s) = %v, want %v", raw, got, want)
		}
	}
}

// A streamed reply with functions on the table must stream its prose and
// deliver the call only in the terminal object, never as fragments of JSON.
// Buffering the whole reply was the alternative, and it cost a fleet agent
// every sign of progress until the end.
func TestStreamedToolReplyStreamsProseAndClosesWithCalls(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	srv.open = func(p string, _ engine.Options) (generator, error) {
		return &fakeEngine{path: p, tools: true, frags: []string{
			"Let me ", "look that up.", " <tool_c", "all>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Berlin\"}}\n</tool_call>"}}, nil
	}
	body := `{"model":"alpha","stream":true,"messages":[{"role":"user","content":"weather?"}],
	          "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}`
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/api/chat", strings.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var prose strings.Builder
	var final ollamaChatResponse
	n := 0
	for _, line := range strings.Split(strings.TrimSpace(w.Body.String()), "\n") {
		var r ollamaChatResponse
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("bad chunk %q: %v", line, err)
		}
		if strings.Contains(r.Message.Content, "<tool_call>") {
			t.Errorf("a chunk leaked call syntax into the stream: %q", r.Message.Content)
		}
		if r.Done {
			final = r
		} else {
			prose.WriteString(r.Message.Content)
			n++
		}
	}
	if n < 2 || prose.String() != "Let me look that up. " {
		t.Errorf("streamed %d chunks, prose %q; want the prose streamed as it came", n, prose.String())
	}
	if final.DoneReason != "tool_calls" || len(final.Message.ToolCalls) != 1 || final.Message.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("terminal object = %+v; want done_reason tool_calls with the parsed call", final)
	}
}
