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
)

// fakeEngine stands in for a loaded model. Chat blocks until release is closed,
// which is what lets a test hold a model "mid-generation" while another request
// asks for a different one.
type fakeEngine struct {
	path    string
	err     error
	frags   []string
	release chan struct{}

	tools bool

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

func (f *fakeEngine) SupportsTools() bool { return f.tools }

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
