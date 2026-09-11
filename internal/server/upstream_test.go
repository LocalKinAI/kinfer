package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// cloudServer is a kinfer whose store knows one cloud model, pointed at a stub
// upstream.
//
// The cloud model arrives through a real Ollama manifest tree rather than being
// injected, so the thing under test includes how a cloud entry is recognised —
// by having no weights layer, which is the part that must not become a guess
// about the ":cloud" suffix.
func cloudServer(t *testing.T, up *httptest.Server) *Server {
	t.Helper()

	root := t.TempDir()
	man := filepath.Join(root, "manifests", "registry.ollama.ai", "library", "kimi-k2.5", "cloud")
	if err := os.MkdirAll(filepath.Dir(man), 0o755); err != nil {
		t.Fatal(err)
	}
	// A cloud manifest: layers, but none of them the weights.
	if err := os.WriteFile(man, []byte(`{"layers":[
		{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:aa","size":12},
		{"mediaType":"application/vnd.ollama.image.params","digest":"sha256:bb","size":7}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	// A store whose blobs directory is empty loses to any real one on the
	// machine: ollamaRoot ranks candidates by whether they actually hold
	// weights, and treats a manifests-only tree as a fallback. Without this the
	// test reads the developer's own Ollama store and passes or fails according
	// to what they happen to have installed — which it did, once.
	if err := os.MkdirAll(filepath.Join(root, "blobs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "blobs", "sha256-deadbeef"),
		make([]byte, 2<<20), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_MODELS", root)

	srv, _ := newTestServer(t, "local")
	srv.store.Borrow(true)
	if up != nil {
		srv.upstream = up.URL
	}
	srv.open = func(string, engine.Options) (generator, error) {
		t.Error("a cloud model was loaded locally; kinfer has no weights for one")
		return nil, nil
	}
	return srv
}

func post(srv *Server, path, body string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", path, strings.NewReader(body)))
	return w
}

// A cloud model must reach the upstream unchanged. Re-encoding the parsed
// request would be a second place for its shape to drift from what was sent.
func TestCloudRequestIsForwardedVerbatim(t *testing.T) {
	var got struct {
		path, body string
	}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got.path, got.body = r.URL.Path, string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"kimi-k2.5:cloud","message":{"role":"assistant","content":"hi"},"done":true}`))
	}))
	defer up.Close()

	body := `{"model":"kimi-k2.5:cloud","messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"temperature":0.3}}`
	w := post(cloudServer(t, up), "/api/chat", body)

	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	if got.path != "/api/chat" {
		t.Errorf("upstream path = %q, want the same path the caller used", got.path)
	}
	if got.body != body {
		t.Errorf("upstream body differs from what was sent:\n got %s\nwant %s", got.body, body)
	}
	if !strings.Contains(w.Body.String(), `"content":"hi"`) {
		t.Errorf("the upstream reply did not reach the caller: %s", w.Body.String())
	}
}

// Both dialects route, and each keeps its own path upstream — Ollama speaks
// both, so there is nothing to translate.
func TestBothDialectsForward(t *testing.T) {
	var seen []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.URL.Path)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer up.Close()

	srv := cloudServer(t, up)
	post(srv, "/api/chat", `{"model":"kimi-k2.5:cloud","messages":[]}`)
	post(srv, "/v1/chat/completions", `{"model":"kimi-k2.5:cloud","messages":[]}`)

	want := []string{"/api/chat", "/v1/chat/completions"}
	if len(seen) != 2 || seen[0] != want[0] || seen[1] != want[1] {
		t.Errorf("upstream saw %v, want %v", seen, want)
	}
}

// An upstream that is not running must fail at once. Turning "Ollama is not
// started" into a five-minute wait would move the original outage, not fix it.
func TestMissingUpstreamFailsImmediately(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	addr := up.URL
	up.Close() // nothing is listening there now

	srv := cloudServer(t, nil)
	srv.upstream = addr
	srv.opts.MaxGenerate = time.Minute

	start := time.Now()
	w := post(srv, "/api/chat", `{"model":"kimi-k2.5:cloud","messages":[]}`)
	el := time.Since(start)

	if el > 5*time.Second {
		t.Errorf("took %v to notice nothing was listening", el)
	}
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status %d, want 503 for an upstream that is not there: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "OLLAMA_HOST") {
		t.Errorf("the error does not say how to fix it: %s", w.Body.String())
	}
}

// The point of being in this path at all: an upstream that stops answering is
// cut off, rather than held open the way the original outage was.
func TestSilentUpstreamHitsTheDeadline(t *testing.T) {
	block := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block // never answers, exactly like the model that caused all this
	}))
	defer up.Close()
	defer close(block)

	srv := cloudServer(t, up)
	srv.opts.MaxGenerate = 300 * time.Millisecond

	start := time.Now()
	w := post(srv, "/api/chat", `{"model":"kimi-k2.5:cloud","messages":[]}`)
	el := time.Since(start)

	if el > 5*time.Second {
		t.Fatalf("waited %v on an upstream that never answers", el)
	}
	if w.Code != http.StatusGatewayTimeout {
		t.Errorf("status %d, want 504: %s", w.Code, w.Body.String())
	}
}

// A local model must not be sent anywhere, and an unknown name must still fail
// fast rather than being forwarded on the chance that it is remote.
func TestOnlyKnownCloudModelsAreForwarded(t *testing.T) {
	var forwarded int
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		forwarded++
		_, _ = w.Write([]byte(`{}`))
	}))
	defer up.Close()

	srv := cloudServer(t, up)
	w := post(srv, "/api/chat", `{"model":"no-such-model","messages":[]}`)
	if forwarded != 0 {
		t.Error("an unknown model was forwarded upstream; a typo must stay a fast 404")
	}
	if w.Code != http.StatusNotFound {
		t.Errorf("unknown model gave status %d, want 404: %s", w.Code, w.Body.String())
	}
}

func TestUpstreamHonoursOllamaHost(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "10.0.0.5:11434")
	if got := Upstream(); got != "http://10.0.0.5:11434" {
		t.Errorf("Upstream() = %q, want a scheme added to a bare host:port", got)
	}
	t.Setenv("OLLAMA_HOST", "")
	if got := Upstream(); got != DefaultUpstream {
		t.Errorf("Upstream() = %q, want %q", got, DefaultUpstream)
	}
}
