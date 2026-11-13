package server

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestOriginAllowed(t *testing.T) {
	extra := []string{"http://192.168.0.*:7700", "https://home.example"}
	for _, c := range []struct {
		origin, host string
		want         bool
	}{
		{"http://localhost:9000", "192.168.0.21:11590", true},
		{"http://127.0.0.1:3000", "192.168.0.21:11590", true},
		{"http://[::1]:3000", "localhost:11590", true},
		{"http://app.localhost", "localhost:11590", true},
		{"http://0.0.0.0:8080", "localhost:11590", true},
		{"http://192.168.0.21:11590", "192.168.0.21:11590", true}, // its own page
		{"app://kinhome", "localhost:11590", true},
		{"file://", "localhost:11590", true},
		{"tauri://localhost", "localhost:11590", true},
		{"http://192.168.0.5:7700", "192.168.0.21:11590", true}, // listed
		{"https://home.example", "192.168.0.21:11590", true},
		{"https://evil.example", "localhost:11590", false},
		{"https://evil.example", "192.168.0.21:11590", false},
		{"http://localhost.evil.example", "localhost:11590", false},
		{"http://192.168.0.5:7701", "192.168.0.21:11590", false},
		{"null", "localhost:11590", false}, // a sandboxed page
		{"ftp://localhost", "localhost:11590", false},
	} {
		if got := originAllowed(c.origin, c.host, extra); got != c.want {
			t.Errorf("origin %s to %s: %v, want %v", c.origin, c.host, got, c.want)
		}
	}
	if !originAllowed("https://anything.example", "x", []string{"*"}) {
		t.Error(`KINFER_ORIGINS="*" does not let every page through`)
	}
}

// ask sends a request through the door to a handler that answers 200.
func ask(t *testing.T, h http.Handler, method, path, from string, header map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, nil)
	req.RemoteAddr = from
	for k, v := range header {
		req.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w
}

// A web page that is not this machine's own is refused at the door, whatever
// it asks; one that is, and a request with no page (curl, an agent, an app),
// go through.
func TestTheDoorRefusesOtherPages(t *testing.T) {
	t.Setenv("KINFER_API_KEY", "")
	t.Setenv("KINFER_ORIGINS", "")
	srv := &Server{}
	h := srv.guard(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }))
	const lan, me = "192.168.0.5:50000", "127.0.0.1:50000"
	if w := ask(t, h, "POST", "/api/generate", me, map[string]string{"Origin": "https://evil.example"}); w.Code != 403 || !strings.Contains(w.Body.String(), "KINFER_ORIGINS") {
		t.Errorf("another site's page, from this machine: %d %s", w.Code, w.Body)
	}
	if w := ask(t, h, "POST", "/api/services/stop", me, map[string]string{"Origin": "http://evil.example"}); w.Code != 403 {
		t.Errorf("another site's page asking a service stopped: %d", w.Code)
	}
	if w := ask(t, h, "POST", "/api/chat", lan, map[string]string{"Origin": "http://localhost:9000"}); w.Code != 200 {
		t.Errorf("a page of localhost: %d", w.Code)
	}
	if w := ask(t, h, "POST", "/api/chat", lan, nil); w.Code != 200 {
		t.Errorf("no page, from the network, no key asked: %d", w.Code)
	}
	t.Setenv("KINFER_ORIGINS", "https://evil.example")
	h = srv.guard(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }))
	if w := ask(t, h, "POST", "/api/generate", me, map[string]string{"Origin": "https://evil.example"}); w.Code != 200 {
		t.Errorf("a page KINFER_ORIGINS lists: %d", w.Code)
	}
}

// With KINFER_API_KEY set, a request from another machine needs the key, as
// a Bearer token or x-api-key; this machine's own do not, nor the two
// endpoints that say nothing.
func TestTheKeyIsAskedOfOtherMachines(t *testing.T) {
	t.Setenv("KINFER_API_KEY", "s3cret")
	t.Setenv("KINFER_ORIGINS", "")
	srv := &Server{}
	h := srv.guard(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }))
	const lan = "192.168.0.5:50000"
	w := ask(t, h, "POST", "/v1/chat/completions", lan, nil)
	if w.Code != 401 || w.Header().Get("WWW-Authenticate") == "" {
		t.Errorf("no key: %d %v", w.Code, w.Header())
	}
	for _, hd := range []map[string]string{
		{"Authorization": "Bearer wrong"},
		{"x-api-key": "s3cre"},
		{"Authorization": "Basic s3cret"},
	} {
		if w := ask(t, h, "POST", "/v1/chat/completions", lan, hd); w.Code != 401 {
			t.Errorf("%v: %d", hd, w.Code)
		}
	}
	for _, hd := range []map[string]string{
		{"Authorization": "Bearer s3cret"},
		{"Authorization": "bearer s3cret"},
		{"x-api-key": "s3cret"},
	} {
		if w := ask(t, h, "POST", "/v1/chat/completions", lan, hd); w.Code != 200 {
			t.Errorf("%v: %d", hd, w.Code)
		}
	}
	for _, from := range []string{"127.0.0.1:50000", "[::1]:50000"} {
		if w := ask(t, h, "POST", "/v1/chat/completions", from, nil); w.Code != 200 {
			t.Errorf("this machine (%s), no key: %d", from, w.Code)
		}
	}
	for _, p := range []string{"/api/version", "/health"} {
		if w := ask(t, h, "GET", p, lan, nil); w.Code != 200 {
			t.Errorf("%s, no key: %d", p, w.Code)
		}
	}
	// The page check comes first: a page refused is refused with the key.
	if w := ask(t, h, "POST", "/api/chat", lan, map[string]string{"Origin": "https://evil.example", "x-api-key": "s3cret"}); w.Code != 403 {
		t.Errorf("another site's page, with the key: %d", w.Code)
	}
}

// The real door is in front of every endpoint, the speech addresses' too.
func TestTheDoorIsInFrontOfEverything(t *testing.T) {
	t.Setenv("KINFER_API_KEY", "s3cret")
	srv, _, _, _ := newFrontServer(t)
	defer srv.Close()
	for _, h := range []http.Handler{srv.Handler(), srv.guard(srv.SpeechHandler(kindTTS))} {
		if w := ask(t, h, "GET", "/v1/models", "192.168.0.5:50000", nil); w.Code != 401 {
			t.Errorf("no key from the network: %d", w.Code)
		}
		if w := ask(t, h, "GET", "/v1/models", "127.0.0.1:50000", map[string]string{"Origin": "https://evil.example"}); w.Code != 403 {
			t.Errorf("another site's page: %d", w.Code)
		}
	}
}

// Taking a service in, letting it go and stopping it is asked as JSON: a
// form or plain text, which a page can have a browser send unasked, is not.
func TestServiceActionsAreAskedAsJSON(t *testing.T) {
	srv, _, _, _ := newFrontServer(t)
	defer srv.Close()
	for _, ct := range []string{"", "text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"} {
		req := httptest.NewRequest("POST", "/api/services/stop", strings.NewReader(`{"name":"comfyui"}`))
		req.RemoteAddr = "127.0.0.1:50000"
		if ct != "" {
			req.Header.Set("Content-Type", ct)
		}
		w := httptest.NewRecorder()
		srv.handleServiceAction("stop")(w, req)
		if w.Code != http.StatusUnsupportedMediaType {
			t.Errorf("as %q: %d %s", ct, w.Code, w.Body)
		}
	}
	req := httptest.NewRequest("POST", "/api/services/stop", strings.NewReader(`{"name":"comfyui"}`))
	req.RemoteAddr = "127.0.0.1:50000"
	req.Header.Set("Content-Type", "application/json; charset=utf-8")
	w := httptest.NewRecorder()
	srv.handleServiceAction("stop")(w, req)
	if w.Code == http.StatusUnsupportedMediaType {
		t.Errorf("as JSON: %d %s", w.Code, w.Body)
	}
}

// A default model that is not here is fetched when asked for from this
// machine, not from another: that one is told to pull it, and sees a fetch
// already under way.
func TestOnlyThisMachineStartsAFetch(t *testing.T) {
	srv, _, _, tts := newFrontServer(t)
	defer srv.Close()
	os.RemoveAll(tts)
	var mu sync.Mutex
	var started []string
	srv.pulls.start = func(p *pull) {
		mu.Lock()
		started = append(started, p.model.Repo)
		mu.Unlock()
		p.total.Store(1800 << 20)
		p.done.Store(600 << 20)
	}
	body := `{"model":"tts-1","input":"hi"}`
	from := func(addr string) (int, string) {
		req := httptest.NewRequest("POST", "/v1/audio/speech", strings.NewReader(body))
		req.RemoteAddr = addr
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, req)
		return w.Code, w.Body.String()
	}
	if code, msg := from("192.168.0.5:50000"); code != 404 || !strings.Contains(msg, "kinfer pull mlx-community/Qwen3-TTS") || !strings.Contains(msg, "on it") {
		t.Errorf("from the network: %d %s", code, msg)
	}
	time.Sleep(20 * time.Millisecond)
	mu.Lock()
	if len(started) != 0 {
		t.Fatalf("a request from the network started a fetch: %v", started)
	}
	mu.Unlock()
	if code, msg := from("127.0.0.1:50000"); code != 503 || !strings.Contains(msg, "being fetched") {
		t.Fatalf("from this machine: %d %s", code, msg)
	}
	time.Sleep(20 * time.Millisecond)
	if code, msg := from("192.168.0.5:50000"); code != 503 || !strings.Contains(msg, "600.0 MB of 1.8 GB") {
		t.Errorf("from the network, a fetch under way: %d %s", code, msg)
	}
	mu.Lock()
	if len(started) != 1 {
		t.Errorf("fetches started: %v", started)
	}
	mu.Unlock()
}
