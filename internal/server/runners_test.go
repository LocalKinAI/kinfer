package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// speechDir makes an MLX speech model's directory in the store: a config of
// modelType and the files that make it one.
func speechDir(t *testing.T, root, name, modelType string) string {
	t.Helper()
	dir := filepath.Join(root, name)
	os.MkdirAll(filepath.Join(dir, "speech_tokenizer"), 0o755)
	cfg := map[string]any{"model_type": modelType, "tts_model_type": "custom_voice"}
	b, _ := json.Marshal(cfg)
	for f, body := range map[string][]byte{"config.json": b, "model.safetensors": nil, "vocab.json": []byte("{}"),
		"merges.txt": nil, "speech_tokenizer/model.safetensors": nil} {
		if err := os.WriteFile(filepath.Join(dir, f), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// fakeRunners stand in for runner processes: each an httptest server that
// says which model it was asked to use.
type fakeRunners struct {
	mu      sync.Mutex
	started map[string]int // starts by model path
	stopped map[string]int
	live    map[string]*proc
}

func (f *fakeRunners) start(path string) (*proc, error) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		writeJSON(w, http.StatusOK, map[string]any{"runner": filepath.Base(path), "model": r.Header.Get(runnerHeader),
			"path": r.URL.Path, "query": r.URL.RawQuery, "bytes": len(body)})
	}))
	done := make(chan struct{})
	var once sync.Once
	p := &proc{base: ts.URL, done: done}
	p.stop = func() {
		once.Do(func() {
			ts.Close()
			close(done)
			f.mu.Lock()
			f.stopped[path]++
			f.mu.Unlock()
		})
	}
	f.mu.Lock()
	f.started[path]++
	f.live[path] = p
	f.mu.Unlock()
	return p, nil
}

func (f *fakeRunners) count(m map[string]int, path string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return m[path]
}

// newFrontServer is a front server whose store has the default ASR and TTS
// and the chat model "chat", its runners fakes.
func newFrontServer(t *testing.T) (*Server, *fakeRunners, string, string) {
	t.Helper()
	srv, _ := newTestServer(t, "chat")
	root := srv.store.Root()
	asr := speechDir(t, root, "Qwen3-ASR-1.7B-8bit", "qwen3_asr")
	tts := speechDir(t, root, "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", "qwen3_tts")
	srv.speechDefaults = []SpeechModel{
		{kindASR, "mlx-community/Qwen3-ASR-1.7B-8bit", "Qwen3-ASR-1.7B-8bit", "2.5 GB"},
		{kindTTS, "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", "1.8 GB"},
	}
	srv.SetSpeech(SpeechAuto, true, ":11500")
	f := &fakeRunners{started: map[string]int{}, stopped: map[string]int{}, live: map[string]*proc{}}
	srv.runners.start = f.start
	return srv, f, asr, tts
}

func do(t *testing.T, srv *Server, method, path, contentType, body string) (int, map[string]any, http.Header) {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	var out map[string]any
	json.Unmarshal(w.Body.Bytes(), &out)
	return w.Code, out, w.Header()
}

// A speech request goes to a runner of the model it names when that is a
// speech model of its kind, and else to the default — OpenAI's names, the
// Python service's "tts" — while the chat model is served here.
func TestSpeechGoesToItsRunner(t *testing.T) {
	srv, f, asr, tts := newFrontServer(t)
	defer srv.Close()

	code, out, _ := do(t, srv, "POST", "/v1/audio/speech", "application/json", `{"model":"tts-1","input":"hi","voice":"alloy"}`)
	if code != 200 || out["model"] != tts || out["path"] != "/v1/audio/speech" {
		t.Fatalf("speech: %d %v", code, out)
	}
	form := "--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nRIFF\r\n--b--\r\n"
	code, out, _ = do(t, srv, "POST", "/v1/audio/transcriptions", "multipart/form-data; boundary=b", form)
	if code != 200 || out["model"] != asr || out["bytes"] != float64(len(form)) {
		t.Fatalf("transcription: %d %v", code, out)
	}
	// Named by its own name, a speech model is that one; named as a model
	// of the other kind, it is the default of this one.
	other := speechDir(t, srv.store.Root(), "my-voice", "qwen3_tts")
	if _, out, _ = do(t, srv, "POST", "/v1/audio/speech", "application/json", `{"model":"my-voice","input":"hi"}`); out["model"] != other {
		t.Errorf("my-voice went to %v", out["model"])
	}
	if _, out, _ = do(t, srv, "POST", "/v1/audio/speech", "application/json", `{"model":"Qwen3-ASR-1.7B-8bit","input":"hi"}`); out["model"] != tts {
		t.Errorf("an ASR model named for speech went to %v", out["model"])
	}
	if _, out, _ = do(t, srv, "GET", "/v1/audio/voices?model=tts", "", ""); out["model"] != tts || out["path"] != "/v1/audio/voices" {
		t.Errorf("voices: %v", out)
	}
	// One runner a model, kept between requests.
	if f.count(f.started, tts) != 1 || f.count(f.started, asr) != 1 {
		t.Errorf("runners started: %v", f.started)
	}
	// The chat model is this process's, as ever.
	if code, out, _ := do(t, srv, "POST", "/api/chat", "application/json", `{"model":"chat","stream":false,"messages":[{"role":"user","content":"hi"}]}`); code != 200 {
		t.Errorf("chat: %d %v", code, out)
	}
	// /api/ps shows the speech models beside the chat model.
	_, ps, _ := do(t, srv, "GET", "/api/ps", "", "")
	names := []string{}
	for _, m := range ps["models"].([]any) {
		names = append(names, m.(map[string]any)["name"].(string))
	}
	if all := strings.Join(names, ","); !strings.Contains(all, "chat") || !strings.Contains(all, "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit") || !strings.Contains(all, "Qwen3-ASR-1.7B-8bit") {
		t.Errorf("ps: %v", names)
	}
}

// An idle runner is stopped after -keepalive; one that exits is started
// again by the next request; Close stops them all.
func TestRunnersStopWhenIdleAndComeBack(t *testing.T) {
	srv, f, _, tts := newFrontServer(t)
	srv.SetKeepAlive(50 * time.Millisecond)
	body := `{"model":"tts","input":"hi"}`
	do(t, srv, "POST", "/v1/audio/speech", "application/json", body)
	deadline := time.Now().Add(2 * time.Second)
	for f.count(f.stopped, tts) == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if f.count(f.stopped, tts) != 1 {
		t.Fatal("an idle runner was not stopped")
	}
	do(t, srv, "POST", "/v1/audio/speech", "application/json", body)
	if f.count(f.started, tts) != 2 {
		t.Fatalf("started %d times, want 2", f.count(f.started, tts))
	}

	srv.SetKeepAlive(time.Hour)
	do(t, srv, "POST", "/v1/audio/speech", "application/json", body)
	f.mu.Lock()
	p := f.live[tts]
	f.mu.Unlock()
	p.stop() // it crashed
	time.Sleep(20 * time.Millisecond)
	if code, _, _ := do(t, srv, "POST", "/v1/audio/speech", "application/json", body); code != 200 || f.count(f.started, tts) < 3 {
		t.Fatalf("after its runner exited: %d, started %d", code, f.count(f.started, tts))
	}
	srv.Close()
	if f.count(f.stopped, tts) != f.count(f.started, tts) {
		t.Errorf("after Close: started %d, stopped %d", f.count(f.started, tts), f.count(f.stopped, tts))
	}
}

// A default model that is not here is fetched, the request answered 503
// with how far that has got and when to ask again; with -auto-pull=false,
// 404 saying what to pull; a kind this machine has no model of, 501.
func TestMissingSpeechModel(t *testing.T) {
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
	code, out, h := do(t, srv, "POST", "/v1/audio/speech", "application/json", `{"model":"tts-1","input":"hi"}`)
	msg, _ := out["error"].(map[string]any)["message"].(string)
	if code != 503 || h.Get("Retry-After") == "" || !strings.Contains(msg, "being fetched") {
		t.Fatalf("%d %v %v", code, out, h)
	}
	time.Sleep(20 * time.Millisecond)
	code, out, _ = do(t, srv, "POST", "/v1/audio/speech", "application/json", `{"model":"tts-1","input":"hi"}`)
	if msg, _ := out["error"].(map[string]any)["message"].(string); code != 503 || !strings.Contains(msg, "600.0 MB of 1.8 GB (33%)") {
		t.Errorf("asked again: %d %q", code, msg)
	}
	mu.Lock()
	if len(started) != 1 || started[0] != "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit" {
		t.Errorf("fetches started: %v", started)
	}
	mu.Unlock()

	srv.SetSpeech(SpeechAuto, false, ":11500")
	code, out, _ = do(t, srv, "POST", "/v1/audio/speech", "application/json", `{"model":"tts-1","input":"hi"}`)
	if msg, _ := out["error"].(map[string]any)["message"].(string); code != 404 || !strings.Contains(msg, "kinfer pull mlx-community/Qwen3-TTS") {
		t.Errorf("without auto-pull: %d %v", code, out)
	}

	srv.speechDefaults = srv.speechDefaults[:1] // a machine with no text to speech
	if code, _, _ = do(t, srv, "POST", "/v1/audio/speech", "application/json", `{"model":"tts-1","input":"hi"}`); code != 501 {
		t.Errorf("no text to speech here: %d", code)
	}
}

// A runner takes the model the front server chose, not the request's own
// name, which may be OpenAI's.
func TestRunnerTakesTheChosenModel(t *testing.T) {
	srv, sp := newSpeechServer(t)
	defer srv.Close()
	srv.SetRunner()
	req := httptest.NewRequest("POST", "/v1/audio/speech", strings.NewReader(`{"model":"tts-1","input":"hello"}`))
	req.Header.Set(runnerHeader, "tts")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != 200 || sp.request().Text != "hello" {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}
