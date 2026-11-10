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
)

// audioRunners stand in for speech runners as the legacy endpoints use them:
// each answers OpenAI's speech endpoints and keeps what it was asked.
type audioRunners struct {
	mu   sync.Mutex
	asks []audioAsk
}

type audioAsk struct {
	runner, path string
	form         map[string]string
	json         map[string]any
}

func (a *audioRunners) start(path string) (*proc, error) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ask := audioAsk{runner: filepath.Base(path), path: r.URL.Path, form: map[string]string{}}
		switch r.URL.Path {
		case "/v1/audio/transcriptions":
			r.ParseMultipartForm(1 << 20)
			for k, v := range r.MultipartForm.Value {
				ask.form[k] = v[0]
			}
			if f, _, err := r.FormFile("file"); err == nil {
				b, _ := io.ReadAll(f)
				ask.form["file"] = string(b)
			}
			if ask.form["response_format"] == "text" {
				w.Write([]byte("你好\n"))
			} else {
				writeJSON(w, http.StatusOK, map[string]any{"task": "transcribe", "text": "你好", "language": "chinese",
					"duration": 1.5, "emotion": "happy", "segments": []any{map[string]any{"start": 0}}})
			}
		case "/v1/audio/speech":
			json.NewDecoder(r.Body).Decode(&ask.json)
			w.Header().Set("X-Kinfer-Voice", "uncle_fu")
			w.Write([]byte("RIFF....WAVEfmt "))
		case "/v1/audio/voices":
			if strings.Contains(path, "Design") {
				writeJSON(w, http.StatusOK, map[string]any{"voices": []string{}, "languages": []string{"auto"}})
			} else {
				writeJSON(w, http.StatusOK, map[string]any{"voices": []string{"vivian", "uncle_fu"}, "languages": []string{"auto", "chinese"}})
			}
		default:
			http.NotFound(w, r)
		}
		a.mu.Lock()
		a.asks = append(a.asks, ask)
		a.mu.Unlock()
	}))
	done := make(chan struct{})
	var once sync.Once
	return &proc{base: ts.URL, done: done, stop: func() { once.Do(func() { ts.Close(); close(done) }) }}, nil
}

func (a *audioRunners) last() audioAsk {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.asks[len(a.asks)-1]
}

// legacyServer is a front server with the default speech models, VoiceDesign
// among them, and its runners fakes.
func legacyServer(t *testing.T) (*Server, *audioRunners) {
	t.Helper()
	srv, _, _, _ := newFrontServer(t)
	dir := speechDir(t, srv.store.Root(), "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit", "qwen3_tts")
	cfg, _ := json.Marshal(map[string]any{"model_type": "qwen3_tts", "tts_model_type": "voice_design"})
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfg, 0o644); err != nil {
		t.Fatal(err)
	}
	srv.speechDefaults = append(srv.speechDefaults, SpeechModel{kindDesign, "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit", "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit", "2.9 GB"})
	a := &audioRunners{}
	srv.runners.start = a.start
	t.Cleanup(srv.Close)
	return srv, a
}

func legacyDo(t *testing.T, h http.Handler, method, path, contentType, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w
}

const recordingForm = "--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nRIFF\r\n--b--\r\n"

// The Python service's /transcribe, at the ASR address: the recording goes to
// the ASR runner as OpenAI's request, and the answer comes back in the
// Python service's shape.
func TestLegacyTranscribe(t *testing.T) {
	srv, a := legacyServer(t)
	h := srv.SpeechHandler(kindASR)

	w := legacyDo(t, h, "POST", "/transcribe?language=zh", "multipart/form-data; boundary=b", recordingForm)
	var got map[string]any
	json.Unmarshal(w.Body.Bytes(), &got)
	if w.Code != 200 || got["text"] != "你好" || got["language"] != "chinese" || got["duration"] != 1.5 || got["emotion"] != "happy" || got["segments"] != nil {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	ask := a.last()
	if ask.runner != "Qwen3-ASR-1.7B-8bit" || ask.form["model"] != "Qwen3-ASR-1.7B-8bit" || ask.form["language"] != "zh" ||
		ask.form["response_format"] != "verbose_json" || ask.form["file"] != "RIFF" {
		t.Errorf("the runner was asked %+v", ask)
	}

	if w = legacyDo(t, h, "POST", "/transcribe?response_format=text", "multipart/form-data; boundary=b", recordingForm); w.Code != 200 || w.Body.String() != "你好\n" {
		t.Errorf("text: %d %q", w.Code, w.Body.String())
	}
	if w = legacyDo(t, h, "POST", "/transcribe?response_format=srt", "multipart/form-data; boundary=b", recordingForm); w.Code != 400 {
		t.Errorf("srt: %d", w.Code)
	}
	// OpenAI's endpoint at the same address, whatever model it names.
	form := "--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n" + recordingForm
	if w = legacyDo(t, h, "POST", "/v1/audio/transcriptions", "multipart/form-data; boundary=b", form); w.Code != 200 || a.last().runner != "Qwen3-ASR-1.7B-8bit" {
		t.Errorf("OpenAI's: %d %s %+v", w.Code, w.Body.String(), a.last())
	}
	if w = legacyDo(t, h, "GET", "/health", "", ""); !strings.Contains(w.Body.String(), `"status":"healthy"`) || !strings.Contains(w.Body.String(), `"loaded":true`) {
		t.Errorf("health: %s", w.Body.String())
	}
	if w = legacyDo(t, h, "POST", "/synthesize", "application/json", `{"text":"hi"}`); w.Code != 404 {
		t.Errorf("an ASR address speaks: %d", w.Code)
	}
}

// /synthesize and /voices at the TTS and VoiceDesign addresses: each in its
// own model, the request's fields as OpenAI's, the answer WAV.
func TestLegacySynthesize(t *testing.T) {
	srv, a := legacyServer(t)
	tts, design := srv.SpeechHandler(kindTTS), srv.SpeechHandler(kindDesign)

	w := legacyDo(t, tts, "POST", "/synthesize", "application/json", `{"text":"你好","speaker":"uncle_fu","language":"zh","speed":1.2,"instruct":"开心"}`)
	if w.Code != 200 || !strings.HasPrefix(w.Body.String(), "RIFF") || w.Header().Get("Content-Type") != "audio/wav" {
		t.Fatalf("%d %q %v", w.Code, w.Body.String(), w.Header())
	}
	ask := a.last()
	if ask.runner != "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit" || ask.json["input"] != "你好" || ask.json["voice"] != "uncle_fu" ||
		ask.json["instructions"] != "开心" || ask.json["speed"] != 1.2 || ask.json["language"] != "zh" || ask.json["response_format"] != "wav" {
		t.Errorf("the runner was asked %+v", ask)
	}

	w = legacyDo(t, design, "POST", "/synthesize", "application/json", `{"text":"你好","instruct":"一位温和的老人"}`)
	if w.Code != 200 || a.last().runner != "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit" || a.last().json["instructions"] != "一位温和的老人" {
		t.Errorf("design: %d %+v", w.Code, a.last())
	}
	if w = legacyDo(t, design, "POST", "/v1/audio/speech", "application/json", `{"model":"tts-1","input":"hi","instructions":"x"}`); w.Code != 200 || a.last().runner != "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit" {
		t.Errorf("OpenAI's at the design address went to %+v", a.last())
	}
	if w = legacyDo(t, tts, "POST", "/synthesize", "application/json", `{"text":"hi","reference_audio":"UklGRg=="}`); w.Code != 400 {
		t.Errorf("cloning: %d", w.Code)
	}

	w = legacyDo(t, tts, "GET", "/voices", "", "")
	var v struct {
		Model        string           `json:"model"`
		DefaultVoice string           `json:"default_voice"`
		Voices       []map[string]any `json:"voices"`
	}
	json.Unmarshal(w.Body.Bytes(), &v)
	if v.DefaultVoice != "vivian" || len(v.Voices) != 2 || v.Voices[0]["id"] != "uncle_fu" || v.Voices[0]["gender"] != "male" || v.Voices[1]["language"] != "zh" {
		t.Errorf("voices: %s", w.Body.String())
	}
	if w = legacyDo(t, design, "GET", "/voices", "", ""); !strings.Contains(w.Body.String(), `"voices":[]`) || !strings.Contains(w.Body.String(), `"default_voice":null`) {
		t.Errorf("design's voices: %s", w.Body.String())
	}
	if w = legacyDo(t, tts, "GET", "/nothing", "", ""); w.Code != 404 || !strings.Contains(w.Body.String(), "Not Found") {
		t.Errorf("unknown path: %d %s", w.Code, w.Body.String())
	}
}

func TestParseSpeechAddrs(t *testing.T) {
	got, err := ParseSpeechAddrs("asr=:8100, tts=:8101,design=0.0.0.0:8102")
	if err != nil || got["asr"] != ":8100" || got["tts"] != ":8101" || got["design"] != "0.0.0.0:8102" {
		t.Errorf("%v %v", got, err)
	}
	for _, bad := range []string{"asr", "video=:1", "asr=:1,asr=:2", "tts="} {
		if _, err := ParseSpeechAddrs(bad); err == nil {
			t.Errorf("%q passed", bad)
		}
	}
}
