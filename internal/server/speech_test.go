package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/mlxtts"
)

// fakeSpeaker is a text to speech engine: it says anything as pieces of a
// rising ramp, 2,400 samples each, one piece unless streamed.
type fakeSpeaker struct {
	fakeEngine
	mu      sync.Mutex
	last    mlxtts.Request
	pieces  int
	calls   int
	runaway int // the next this many answers talk on to the frame limit
}

func (f *fakeSpeaker) Speaks() bool        { return true }
func (f *fakeSpeaker) Voices() []string    { return []string{"ryan", "serena", "vivian"} }
func (f *fakeSpeaker) Languages() []string { return []string{"chinese", "english", "japanese"} }
func (f *fakeSpeaker) SampleRate() int     { return 24000 }

func (f *fakeSpeaker) Speak(ctx context.Context, r mlxtts.Request, onAudio func([]float32) error) (mlxtts.Result, error) {
	f.mu.Lock()
	f.last = r
	f.calls++
	n := f.pieces
	runaway := f.runaway > 0
	if runaway {
		f.runaway--
	}
	f.mu.Unlock()
	if runaway {
		onAudio(make([]float32, 24000))
		return mlxtts.Result{Frames: r.MaxFrames, Stopped: true}, nil
	}
	if r.Stream == 0 {
		n = 1
	}
	for i := 0; i < n; i++ {
		p := make([]float32, 2400)
		for j := range p {
			p[j] = float32(j) / 2400
		}
		if err := onAudio(p); err != nil {
			return mlxtts.Result{}, err
		}
	}
	return mlxtts.Result{Frames: 5 * n}, nil
}

func (f *fakeSpeaker) request() mlxtts.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.last
}

// newSpeechServer serves "tts", which speaks, and "chat", which does not.
func newSpeechServer(t *testing.T) (*Server, *fakeSpeaker) {
	t.Helper()
	srv, _ := newTestServer(t, "tts", "chat")
	sp := &fakeSpeaker{pieces: 3}
	srv.open = func(path string, _ engine.Options) (generator, error) {
		if strings.TrimSuffix(filepath.Base(path), ".gguf") == "tts" {
			return sp, nil
		}
		return &fakeEngine{path: path, frags: []string{"ok"}}, nil
	}
	return srv, sp
}

func speak(t *testing.T, srv *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/v1/audio/speech", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	return w
}

// Without a format the answer is a WAV, as the Python service's is: a header
// that adds up and 16-bit samples, the voice and the language it was said in
// told in the headers.
func TestSpeechIsAWav(t *testing.T) {
	srv, sp := newSpeechServer(t)
	defer srv.Close()
	w := speak(t, srv, `{"model":"tts","input":"你好，今天天气不错。","voice":"serena"}`)
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "audio/wav" {
		t.Fatalf("%d %s: %s", w.Code, w.Header().Get("Content-Type"), w.Body.String())
	}
	b := w.Body.Bytes()
	if len(b) != 44+2*2400 || string(b[:4]) != "RIFF" || string(b[8:16]) != "WAVEfmt " {
		t.Fatalf("%d bytes, %q", len(b), b[:16])
	}
	if riff, data, rate := binary.LittleEndian.Uint32(b[4:]), binary.LittleEndian.Uint32(b[40:]), binary.LittleEndian.Uint32(b[24:]); riff != uint32(len(b)-8) || data != 2*2400 || rate != 24000 {
		t.Errorf("RIFF size %d, data size %d, rate %d", riff, data, rate)
	}
	if v := int16(binary.LittleEndian.Uint16(b[44+2*1200:])); v != 16383 { // 0.5 of full scale
		t.Errorf("sample 1200 is %d, want half of full scale", v)
	}
	r := sp.request()
	if r.Voice != "serena" || r.Language != "chinese" || r.Temperature != 0.9 || r.Stream != 0 || r.MaxFrames != speechFrames("你好，今天天气不错。") {
		t.Errorf("asked the model for %+v", r)
	}
	if w.Header().Get("X-Kinfer-Voice") != "serena" || w.Header().Get("X-Kinfer-Language") != "chinese" {
		t.Errorf("headers %v", w.Header())
	}
}

// A voice the model lacks is spoken in the nearest it has, as the Python
// service does: a Kokoro voice by its prefix, anything else (OpenAI's alloy)
// in the default; the language comes from the text's script unless named.
func TestSpeechPicksTheVoiceAndLanguage(t *testing.T) {
	srv, sp := newSpeechServer(t)
	defer srv.Close()
	for _, c := range []struct{ body, voice, language string }{
		{`{"model":"tts","input":"Hello there.","voice":"alloy"}`, "vivian", "english"},
		{`{"model":"tts","input":"Hello there.","voice":"af_bella"}`, "serena", "english"},
		{`{"model":"tts","input":"Hello there.","voice":"RYAN"}`, "ryan", "english"},
		{`{"model":"tts","input":"こんにちは、元気ですか。"}`, "vivian", "japanese"},
		{`{"model":"tts","input":"Hello.","language":"zh-CN"}`, "vivian", "chinese"},
		{`{"model":"tts","input":"Bonjour.","language":"fr"}`, "vivian", "auto"}, // the fake has no French
	} {
		if w := speak(t, srv, c.body); w.Code != http.StatusOK {
			t.Fatalf("%s: %d %s", c.body, w.Code, w.Body.String())
		}
		if r := sp.request(); r.Voice != c.voice || r.Language != c.language {
			t.Errorf("%s: voice %q language %q, want %q %q", c.body, r.Voice, r.Language, c.voice, c.language)
		}
	}
	t.Setenv("KINFER_TTS_VOICE", "ryan")
	speak(t, srv, `{"model":"tts","input":"Hello.","voice":"nova"}`)
	if r := sp.request(); r.Voice != "ryan" {
		t.Errorf("KINFER_TTS_VOICE=ryan, spoke in %q", r.Voice)
	}
}

// pcm streams by default, as OpenAI's does: the pieces back to back, no
// header; a streamed WAV starts with a header of unknown length.
func TestSpeechStreams(t *testing.T) {
	srv, sp := newSpeechServer(t)
	defer srv.Close()
	w := speak(t, srv, `{"model":"tts","input":"hello","response_format":"pcm"}`)
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "audio/pcm" || w.Body.Len() != 3*2*2400 {
		t.Fatalf("%d %s, %d bytes", w.Code, w.Header().Get("Content-Type"), w.Body.Len())
	}
	if r := sp.request(); r.Stream != streamFrames {
		t.Errorf("pieces of %d frames, want %d", r.Stream, streamFrames)
	}
	w = speak(t, srv, `{"model":"tts","input":"hello","stream_format":"audio"}`)
	b := w.Body.Bytes()
	if w.Code != http.StatusOK || len(b) != 44+3*2*2400 || binary.LittleEndian.Uint32(b[40:]) != 0xFFFFFFFF {
		t.Fatalf("%d, %d bytes", w.Code, len(b))
	}
}

// stream_format sse sends OpenAI's events: a delta for each piece, the
// audio in base64, then done with the usage.
func TestSpeechSSE(t *testing.T) {
	srv, _ := newSpeechServer(t)
	defer srv.Close()
	// pcm: a WAV's first delta would carry its header, as the Python service's does.
	w := speak(t, srv, `{"model":"tts","input":"hello","stream_format":"sse","response_format":"pcm"}`)
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("%d %s", w.Code, w.Header().Get("Content-Type"))
	}
	deltas, bytesIn := 0, 0
	var done map[string]any
	sc := bufio.NewScanner(bytes.NewReader(w.Body.Bytes()))
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		line, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		var e map[string]any
		json.Unmarshal([]byte(line), &e)
		switch e["type"] {
		case "speech.audio.delta":
			pcm, _ := base64.StdEncoding.DecodeString(e["audio"].(string))
			deltas, bytesIn = deltas+1, bytesIn+len(pcm)
		case "speech.audio.done":
			done = e
		}
	}
	if deltas != 3 || bytesIn != 3*2*2400 || done == nil {
		t.Fatalf("%d deltas, %d bytes, done %v", deltas, bytesIn, done)
	}
	if u := done["usage"].(map[string]any); u["output_tokens"] != float64(15) {
		t.Errorf("usage %v", u)
	}
}

// speed time-stretches: twice as fast is half as long.
func TestSpeechSpeed(t *testing.T) {
	srv, _ := newSpeechServer(t)
	defer srv.Close()
	w := speak(t, srv, `{"model":"tts","input":"hello","speed":2,"response_format":"pcm","stream_format":""}`)
	// pcm streams by default; an empty stream_format is the default too, so
	// ask for a WAV to be stretched.
	if w.Code != http.StatusBadRequest {
		t.Fatalf("speed on a stream: %d %s", w.Code, w.Body.String())
	}
	w = speak(t, srv, `{"model":"tts","input":"hello","speed":2}`)
	if n := (w.Body.Len() - 44) / 2; w.Code != http.StatusOK || n != 1200 {
		t.Fatalf("%d, %d samples, want 1200", w.Code, n)
	}
}

func TestSpeechRefuses(t *testing.T) {
	srv, _ := newSpeechServer(t)
	defer srv.Close()
	for _, c := range []struct {
		body string
		code int
		want string
	}{
		{`{"model":"tts","input":"  "}`, 400, "nothing to say"},
		{`{"input":"hi"}`, 400, "no model"},
		{`{"model":"tts","input":"hi","response_format":"mp3"}`, 400, "wav and pcm"},
		{`{"model":"tts","input":"hi","stream_format":"video"}`, 400, "audio or sse"},
		{`{"model":"chat","input":"hi"}`, 400, "does not speak"},
		{`{"model":"tts","input":"` + strings.Repeat("字", maxSpeechInput+1) + `"}`, 400, "longer than"},
		{`not json`, 400, "send JSON"},
	} {
		w := speak(t, srv, c.body)
		if w.Code != c.code || !strings.Contains(w.Body.String(), c.want) {
			t.Errorf("%.60s: %d %s, want %d %q", c.body, w.Code, w.Body.String(), c.code, c.want)
		}
	}
}

// The voices a model speaks in, and its languages.
func TestSpeechVoices(t *testing.T) {
	srv, _ := newSpeechServer(t)
	defer srv.Close()
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("GET", "/v1/audio/voices?model=tts", nil))
	var out struct{ Voices, Languages []string }
	if json.Unmarshal(w.Body.Bytes(), &out); w.Code != http.StatusOK || len(out.Voices) != 3 || out.Languages[0] != "auto" {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

// An answer that talks on to the end of its frames is said again, with the
// next seed: the second, not the babble, is what goes back.
func TestSpeechThatTalksOnIsSaidAgain(t *testing.T) {
	srv, sp := newSpeechServer(t)
	defer srv.Close()
	sp.runaway = 1
	w := speak(t, srv, `{"model":"tts","input":"hello","seed":41}`)
	if w.Code != http.StatusOK || w.Body.Len() != 44+2*2400 || sp.calls != 2 || *sp.request().Seed != 42 {
		t.Fatalf("%d, %d bytes, %d calls, seed %v", w.Code, w.Body.Len(), sp.calls, sp.request().Seed)
	}
}
