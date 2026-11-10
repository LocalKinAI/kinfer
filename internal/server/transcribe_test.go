package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// wav is seconds of 16 kHz 16-bit mono silence with a WAV header.
func wav(seconds int) []byte {
	n := 16000 * 2 * seconds
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+n))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(1), uint16(1), uint32(16000), uint32(32000), uint16(2), uint16(16)} {
		binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(n))
	b.Write(make([]byte, n))
	return b.Bytes()
}

// transcribe posts a recording and fields as OpenAI's clients send them.
func transcribe(t *testing.T, srv *Server, sound []byte, fields map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	return postAudio(t, srv, "/v1/audio/transcriptions", sound, fields)
}

func postAudio(t *testing.T, srv *Server, path string, sound []byte, fields map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if sound != nil {
		fw, _ := mw.CreateFormFile("file", "speech.wav")
		fw.Write(sound)
	}
	for k, v := range fields {
		mw.WriteField(k, v)
	}
	mw.Close()
	req := httptest.NewRequest("POST", path, &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	return w
}

// hearing loads model in srv and makes its fake engine hear, replying frags.
func hearing(t *testing.T, srv *Server, reg *registry, model string, frags ...string) *fakeEngine {
	t.Helper()
	transcribe(t, srv, wav(1), map[string]string{"model": model}) // loads it; it does not hear yet
	f := reg.get(model)
	f.mu.Lock()
	f.audio, f.frags = true, frags
	f.mu.Unlock()
	return f
}

// The recording goes where a picture would, the prompt is the system message,
// and the language the model opens its reply with is taken off the transcript.
func TestTranscriptionSendsTheSoundAsMedia(t *testing.T) {
	srv, reg := newTestServer(t, "asr")
	defer srv.Close()
	f := hearing(t, srv, reg, "asr", "language English", "<asr_text>", "Hello from kinfer.")
	sound := wav(2)

	w := transcribe(t, srv, sound, map[string]string{"model": "asr", "prompt": "kinfer, Qwen"})
	var out map[string]any
	if json.Unmarshal(w.Body.Bytes(), &out); w.Code != http.StatusOK || out["text"] != "Hello from kinfer." {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	f.mu.Lock()
	msgs, p := f.msgs, f.params
	f.mu.Unlock()
	if len(msgs) != 2 || msgs[0].Role != "system" || msgs[0].Content != "kinfer, Qwen" ||
		msgs[1].Content != engine.ImageMarker || len(msgs[1].Images) != 1 || !bytes.Equal(msgs[1].Images[0], sound) {
		t.Errorf("the model was sent %+v", msgs)
	}
	if p.Temperature != 0 || !p.NoThink || p.Resume != "" || p.MaxTokens != asrTokens {
		t.Errorf("params %+v", p)
	}

	w = transcribe(t, srv, sound, map[string]string{"model": "asr", "response_format": "verbose_json"})
	json.Unmarshal(w.Body.Bytes(), &out)
	if out["language"] != "english" || out["duration"] != 2.0 || out["task"] != "transcribe" {
		t.Errorf("verbose_json: %s", w.Body.String())
	}
	w = transcribe(t, srv, sound, map[string]string{"model": "asr", "response_format": "text"})
	if w.Body.String() != "Hello from kinfer.\n" || !strings.HasPrefix(w.Header().Get("Content-Type"), "text/plain") {
		t.Errorf("text: %q %q", w.Body.String(), w.Header().Get("Content-Type"))
	}
}

// A language the request names starts the model's reply, as Qwen3-ASR is told
// one; a language it cannot name is refused, not guessed.
func TestTranscriptionLanguage(t *testing.T) {
	srv, reg := newTestServer(t, "asr")
	defer srv.Close()
	f := hearing(t, srv, reg, "asr", "你好。")
	for in, want := range map[string]string{"zh": "Chinese", "zh-CN": "Chinese", "Cantonese": "Cantonese", "EN": "English"} {
		w := transcribe(t, srv, wav(1), map[string]string{"model": "asr", "language": in, "response_format": "verbose_json"})
		var out map[string]any
		json.Unmarshal(w.Body.Bytes(), &out)
		f.mu.Lock()
		resume := f.params.Resume
		f.mu.Unlock()
		if w.Code != http.StatusOK || resume != "language "+want+"<asr_text>" || out["text"] != "你好。" || out["language"] != strings.ToLower(want) {
			t.Errorf("%s: %d %s, resume %q", in, w.Code, w.Body.String(), resume)
		}
	}
	if w := transcribe(t, srv, wav(1), map[string]string{"model": "asr", "language": "klingon"}); w.Code != http.StatusBadRequest {
		t.Errorf("an unknown language: %d", w.Code)
	}
}

// What the caller has to fix is a 400: no recording, no model, a format kinfer
// does not write, a model that hears nothing.
func TestTranscriptionRefusals(t *testing.T) {
	srv, reg := newTestServer(t, "asr", "chat")
	defer srv.Close()
	hearing(t, srv, reg, "asr", "x")
	for name, w := range map[string]*httptest.ResponseRecorder{
		"no file":        transcribe(t, srv, nil, map[string]string{"model": "asr"}),
		"empty file":     transcribe(t, srv, []byte{}, map[string]string{"model": "asr"}),
		"no model":       transcribe(t, srv, wav(1), nil),
		"srt":            transcribe(t, srv, wav(1), map[string]string{"model": "asr", "response_format": "srt"}),
		"stream verbose": transcribe(t, srv, wav(1), map[string]string{"model": "asr", "stream": "true", "response_format": "verbose_json"}),
		"temp":           transcribe(t, srv, wav(1), map[string]string{"model": "asr", "temperature": "warm"}),
		"deaf model":     transcribe(t, srv, wav(1), map[string]string{"model": "chat"}),
	} {
		if w.Code != http.StatusBadRequest {
			t.Errorf("%s: %d %s, want 400", name, w.Code, w.Body.String())
		}
	}
	f := reg.get("asr")
	f.mu.Lock()
	f.err = engine.PictureError{Err: errorString("input 0 is not sound llama.cpp reads")}
	f.mu.Unlock()
	if w := transcribe(t, srv, []byte("not sound"), map[string]string{"model": "asr"}); w.Code != http.StatusBadRequest {
		t.Errorf("unreadable sound: %d, want 400", w.Code)
	}
}

type errorString string

func (e errorString) Error() string { return string(e) }

func TestSplitTranscript(t *testing.T) {
	for _, c := range []struct{ reply, forced, lang, text string }{
		{"language Chinese<asr_text>今天天气很好。", "", "Chinese", "今天天气很好。"},
		{" language English <asr_text> Hi. ", "", "English", "Hi."},
		{"language None<asr_text>", "", "None", ""},
		{"Hi.", "English", "English", "Hi."},
		{"no prefix", "", "", "no prefix"},
	} {
		if lang, text := splitTranscript(c.reply, c.forced); lang != c.lang || text != c.text {
			t.Errorf("%q: %q %q, want %q %q", c.reply, lang, text, c.lang, c.text)
		}
	}
	if d := wavSeconds(wav(3)); d != 3 {
		t.Errorf("wavSeconds = %v, want 3", d)
	}
	if d := wavSeconds([]byte("ID3 not a wav")); d != 0 {
		t.Errorf("an mp3: %v", d)
	}
}

// events reads a stream's server-sent events.
func events(t *testing.T, body string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, line := range strings.Split(body, "\n") {
		if d, ok := strings.CutPrefix(line, "data: "); ok {
			var e map[string]any
			if err := json.Unmarshal([]byte(d), &e); err != nil {
				t.Fatalf("event %q: %v", d, err)
			}
			out = append(out, e)
		}
	}
	return out
}

// A streamed transcript arrives as it is written, without the language the
// model opens with, then whole; with a language named, from its first word.
func TestTranscriptionStreams(t *testing.T) {
	srv, reg := newTestServer(t, "asr")
	defer srv.Close()
	hearing(t, srv, reg, "asr", "language", " English", "<asr_text>", "Hello", " from", " kinfer.")
	w := transcribe(t, srv, wav(1), map[string]string{"model": "asr", "stream": "true"})
	ev := events(t, w.Body.String())
	var deltas []string
	for _, e := range ev[:len(ev)-1] {
		if e["type"] != "transcript.text.delta" {
			t.Fatalf("event %v", e)
		}
		deltas = append(deltas, e["delta"].(string))
	}
	last := ev[len(ev)-1]
	if strings.Join(deltas, "") != "Hello from kinfer." || last["type"] != "transcript.text.done" || last["text"] != "Hello from kinfer." ||
		w.Header().Get("Content-Type") != "text/event-stream" {
		t.Errorf("deltas %q, last %v, %s", deltas, last, w.Header().Get("Content-Type"))
	}

	f := reg.get("asr")
	f.mu.Lock()
	f.frags = []string{"language", " is", " what", " we", " speak."}
	f.mu.Unlock()
	ev = events(t, transcribe(t, srv, wav(1), map[string]string{"model": "asr", "stream": "true", "language": "en"}).Body.String())
	if ev[0]["delta"] != "language" || ev[len(ev)-1]["text"] != "language is what we speak." {
		t.Errorf("a named language: %v", ev)
	}
}

// A translation is the transcript, translated by the chat server named for it;
// one heard in English is not sent; with none named it is refused.
func TestTranslation(t *testing.T) {
	var asked []string
	chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model    string
			Messages []struct{ Role, Content string }
		}
		json.NewDecoder(r.Body).Decode(&req)
		asked = append(asked, req.Model+": "+req.Messages[len(req.Messages)-1].Content)
		json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "<think>easy</think>The meeting is at two."}}}})
	}))
	defer chatSrv.Close()

	srv, reg := newTestServer(t, "asr")
	defer srv.Close()
	f := hearing(t, srv, reg, "asr", "language Chinese<asr_text>", "会议在两点。")
	if w := postAudio(t, srv, "/v1/audio/translations", wav(1), map[string]string{"model": "asr"}); w.Code != http.StatusBadRequest {
		t.Errorf("no translator: %d %s", w.Code, w.Body.String())
	}
	t.Setenv("KINFER_TRANSLATE_URL", chatSrv.URL+"/")
	t.Setenv("KINFER_TRANSLATE_MODEL", "ornith")
	w := postAudio(t, srv, "/v1/audio/translations", wav(2), map[string]string{"model": "asr", "response_format": "verbose_json"})
	var out map[string]any
	json.Unmarshal(w.Body.Bytes(), &out)
	if w.Code != http.StatusOK || out["text"] != "The meeting is at two." || out["task"] != "translate" || out["language"] != "english" || out["duration"] != 2.0 {
		t.Errorf("%d %s", w.Code, w.Body.String())
	}
	if len(asked) != 1 || asked[0] != "ornith: 会议在两点。" {
		t.Errorf("the chat server was asked %q", asked)
	}
	f.mu.Lock()
	f.frags = []string{"language English<asr_text>", "Already English."}
	f.mu.Unlock()
	w = postAudio(t, srv, "/v1/audio/translations", wav(1), map[string]string{"model": "asr"})
	if json.Unmarshal(w.Body.Bytes(), &out); out["text"] != "Already English." || len(asked) != 1 {
		t.Errorf("English: %s, asked %q", w.Body.String(), asked)
	}
	if w := postAudio(t, srv, "/v1/audio/translations", wav(1), map[string]string{"model": "asr", "stream": "true"}); w.Code != http.StatusBadRequest {
		t.Errorf("a streamed translation: %d", w.Code)
	}
}

// A model that hears more than words adds the emotion and the sounds, in JSON
// only, as the Python service does; one that does not adds nothing.
func TestTranscriptionCarriesFeelings(t *testing.T) {
	srv, reg := newTestServer(t, "asr")
	defer srv.Close()
	f := hearing(t, srv, reg, "asr", "language Chinese<asr_text>", "太好了！")
	var out map[string]any
	w := transcribe(t, srv, wav(1), map[string]string{"model": "asr"})
	if json.Unmarshal(w.Body.Bytes(), &out); out["emotion"] != nil || out["audio_events"] != nil {
		t.Errorf("no sidecar: %s", w.Body.String())
	}
	f.mu.Lock()
	f.feel, f.events = "happy", []string{"Laughter"}
	f.mu.Unlock()
	w = transcribe(t, srv, wav(1), map[string]string{"model": "asr", "response_format": "verbose_json"})
	out = nil
	json.Unmarshal(w.Body.Bytes(), &out)
	if out["emotion"] != "happy" || fmt.Sprint(out["audio_events"]) != "[Laughter]" || out["text"] != "太好了！" {
		t.Errorf("verbose_json: %s", w.Body.String())
	}
	if w = transcribe(t, srv, wav(1), map[string]string{"model": "asr", "response_format": "text"}); strings.Contains(w.Body.String(), "happy") {
		t.Errorf("text: %q", w.Body.String())
	}
	ev := events(t, transcribe(t, srv, wav(1), map[string]string{"model": "asr", "stream": "true"}).Body.String())
	if last := ev[len(ev)-1]; last["emotion"] != "happy" {
		t.Errorf("stream done: %v", last)
	}
}
