package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
)

// ─── speech to text ──────────────────────────────────────────────────────────
//
// POST /v1/audio/transcriptions, as OpenAI has it: a multipart form with the
// recording in "file" and the model in "model". A speech recognition model
// whose projector hears sound — Qwen3-ASR — takes the recording where a
// picture would go, through the same libmtmd path, and its reply is the
// transcript. Nothing on the chat path is changed for it.

// maxRecording is the largest recording taken: about half an hour of 16 kHz
// 16-bit mono WAV, more than a speech model's context holds anyway.
const maxRecording = 64 << 20

// asrTokens bounds a transcript. Speech runs a few tokens a second in any
// language, so this only stops a reply that repeats itself.
const asrTokens = 8192

// hearer is an engine whose projector takes sound (engine.Engine).
type hearer interface{ HearsAudio() bool }

// hears says whether the model behind g takes sound: the engine, not the
// wrapper that reloads it (retry.go).
func hears(g generator) bool {
	if rt, ok := g.(*retrying); ok {
		g = rt.current()
	}
	h, ok := g.(hearer)
	return ok && h.HearsAudio()
}

// feeler is an engine that also tells the emotion in a recording and the
// sounds besides speech: SenseVoice, alone or beside Qwen3-ASR (mlxasr).
type feeler interface {
	FeelsAudio() bool
	Feelings(ctx context.Context, sound []byte) (emotion string, events []string, err error)
}

// feelings is what g's model hears besides the words, as the Python service
// adds it: "emotion" and "audio_events", each only when there is one. It never
// fails a transcript; a failure leaves them out.
func feelings(ctx context.Context, g generator, sound []byte) map[string]any {
	if rt, ok := g.(*retrying); ok {
		g = rt.current()
	}
	f, ok := g.(feeler)
	if !ok || !f.FeelsAudio() {
		return nil
	}
	emotion, events, err := f.Feelings(ctx, sound)
	if err != nil {
		log.Printf("transcription: the emotion was not heard: %v", err)
		return nil
	}
	out := map[string]any{}
	if emotion != "" {
		out["emotion"] = emotion
	}
	if len(events) > 0 {
		out["audio_events"] = events
	}
	return out
}

// asrPrefix is how Qwen3-ASR opens its reply: the language it heard, then the
// transcript — "language Chinese<asr_text>…". Forcing a language is starting
// the reply with it.
var asrPrefix = regexp.MustCompile(`^\s*language\s+([^<\s]*)\s*<asr_text>`)

// asrLanguages are the languages Qwen3-ASR names, by their ISO 639 codes.
var asrLanguages = map[string]string{
	"zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
	"fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
	"ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
	"tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
	"da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino",
	"fa": "Persian", "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}

// asrLanguage is the name Qwen3-ASR gives the language a request named, by
// its code ("zh", "zh-CN") or its English name ("Chinese"); "" for none.
func asrLanguage(v string) (string, error) {
	v = strings.TrimSpace(v)
	if v == "" {
		return "", nil
	}
	code := strings.ToLower(v)
	if i := strings.IndexAny(code, "-_"); i > 0 {
		code = code[:i]
	}
	if name, ok := asrLanguages[code]; ok {
		return name, nil
	}
	if code == "tl" {
		return "Filipino", nil
	}
	for _, name := range asrLanguages {
		if strings.EqualFold(name, v) {
			return name, nil
		}
	}
	return "", fmt.Errorf("language %q is not one the speech model names; give an ISO 639-1 code such as zh or en, or leave it out for the model to tell", v)
}

// splitTranscript takes the language a reply opens with off it; forced is the
// language the request named, which the reply then does not repeat.
func splitTranscript(reply, forced string) (language, text string) {
	if m := asrPrefix.FindStringSubmatchIndex(reply); m != nil {
		return reply[m[2]:m[3]], strings.TrimSpace(reply[m[1]:])
	}
	return forced, strings.TrimSpace(reply)
}

// wavSeconds is how long a WAV recording runs, from its header; 0 for any
// other format, or a header it cannot read.
func wavSeconds(b []byte) float64 {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return 0
	}
	var rate, align uint32
	for off := 12; off+8 <= len(b); {
		id, size := string(b[off:off+4]), binary.LittleEndian.Uint32(b[off+4:off+8])
		body := off + 8
		switch {
		case id == "fmt " && body+16 <= len(b):
			rate = binary.LittleEndian.Uint32(b[body+4 : body+8])
			align = uint32(binary.LittleEndian.Uint16(b[body+12 : body+14]))
		case id == "data" && rate > 0 && align > 0:
			n := uint64(size)
			if rest := uint64(len(b) - body); n > rest || size == 0xffffffff {
				n = rest // a header written before the length was known
			}
			return float64(n/uint64(align)) / float64(rate)
		}
		off = body + int(size) + int(size&1)
	}
	return 0
}

func transcriptionError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"message": msg, "type": "invalid_request_error"}})
}

func (s *Server) handleTranscription(w http.ResponseWriter, r *http.Request) {
	s.audioText(w, r, false)
}

// handleTranslation is POST /v1/audio/translations: the recording's words in
// English. A speech recognition model writes down what it hears in the language
// it hears — Qwen3-ASR does not translate, told to or not — so the transcript
// is then translated by a chat model on another server, as the speech server
// runs apart from the chat one: KINFER_TRANSLATE_URL is its address and
// KINFER_TRANSLATE_MODEL the model. A recording already in English is not sent.
func (s *Server) handleTranslation(w http.ResponseWriter, r *http.Request) { s.audioText(w, r, true) }

// translator is the chat server and model a translation goes to, or "".
func translator() (url, model string) {
	return strings.TrimRight(os.Getenv("KINFER_TRANSLATE_URL"), "/"), os.Getenv("KINFER_TRANSLATE_MODEL")
}

func (s *Server) audioText(w http.ResponseWriter, r *http.Request, translate bool) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if s.speech != SpeechOff { // answered in a speech model's runner (runners.go)
		if body, asked, ok := speechBody(w, r, maxRecording+1<<20); ok {
			s.proxySpeech(w, r, kindASR, asked, body)
		}
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxRecording+1<<20)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		var big *http.MaxBytesError
		if errors.As(err, &big) {
			transcriptionError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("the recording is larger than %d MiB", maxRecording>>20))
			return
		}
		transcriptionError(w, http.StatusBadRequest, "send a multipart form with the recording in \"file\" and the model in \"model\": "+err.Error())
		return
	}
	defer r.MultipartForm.RemoveAll()

	file, _, err := r.FormFile("file")
	if err != nil {
		transcriptionError(w, http.StatusBadRequest, "no recording in \"file\"")
		return
	}
	sound, err := io.ReadAll(io.LimitReader(file, maxRecording+1))
	file.Close()
	switch {
	case err != nil:
		transcriptionError(w, http.StatusBadRequest, "reading the recording: "+err.Error())
		return
	case len(sound) == 0:
		transcriptionError(w, http.StatusBadRequest, "the recording in \"file\" is empty")
		return
	case len(sound) > maxRecording:
		transcriptionError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("the recording is larger than %d MiB", maxRecording>>20))
		return
	}

	model := s.runnerModel(r, r.FormValue("model"))
	if model == "" {
		transcriptionError(w, http.StatusBadRequest, "no model in \"model\"")
		return
	}
	format := r.FormValue("response_format")
	switch format {
	case "":
		format = "json"
	case "json", "text", "verbose_json":
	default:
		transcriptionError(w, http.StatusBadRequest, fmt.Sprintf("response_format %q: kinfer writes json, text and verbose_json", format))
		return
	}
	stream := r.FormValue("stream") == "true"
	switch {
	case stream && translate:
		transcriptionError(w, http.StatusBadRequest, "a translation is not streamed: it is made once the whole transcript is")
		return
	case stream && format == "verbose_json":
		transcriptionError(w, http.StatusBadRequest, "a streamed transcript is json or text, not verbose_json")
		return
	}
	trURL, trModel := translator()
	if translate && (trURL == "" || trModel == "") {
		transcriptionError(w, http.StatusBadRequest, "a translation needs a chat model to translate the transcript with: start this kinfer with KINFER_TRANSLATE_URL, a chat server's address such as http://127.0.0.1:11590, and KINFER_TRANSLATE_MODEL, a model it serves")
		return
	}
	language, err := asrLanguage(r.FormValue("language"))
	if err != nil {
		transcriptionError(w, http.StatusBadRequest, err.Error())
		return
	}
	params := engine.DefaultGenParams()
	params.Temperature = 0
	if v := r.FormValue("temperature"); v != "" {
		t, err := strconv.ParseFloat(v, 32)
		if err != nil {
			transcriptionError(w, http.StatusBadRequest, fmt.Sprintf("temperature %q is not a number", v))
			return
		}
		params.Temperature = float32(t)
	}
	params.NoThink = true
	params.MaxTokens = asrTokens
	if language != "" {
		params.Resume = "language " + language + "<asr_text>"
	}

	eng, release, err := s.acquire(model)
	if err != nil {
		writeJSON(w, acquireStatus(err), map[string]any{"error": map[string]string{"message": err.Error()}})
		return
	}
	defer release()
	if !hears(eng) {
		transcriptionError(w, http.StatusBadRequest, fmt.Sprintf("%s takes no sound: a transcript needs a speech recognition model, such as Qwen3-ASR (a GGUF with its audio projector, or MLX's conversion on a Mac)", model))
		return
	}

	// Qwen3-ASR's own prompt: what is known about the recording — names, terms —
	// as the system message, and the sound alone as the user's.
	msgs := []chat.Message{
		{Role: "system", Content: r.FormValue("prompt")},
		{Role: "user", Content: engine.ImageMarker, Images: [][]byte{sound}},
	}

	var sse *asrStream
	var onToken func(string)
	if stream {
		flusher, ok := w.(http.Flusher)
		if !ok {
			transcriptionError(w, http.StatusInternalServerError, "streaming unsupported")
			return
		}
		sse = &asrStream{forced: language != "", emit: func(delta string) {
			sendEvent(w, flusher, map[string]any{"type": "transcript.text.delta", "delta": delta})
		}}
		onToken = sse.add
	}
	reply, st, err := eng.ChatFull(r.Context(), msgs, params, onToken)
	if err != nil {
		if sse != nil && sse.sent {
			sendEvent(w, w.(http.Flusher), map[string]any{"type": "error", "error": map[string]string{"message": err.Error()}})
			return
		}
		writeOpenAIGenError(w, err)
		return
	}
	heard, text := splitTranscript(reply, language)
	extra := feelings(r.Context(), eng, sound)
	if sse != nil {
		if !sse.started && text != "" {
			sse.emit(text)
		}
		done := map[string]any{"type": "transcript.text.done", "text": text,
			"usage": map[string]any{"type": "tokens", "input_tokens": st.PromptTokens, "output_tokens": st.EvalTokens,
				"total_tokens": st.PromptTokens + st.EvalTokens}}
		for k, v := range extra {
			done[k] = v
		}
		sendEvent(w, w.(http.Flusher), done)
		return
	}

	task := "transcribe"
	if translate {
		task = "translate"
		release() // the speech model is done; the chat server does the rest
		if !strings.EqualFold(heard, "English") && strings.TrimSpace(text) != "" {
			if text, err = translateText(r.Context(), trURL, trModel, text); err != nil {
				transcriptionError(w, http.StatusBadGateway, "translating the transcript: "+err.Error())
				return
			}
		}
		heard = "english"
	}

	switch format {
	case "text":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		io.WriteString(w, text+"\n")
	case "verbose_json":
		out := map[string]any{"task": task, "language": strings.ToLower(heard), "text": text}
		if d := wavSeconds(sound); d > 0 {
			out["duration"] = d
		}
		for k, v := range extra {
			out[k] = v
		}
		writeJSON(w, http.StatusOK, out)
	default:
		out := map[string]any{"text": text}
		for k, v := range extra { // in JSON only, as the Python service has them: text has nowhere for them
			out[k] = v
		}
		writeJSON(w, http.StatusOK, out)
	}
}

func sendEvent(w http.ResponseWriter, f http.Flusher, v any) {
	if w.Header().Get("Content-Type") != "text/event-stream" {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
	}
	b, _ := json.Marshal(v)
	fmt.Fprintf(w, "data: %s\n\n", b)
	f.Flush()
}

// asrStream passes a reply on as it is written, less the "language X<asr_text>"
// it opens with: held back until it has gone by, or until what was written
// cannot be it.
type asrStream struct {
	forced  bool // a language was named, so the reply does not open with one
	started bool
	sent    bool
	held    strings.Builder
	emit    func(string)
}

func (a *asrStream) add(piece string) {
	if a.started || a.forced {
		a.started = true
		a.send(piece)
		return
	}
	a.held.WriteString(piece)
	b := a.held.String()
	if m := asrPrefix.FindStringIndex(b); m != nil {
		a.started = true
		if rest := strings.TrimLeft(b[m[1]:], " "); rest != "" {
			a.send(rest)
		}
		return
	}
	t := strings.TrimLeft(b, " \n")
	if len(b) <= 64 && (strings.HasPrefix("language ", t) || strings.HasPrefix(t, "language")) {
		return // it may still be the opening
	}
	a.started = true
	a.send(b)
}

func (a *asrStream) send(s string) {
	if s != "" {
		a.sent = true
		a.emit(s)
	}
}

// translateText asks the chat model at base for text in English.
func translateText(ctx context.Context, base, model, text string) (string, error) {
	body, _ := json.Marshal(map[string]any{
		"model": model, "temperature": 0, "stream": false, "reasoning_effort": "none",
		"messages": []map[string]string{
			{"role": "system", "content": "Translate the user's text into English. Write the translation and nothing else."},
			{"role": "user", "content": text},
		},
	})
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("%s answered %d: %.300s", base, resp.StatusCode, raw)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil || len(out.Choices) == 0 {
		return "", fmt.Errorf("%s answered without a reply: %.300s", base, raw)
	}
	_, answer := chat.SplitThinking(out.Choices[0].Message.Content)
	return strings.TrimSpace(answer), nil
}
