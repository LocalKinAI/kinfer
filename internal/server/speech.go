package server

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strings"
	"time"
	"unicode"

	"github.com/LocalKinAI/kinfer/internal/mlxtts"
)

// ─── text to speech ──────────────────────────────────────────────────────────
//
// POST /v1/audio/speech, as OpenAI has it — model, input, voice,
// instructions, response_format, speed, stream_format — answered by a text to
// speech model (Qwen3-TTS on MLX). As the Python speech service answers it:
// WAV unless another format is asked for, a voice it does not have (OpenAI's
// alloy...) spoken in the default one, the language told from the text's
// script unless "language" names it, speed by time-stretching.

// maxSpeechInput is OpenAI's limit on what one request says.
const maxSpeechInput = 4096

// streamFrames is how much a streamed piece holds: 7 frames, 0.56 s, as the
// Python service's 0.6 s.
const streamFrames = 7

// speaker is an engine that says things (mlxtts).
type speaker interface {
	Speaks() bool
	Voices() []string
	Languages() []string
	SampleRate() int
	Speak(ctx context.Context, r mlxtts.Request, onAudio func(pcm []float32) error) (mlxtts.Result, error)
}

// speaks is g's model as a speaker, if it is one: the engine, not the
// wrapper that reloads it.
func speaks(g generator) (speaker, bool) {
	if rt, ok := g.(*retrying); ok {
		g = rt.current()
	}
	sp, ok := g.(speaker)
	return sp, ok && sp.Speaks()
}

// voiceAliases map a Kokoro voice's prefix — its language and sex — to the
// nearest Qwen3-TTS voice, so that clients sending "zf_xiaoxiao" keep working,
// as the Python service maps them.
var voiceAliases = map[string]string{"zf": "vivian", "zm": "uncle_fu", "af": "serena", "bf": "serena",
	"am": "ryan", "bm": "aiden", "jf": "ono_anna"}

// pickVoice is the voice to speak in: the one asked for if the model has it,
// a Kokoro voice's nearest, else the default (KINFER_TTS_VOICE, or vivian).
func pickVoice(asked string, voices []string) string {
	have := map[string]bool{}
	for _, v := range voices {
		have[v] = true
	}
	a := strings.ToLower(strings.TrimSpace(asked))
	if have[a] {
		return a
	}
	if len(a) >= 2 && have[voiceAliases[a[:2]]] {
		return voiceAliases[a[:2]]
	}
	if d := strings.ToLower(os.Getenv("KINFER_TTS_VOICE")); have[d] {
		return d
	}
	if have["vivian"] {
		return "vivian"
	}
	if len(voices) > 0 {
		return voices[0]
	}
	return a
}

// speechLanguages are the model's language names by ISO 639-1 code.
var speechLanguages = map[string]string{"zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean",
	"de": "german", "fr": "french", "ru": "russian", "pt": "portuguese", "es": "spanish", "it": "italian"}

// scriptLanguage tells the language from the text's script as the Python
// service does: kana Japanese (kanji are Han too, so it goes first), hangul
// Korean, Han Chinese, anything else English.
func scriptLanguage(text string) string {
	for _, r := range text {
		if unicode.In(r, unicode.Hiragana, unicode.Katakana) {
			return "ja"
		}
	}
	for _, r := range text {
		if unicode.Is(unicode.Hangul, r) {
			return "ko"
		}
	}
	for _, r := range text {
		if unicode.Is(unicode.Han, r) {
			return "zh"
		}
	}
	return "en"
}

// speechLanguage is the language name the model is told: the one asked for
// (a code, "zh-CN", or a name), or the text's; one it does not have, auto.
func speechLanguage(asked, text string, have []string) string {
	a := strings.ToLower(strings.TrimSpace(asked))
	if i := strings.IndexAny(a, "-_"); i > 0 {
		a = a[:i]
	}
	if a == "" {
		a = scriptLanguage(text)
	}
	if a == "auto" {
		return "auto"
	}
	name := speechLanguages[a]
	if name == "" {
		name = a
	}
	for _, h := range have {
		if h == name {
			return name
		}
	}
	return "auto"
}

// speechFormats are what kinfer writes: WAV and OpenAI's pcm (16-bit
// little-endian mono at 24 kHz, no header).
var speechFormats = map[string]string{"wav": "audio/wav", "pcm": "audio/pcm"}

func speechError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"message": msg, "type": "invalid_request_error"}})
}

// pcm16 is audio [-1, 1] as 16-bit little-endian samples.
func pcm16(pcm []float32) []byte {
	b := make([]byte, 2*len(pcm))
	for i, v := range pcm {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(int16(max(-1, min(1, v))*32767)))
	}
	return b
}

// wavHeader is a 16-bit mono WAV header for n bytes of samples; n < 0 for a
// stream of unknown length, which WAV readers take as "to the end".
func wavHeader(rate, n int) []byte {
	size, riff := uint32(0xFFFFFFFF), uint32(0xFFFFFFFF)
	if n >= 0 {
		size, riff = uint32(n), uint32(36+n)
	}
	h := make([]byte, 44)
	copy(h, "RIFF")
	binary.LittleEndian.PutUint32(h[4:], riff)
	copy(h[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(h[16:], 16)
	binary.LittleEndian.PutUint16(h[20:], 1) // PCM
	binary.LittleEndian.PutUint16(h[22:], 1) // mono
	binary.LittleEndian.PutUint32(h[24:], uint32(rate))
	binary.LittleEndian.PutUint32(h[28:], uint32(2*rate))
	binary.LittleEndian.PutUint16(h[32:], 2)
	binary.LittleEndian.PutUint16(h[34:], 16)
	copy(h[36:], "data")
	binary.LittleEndian.PutUint32(h[40:], size)
	return h
}

type speechRequest struct {
	Model          string   `json:"model"`
	Input          string   `json:"input"`
	Voice          string   `json:"voice"`
	Instructions   string   `json:"instructions"`
	ResponseFormat string   `json:"response_format"`
	Speed          *float64 `json:"speed"`
	StreamFormat   string   `json:"stream_format"`
	Language       string   `json:"language"`    // kinfer's: the language to speak in
	Temperature    *float32 `json:"temperature"` // kinfer's: 0.9 by default, 0 for the most likely speech
}

func (s *Server) handleSpeech(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req speechRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		speechError(w, http.StatusBadRequest, "send JSON with what to say in \"input\" and the model in \"model\": "+err.Error())
		return
	}
	text := strings.TrimSpace(req.Input)
	switch {
	case req.Model == "":
		speechError(w, http.StatusBadRequest, "no model in \"model\"")
		return
	case text == "":
		speechError(w, http.StatusBadRequest, "nothing to say in \"input\"")
		return
	case len([]rune(text)) > maxSpeechInput:
		speechError(w, http.StatusBadRequest, fmt.Sprintf("\"input\" is longer than %d characters", maxSpeechInput))
		return
	}
	format := strings.ToLower(req.ResponseFormat)
	if format == "" {
		format = "wav"
	}
	if _, ok := speechFormats[format]; !ok {
		speechError(w, http.StatusBadRequest, fmt.Sprintf("response_format %q: kinfer writes wav and pcm", format))
		return
	}
	stream := req.StreamFormat
	if stream == "" && format == "pcm" {
		stream = "audio" // as OpenAI's pcm comes
	}
	if stream != "" && stream != "audio" && stream != "sse" {
		speechError(w, http.StatusBadRequest, "stream_format must be audio or sse")
		return
	}
	speed := 1.0
	if req.Speed != nil && *req.Speed != 0 {
		speed = max(0.5, min(2, *req.Speed))
	}
	if stream != "" && math.Abs(speed-1) > 1e-3 {
		speechError(w, http.StatusBadRequest, "a streamed answer is spoken at its own pace: leave speed out, or stream_format")
		return
	}

	eng, release, err := s.acquire(req.Model)
	if err != nil {
		writeJSON(w, acquireStatus(err), map[string]any{"error": map[string]string{"message": err.Error()}})
		return
	}
	defer release()
	sp, ok := speaks(eng)
	if !ok {
		speechError(w, http.StatusBadRequest, fmt.Sprintf("%s does not speak: speech needs a text to speech model, such as Qwen3-TTS CustomVoice on a Mac", req.Model))
		return
	}
	mr := mlxtts.Request{Text: text, Voice: pickVoice(req.Voice, sp.Voices()), Instruct: strings.TrimSpace(req.Instructions),
		Language: speechLanguage(req.Language, text, sp.Languages()), Temperature: 0.9}
	if req.Temperature != nil {
		mr.Temperature = *req.Temperature
	}
	if stream != "" {
		mr.Stream = streamFrames
	}
	rate := sp.SampleRate()
	w.Header().Set("X-Kinfer-Voice", mr.Voice)
	w.Header().Set("X-Kinfer-Language", mr.Language)
	start := time.Now()

	if stream == "" {
		var pcm []float32
		res, err := sp.Speak(r.Context(), mr, func(p []float32) error { pcm = append(pcm, p...); return nil })
		if err != nil {
			speechFailed(w, err)
			return
		}
		pcm = mlxtts.Stretch(pcm, speed, rate)
		body := pcm16(pcm)
		if format == "wav" {
			body = append(wavHeader(rate, len(body)), body...)
		}
		w.Header().Set("Content-Type", speechFormats[format])
		w.Write(body)
		logSpeech(mr, res, len(pcm), rate, start)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		speechError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	samples, sent := 0, false
	send := func(b []byte) error {
		if stream == "sse" {
			sendEvent(w, flusher, map[string]any{"type": "speech.audio.delta", "audio": base64.StdEncoding.EncodeToString(b)})
		} else {
			if _, err := w.Write(b); err != nil {
				return err
			}
			flusher.Flush()
		}
		return r.Context().Err()
	}
	res, err := sp.Speak(r.Context(), mr, func(p []float32) error {
		if !sent && stream == "audio" {
			w.Header().Set("Content-Type", speechFormats[format])
		}
		b := pcm16(p)
		if !sent && format == "wav" {
			b = append(wavHeader(rate, -1), b...)
		}
		sent = true
		samples += len(p)
		return send(b)
	})
	switch {
	case err != nil && !sent:
		speechFailed(w, err)
		return
	case err != nil:
		if stream == "sse" {
			sendEvent(w, flusher, map[string]any{"type": "error", "error": map[string]string{"message": err.Error()}})
		}
		log.Printf("speech: stopped after %.2f s: %v", float64(samples)/float64(rate), err)
		return
	}
	if stream == "sse" {
		sendEvent(w, flusher, map[string]any{"type": "speech.audio.done",
			"usage": map[string]any{"input_tokens": len([]rune(text)), "output_tokens": res.Frames, "total_tokens": len([]rune(text)) + res.Frames}})
	}
	logSpeech(mr, res, samples, rate, start)
}

func speechFailed(w http.ResponseWriter, err error) {
	if errors.Is(err, context.Canceled) {
		return
	}
	log.Printf("speech failed: %v", err)
	if strings.HasPrefix(err.Error(), "voice ") || err.Error() == "nothing to say" {
		speechError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]string{"message": err.Error()}})
}

func logSpeech(r mlxtts.Request, res mlxtts.Result, samples, rate int, start time.Time) {
	secs := float64(samples) / float64(rate)
	took := time.Since(start)
	stopped := ""
	if res.Stopped {
		stopped = ", cut off at the frame limit"
	}
	log.Printf("speech: %d chars as %s (%s): %.2f s of audio in %v (%.1f× real time), first audio in %v%s",
		len([]rune(r.Text)), r.Voice, r.Language, secs, took.Round(time.Millisecond), secs/max(took.Seconds(), 1e-9),
		res.First.Round(time.Millisecond), stopped)
}

// handleVoices is GET /v1/audio/voices?model=: the voices and languages a
// text to speech model speaks.
func (s *Server) handleVoices(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")
	if model == "" {
		speechError(w, http.StatusBadRequest, "name the model: /v1/audio/voices?model=...")
		return
	}
	eng, release, err := s.acquire(model)
	if err != nil {
		writeJSON(w, acquireStatus(err), map[string]any{"error": map[string]string{"message": err.Error()}})
		return
	}
	defer release()
	sp, ok := speaks(eng)
	if !ok {
		speechError(w, http.StatusBadRequest, model+" does not speak")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"voices": sp.Voices(), "languages": append([]string{"auto"}, sp.Languages()...)})
}
