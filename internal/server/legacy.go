package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"sort"
	"strings"
	"time"
)

// ─── the Python audio service's addresses ────────────────────────────────────
//
// Before kinfer heard and spoke, the box's speech was localkin-service-audio
// (`kin audio serve`): one Python process a model, each at a port of its own —
// :8100 Qwen3-ASR, :8101 Qwen3-TTS's preset voices, :8102 its VoiceDesign —
// answering its own API (/transcribe, /synthesize, /voices) beside OpenAI's.
// Programs on the box and devices on the network were written against those
// addresses. -speech-addr has kinfer answer at them in its place: each address
// one kind of speech, in the model of that kind this server would choose, in
// that API and in OpenAI's, the model a request names ignored as the Python
// service ignored it. The speech runs in the same runners as the speech at
// the main address.

// legacyModels are the Python service's names for the models it served at
// each kind's port, which its /health and / reported.
var legacyModels = map[string]string{kindASR: "qwen3-asr:1.7b", kindTTS: "qwen3-tts:0.6b", kindDesign: "qwen3-tts:1.7b-voicedesign"}

// qwen3Voices are Qwen3-TTS CustomVoice's voices as the Python service lists
// them: a name to show, the language each is native in (every one speaks
// all ten), and whose voice it is.
var qwen3Voices = map[string][3]string{
	"vivian":   {"Vivian · 明亮女声", "zh", "female"},
	"serena":   {"Serena · 温柔女声", "zh", "female"},
	"uncle_fu": {"Uncle Fu · 成熟男声", "zh", "male"},
	"dylan":    {"Dylan · 北京话男声", "zh", "male"},
	"eric":     {"Eric · 四川话男声", "zh", "male"},
	"ryan":     {"Ryan · English male", "en", "male"},
	"aiden":    {"Aiden · English male", "en", "male"},
	"ono_anna": {"Ono Anna · 日本語女声", "ja", "female"},
	"sohee":    {"Sohee · 한국어 여성", "ko", "female"},
}

// ParseSpeechAddrs reads -speech-addr: kind=address pairs, comma separated,
// the kinds asr, tts and design.
func ParseSpeechAddrs(v string) (map[string]string, error) {
	out := map[string]string{}
	for _, part := range strings.Split(v, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		kind, addr, ok := strings.Cut(part, "=")
		kind, addr = strings.TrimSpace(kind), strings.TrimSpace(addr)
		if !ok || addr == "" {
			return nil, fmt.Errorf("-speech-addr %q: write kind=address, as asr=:8100", part)
		}
		if _, known := legacyModels[kind]; !known {
			return nil, fmt.Errorf("-speech-addr %q: the kinds are asr, tts and design", part)
		}
		if out[kind] != "" {
			return nil, fmt.Errorf("-speech-addr names %s twice", kind)
		}
		out[kind] = addr
	}
	return out, nil
}

// SetSpeechAddrs records the extra speech addresses, so that -speech keep
// readies their models too. Call before serving.
func (s *Server) SetSpeechAddrs(addrs map[string]string) { s.speechAddrs = addrs }

// SpeechHandler answers the speech of one kind at an address of its own, in
// the Python service's API and OpenAI's.
func (s *Server) SpeechHandler(kind string) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			writeJSON(w, http.StatusNotFound, map[string]string{"detail": "Not Found"})
			return
		}
		endpoints := map[string]string{"GET /": "This information", "GET /health": "Health check", "GET /models": "Loaded models info"}
		if kind == kindASR {
			endpoints["POST /transcribe"] = "Speech to text"
			endpoints["POST /v1/audio/transcriptions"] = "Speech to text, as OpenAI's"
		} else {
			endpoints["POST /synthesize"] = "Text to speech"
			endpoints["GET /voices"] = "Voices of the TTS model"
			endpoints["POST /v1/audio/speech"] = "Text to speech, as OpenAI's"
		}
		writeJSON(w, http.StatusOK, map[string]any{"name": "kinfer, in the place of LocalKin Service Audio",
			"model": s.legacyModelName(kind), "type": legacyType(kind), "status": "running", "endpoints": endpoints})
	})
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		path, fetch := s.speechModelFor(kind, "")
		if path == "" && (fetch == nil || !s.autoPull) {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{"status": "unhealthy", "model": legacyModels[kind], "loaded": false,
				"error": fmt.Sprintf("this machine has no %s model", kindNames[kind])})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "healthy", "model": s.legacyModelName(kind), "loaded": path != "" && s.runners.up(path)})
	})
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		name := s.legacyModelName(kind)
		writeJSON(w, http.StatusOK, map[string]any{"loaded_models": []string{name}, "current_model": name})
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": []map[string]any{
			{"id": s.legacyModelName(kind), "object": "model", "owned_by": "kinfer"}}})
	})
	if kind == kindASR {
		mux.HandleFunc("/transcribe", func(w http.ResponseWriter, r *http.Request) { s.legacyTranscribe(w, r) })
		openai := func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "POST only", http.StatusMethodNotAllowed)
				return
			}
			if body, _, ok := speechBody(w, r, maxRecording+1<<20); ok {
				if path, ok := s.speechPath(w, kindASR, ""); ok {
					s.proxySpeechTo(w, r, kindASR, path, body)
				}
			}
		}
		mux.HandleFunc("/v1/audio/transcriptions", openai)
		mux.HandleFunc("/v1/audio/translations", openai)
	} else {
		mux.HandleFunc("/synthesize", func(w http.ResponseWriter, r *http.Request) { s.legacySynthesize(w, r, kind) })
		mux.HandleFunc("/voices", func(w http.ResponseWriter, r *http.Request) { s.legacyVoices(w, r, kind) })
		mux.HandleFunc("/v1/audio/speech", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "POST only", http.StatusMethodNotAllowed)
				return
			}
			body, _, ok := speechBody(w, r, 1<<20)
			if !ok {
				return
			}
			// The Python service answered in WAV when no format was named,
			// not OpenAI's mp3, and its clients rely on it; so does kinfer's
			// own endpoint, so the request goes as it came.
			if path, ok := s.speechPath(w, kind, ""); ok {
				s.proxySpeechTo(w, r, kind, path, body)
			}
		})
		mux.HandleFunc("/v1/audio/voices", func(w http.ResponseWriter, r *http.Request) {
			if path, ok := s.speechPath(w, kind, ""); ok {
				s.proxySpeechTo(w, r, kind, path, nil)
			}
		})
	}
	return s.count(mux)
}

func legacyType(kind string) string {
	if kind == kindASR {
		return "stt"
	}
	return "tts"
}

// legacyModelName is the store's name for the model answering kind, or the
// Python service's while there is none here.
func (s *Server) legacyModelName(kind string) string {
	if path, _ := s.speechModelFor(kind, ""); path != "" {
		return modelName(path)
	}
	return legacyModels[kind]
}

// legacyError is an error as FastAPI writes one, which the Python service's
// clients read.
func legacyError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"detail": msg})
}

// askRunner sends one request to the runner of the model at path and reads the
// whole answer: the Python service's endpoints answer in shapes of their own,
// made from the answer of OpenAI's.
func (s *Server) askRunner(path, method, urlPath, contentType string, body []byte) (int, http.Header, []byte, error) {
	rn, release, err := s.runners.acquire(path)
	if err != nil {
		return 0, nil, nil, fmt.Errorf("%s could not be started: %v", modelName(path), err)
	}
	defer release()
	req, err := http.NewRequest(method, rn.p.base+urlPath, bytes.NewReader(body))
	if err != nil {
		return 0, nil, nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	req.Header.Set(runnerHeader, path)
	resp, err := (&http.Client{Timeout: 30 * time.Minute}).Do(req)
	if err != nil {
		return 0, nil, nil, fmt.Errorf("%s stopped answering: %v", modelName(path), err)
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(resp.Body)
	return resp.StatusCode, resp.Header, out, err
}

// runnerMessage is the message of an error a runner answered with.
func runnerMessage(body []byte) string {
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &e) == nil && e.Error.Message != "" {
		return e.Error.Message
	}
	return strings.TrimSpace(string(body))
}

// legacyTranscribe is the Python service's POST /transcribe: the recording in
// the form's "file", language and response_format as query parameters (or
// form fields), and the answer {"text", "language", "duration"} with the
// emotion and the sounds heard when there are any; or the text alone.
func (s *Server) legacyTranscribe(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		legacyError(w, http.StatusMethodNotAllowed, "Method Not Allowed")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxRecording+1<<20)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		legacyError(w, http.StatusBadRequest, "send the recording as a multipart form's \"file\": "+err.Error())
		return
	}
	defer r.MultipartForm.RemoveAll()
	file, hdr, err := r.FormFile("file")
	if err != nil {
		legacyError(w, http.StatusUnprocessableEntity, "no recording in \"file\"")
		return
	}
	sound, err := io.ReadAll(io.LimitReader(file, maxRecording+1))
	file.Close()
	if err != nil {
		legacyError(w, http.StatusBadRequest, "reading the recording: "+err.Error())
		return
	}
	format := r.FormValue("response_format")
	switch format {
	case "", "json":
		format = "json"
	case "text":
	default:
		legacyError(w, http.StatusBadRequest, fmt.Sprintf("response_format %q: kinfer writes json and text here; srt, vtt and markdown need segment timestamps, which Qwen3-ASR does not give", format))
		return
	}
	path, ok := s.speechPath(w, kindASR, "")
	if !ok {
		return
	}
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	name := "recording.wav"
	if hdr != nil && hdr.Filename != "" {
		name = hdr.Filename
	}
	fw, _ := mw.CreateFormFile("file", name)
	fw.Write(sound)
	mw.WriteField("model", modelName(path))
	if lang := r.FormValue("language"); lang != "" {
		mw.WriteField("language", lang)
	}
	if format == "text" {
		mw.WriteField("response_format", "text")
	} else {
		mw.WriteField("response_format", "verbose_json")
	}
	mw.Close()
	status, _, out, err := s.askRunner(path, http.MethodPost, "/v1/audio/transcriptions", mw.FormDataContentType(), b.Bytes())
	switch {
	case err != nil:
		legacyError(w, http.StatusBadGateway, err.Error())
		return
	case status != http.StatusOK:
		legacyError(w, status, runnerMessage(out))
		return
	}
	if format == "text" {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Write([]byte(strings.TrimSpace(string(out)) + "\n"))
		return
	}
	var v map[string]any
	if err := json.Unmarshal(out, &v); err != nil {
		legacyError(w, http.StatusBadGateway, "the transcript did not read: "+err.Error())
		return
	}
	ans := map[string]any{"text": v["text"], "language": v["language"]}
	if d, ok := v["duration"]; ok {
		ans["duration"] = d
	}
	for _, k := range []string{"emotion", "audio_events"} {
		if x, ok := v[k]; ok && x != nil && x != "" {
			ans[k] = x
		}
	}
	writeJSON(w, http.StatusOK, ans)
}

// legacySynthesize is the Python service's POST /synthesize: JSON with the
// words in "text", the voice in "speaker", "language", "speed" and
// "instruct" (the tone, or a VoiceDesign model's voice); the answer is WAV.
func (s *Server) legacySynthesize(w http.ResponseWriter, r *http.Request, kind string) {
	if r.Method != http.MethodPost {
		legacyError(w, http.StatusMethodNotAllowed, "Method Not Allowed")
		return
	}
	var req struct {
		Text           string   `json:"text"`
		Speaker        string   `json:"speaker"`
		Language       string   `json:"language"`
		Speed          *float64 `json:"speed"`
		Instruct       string   `json:"instruct"`
		ReferenceAudio string   `json:"reference_audio"`
		Seed           *uint64  `json:"seed"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		legacyError(w, http.StatusUnprocessableEntity, "send JSON with what to say in \"text\": "+err.Error())
		return
	}
	if req.ReferenceAudio != "" {
		legacyError(w, http.StatusBadRequest, "kinfer does not clone voices: it speaks in Qwen3-TTS's voices by name, or in one told in words")
		return
	}
	path, ok := s.speechPath(w, kind, "")
	if !ok {
		return
	}
	body := map[string]any{"model": modelName(path), "input": req.Text, "response_format": "wav"}
	if req.Speaker != "" {
		body["voice"] = req.Speaker
	}
	if req.Language != "" {
		body["language"] = req.Language
	}
	if req.Speed != nil {
		body["speed"] = *req.Speed
	}
	if req.Instruct != "" {
		body["instructions"] = req.Instruct
	}
	if req.Seed != nil {
		body["seed"] = *req.Seed
	}
	b, _ := json.Marshal(body)
	status, hdr, out, err := s.askRunner(path, http.MethodPost, "/v1/audio/speech", "application/json", b)
	switch {
	case err != nil:
		legacyError(w, http.StatusBadGateway, err.Error())
		return
	case status != http.StatusOK:
		legacyError(w, status, runnerMessage(out))
		return
	}
	for _, h := range []string{"X-Kinfer-Voice", "X-Kinfer-Language"} {
		if v := hdr.Get(h); v != "" {
			w.Header().Set(h, v)
		}
	}
	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Content-Disposition", "attachment; filename=speech.wav")
	w.Write(out)
}

// legacyVoices is the Python service's GET /voices: each voice with a name to
// show, the language it is native in and whose it is; none for a VoiceDesign
// model, whose voice is told in words.
func (s *Server) legacyVoices(w http.ResponseWriter, r *http.Request, kind string) {
	path, ok := s.speechPath(w, kind, "")
	if !ok {
		return
	}
	status, _, out, err := s.askRunner(path, http.MethodGet, "/v1/audio/voices?model=x", "", nil)
	switch {
	case err != nil:
		legacyError(w, http.StatusBadGateway, err.Error())
		return
	case status != http.StatusOK:
		legacyError(w, status, runnerMessage(out))
		return
	}
	var v struct {
		Voices []string `json:"voices"`
	}
	json.Unmarshal(out, &v)
	sort.Strings(v.Voices)
	voices := []map[string]any{}
	for _, id := range v.Voices {
		info, known := qwen3Voices[id]
		if !known {
			info = [3]string{id, "", ""}
		}
		voice := map[string]any{"id": id, "name": info[0], "language": nil, "gender": nil}
		if info[1] != "" {
			voice["language"], voice["gender"] = info[1], info[2]
		}
		voices = append(voices, voice)
	}
	var def any
	if len(v.Voices) > 0 {
		def = pickVoice("", v.Voices)
	}
	writeJSON(w, http.StatusOK, map[string]any{"model": modelName(path), "multilingual": true, "default_voice": def, "voices": voices})
}

// ServeSpeechAddrs answers each extra speech address, binding it again every
// ten seconds while something else holds it — the Python service it replaces
// may still be stopping — so the main address is never held up by one.
func (s *Server) ServeSpeechAddrs() {
	for kind, addr := range s.speechAddrs {
		go func() {
			h := s.SpeechHandler(kind)
			for {
				err := http.ListenAndServe(addr, h)
				log.Printf("speech: %s at %s: %v; trying again in 10 s", kindNames[kind], addr, err)
				time.Sleep(10 * time.Second)
			}
		}()
	}
}
