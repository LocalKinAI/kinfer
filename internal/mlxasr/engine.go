package mlxasr

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// Engine serves a speech model as the server serves any other: a chat whose
// user turn carries the recording where a picture would go (which is what
// /v1/audio/transcriptions sends), the system message what is known about it,
// and a reply started with "language X<asr_text>" forcing a language. The
// model is Qwen3-ASR or SenseVoice; beside Qwen3-ASR, a SenseVoice named by
// KINFER_EMOTION_MODEL tells the emotion and the sounds in each recording, as
// the Python service's --emotion does. One recording is heard at a time.
type Engine struct {
	m       *Model      // Qwen3-ASR, or nil
	sv      *SenseVoice // SenseVoice: the model itself, or Qwen3-ASR's sidecar
	mu      sync.Mutex
	waiting atomic.Int32
	busy    atomic.Int32
	prompt  atomic.Int64
	eval    atomic.Int64
	closed  atomic.Bool
}

// Open loads the Qwen3-ASR or SenseVoice model in dir on MLX.
func Open(dir string, _ engine.Options) (*Engine, error) {
	if err := mlxengine.Usable(); err != nil {
		return nil, err
	}
	libs, err := mlxengine.Libraries()
	if err != nil {
		return nil, err
	}
	if err := mlx.Open(libs); err != nil {
		return nil, err
	}
	if _, err := readSVConfig(dir); err == nil {
		sv, err := LoadSenseVoice(dir)
		if err != nil {
			return nil, err
		}
		return &Engine{sv: sv}, nil
	}
	m, err := Load(dir)
	if err != nil {
		return nil, err
	}
	e := &Engine{m: m}
	if name := os.Getenv("KINFER_EMOTION_MODEL"); name != "" {
		side := name
		if !filepath.IsAbs(side) {
			side = filepath.Join(filepath.Dir(dir), name) // a model in the same store
		}
		if e.sv, err = LoadSenseVoice(side); err != nil {
			m.Free()
			return nil, fmt.Errorf("KINFER_EMOTION_MODEL %s: %w", name, err)
		}
	}
	return e, nil
}

// FeelsAudio says whether Feelings has a SenseVoice to ask.
func (e *Engine) FeelsAudio() bool { return e.sv != nil }

// Feelings is the emotion SenseVoice hears in a recording ("" for none it
// could tell) and the sounds besides speech — what the transcript leaves out.
func (e *Engine) Feelings(ctx context.Context, sound []byte) (emotion string, events []string, err error) {
	if e.sv == nil {
		return "", nil, errors.New("no SenseVoice to hear the emotion with")
	}
	pcm, err := Decode(sound)
	if err != nil {
		return "", nil, err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed.Load() {
		return "", nil, errors.New("the model was unloaded")
	}
	h, err := e.sv.Hear(ctx, pcm, "auto", true)
	return h.Emotion, h.Events, err
}

// svNames are the names a reply opens with for SenseVoice's language codes;
// svCode is the code for one.
var svNames = map[string]string{"zh": "Chinese", "en": "English", "yue": "Cantonese", "ja": "Japanese", "ko": "Korean", "nospeech": "None"}

func svCode(name string) string {
	for code, n := range svNames {
		if strings.EqualFold(n, name) {
			return code
		}
	}
	return "auto"
}

// HearsAudio is what makes the server send it recordings.
func (e *Engine) HearsAudio() bool { return true }

var errNoSound = engine.PictureError{Err: errors.New("this is a speech recognition model: send it a recording, as /v1/audio/transcriptions does")}

func (e *Engine) ChatFull(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, engine.Stats, error) {
	var sound []byte
	system := ""
	for _, m := range msgs {
		if m.Role == "system" && system == "" {
			system = m.Content
		}
		if len(m.Images) > 0 && sound == nil {
			sound = m.Images[0]
		}
	}
	if sound == nil {
		return "", engine.Stats{}, errNoSound
	}
	pcm, err := Decode(sound)
	if err != nil {
		return "", engine.Stats{}, engine.PictureError{Err: fmt.Errorf("input 0: %w", err)}
	}
	language := ""
	if r := p.Resume; strings.HasPrefix(r, "language ") && strings.HasSuffix(r, "<asr_text>") {
		language = strings.TrimSuffix(strings.TrimPrefix(r, "language "), "<asr_text>")
	}
	most := p.MaxTokens
	if most <= 0 {
		most = 8192
	}
	e.waiting.Add(1)
	e.mu.Lock()
	e.waiting.Add(-1)
	e.busy.Store(1)
	defer func() { e.busy.Store(0); e.mu.Unlock() }()
	if e.closed.Load() {
		return "", engine.Stats{}, errors.New("the model was unloaded")
	}
	if e.m == nil { // SenseVoice: its reply opened as Qwen3-ASR's is, so the server reads both alike
		h, err := e.sv.Hear(ctx, pcm, svCode(language), true)
		if err != nil {
			return "", engine.Stats{}, err
		}
		reply := h.Text
		if language == "" {
			reply = "language " + svNames[h.Language] + "<asr_text>" + h.Text
		}
		if onToken != nil {
			onToken(reply)
		}
		e.eval.Add(int64(h.Tokens))
		return reply, engine.Stats{EvalTokens: h.Tokens, PromptEvalDuration: h.Took}, nil
	}
	res, err := e.m.Transcribe(ctx, pcm, system, language, most, onToken)
	e.prompt.Add(int64(res.PromptTokens))
	e.eval.Add(int64(res.Tokens))
	return res.Text, engine.Stats{Truncated: res.Truncated, PromptTokens: res.PromptTokens, EvalTokens: res.Tokens,
		PromptEvalDuration: res.Encode + res.Prefill, EvalDuration: res.Decode}, err
}

func (e *Engine) Chat(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, error) {
	text, _, err := e.ChatFull(ctx, msgs, p, onToken)
	return text, err
}

func (e *Engine) Broken() bool             { return false }
func (e *Engine) ToolFormat() tools.Format { return tools.None }
func (e *Engine) Load() (waiting, busy, slots int) {
	return int(e.waiting.Load()), int(e.busy.Load()), 1
}
func (e *Engine) PromptTokens() int64                { return e.prompt.Load() }
func (e *Engine) EvalTokens() int64                  { return e.eval.Load() }
func (e *Engine) Drafts() (drafted, accepted int64)  { return 0, 0 }
func (e *Engine) Reuse() (tokens int64, entries int) { return 0, 0 }
func (e *Engine) ToolGrammar() (held, redrawn int64) { return 0, 0 }
func (e *Engine) Context() (perSlot, most int) {
	if e.m == nil {
		return 0, 0
	}
	n := e.m.cfg.Thinker.Text.MaxPositions
	return n, n
}
func (e *Engine) CountTokens(_ []chat.Message, _ engine.GenParams) (int, error) {
	return 0, errNoSound
}

// Close frees the model once the transcription under way, if any, is done.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.closed.Swap(true) {
		if e.m != nil {
			e.m.Free()
		}
		if e.sv != nil {
			e.sv.Free()
		}
	}
}
