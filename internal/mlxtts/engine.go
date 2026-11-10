package mlxtts

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// Engine serves a Qwen3-TTS model as the server serves any other, for
// /v1/audio/speech: it speaks, and a chat sent to it is refused. One thing is
// said at a time.
type Engine struct {
	m       *Model
	mu      sync.Mutex
	waiting atomic.Int32
	busy    atomic.Int32
	frames  atomic.Int64
	closed  atomic.Bool
}

// Open loads the Qwen3-TTS model in dir on MLX.
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
	m, err := Load(dir)
	if err != nil {
		return nil, err
	}
	return &Engine{m: m}, nil
}

// Speaks is what makes the server send it /v1/audio/speech.
func (e *Engine) Speaks() bool { return true }

// Voices and Languages are the model's.
func (e *Engine) Voices() []string    { return e.m.Voices() }
func (e *Engine) Languages() []string { return e.m.Languages() }
func (e *Engine) SampleRate() int     { return sampleRate }

// Speak says r, handing the audio to onAudio as Model.Speak does.
func (e *Engine) Speak(ctx context.Context, r Request, onAudio func(pcm []float32) error) (Result, error) {
	e.waiting.Add(1)
	e.mu.Lock()
	e.waiting.Add(-1)
	e.busy.Store(1)
	defer func() { e.busy.Store(0); e.mu.Unlock() }()
	if e.closed.Load() {
		return Result{}, errors.New("the model was unloaded")
	}
	res, err := e.m.Speak(ctx, r, onAudio)
	e.frames.Add(int64(res.Frames))
	return res, err
}

var errNoChat = engine.PictureError{Err: errors.New("this is a text to speech model: send it what to say, as /v1/audio/speech does")}

func (e *Engine) ChatFull(context.Context, []chat.Message, engine.GenParams, func(string)) (string, engine.Stats, error) {
	return "", engine.Stats{}, errNoChat
}

func (e *Engine) Chat(context.Context, []chat.Message, engine.GenParams, func(string)) (string, error) {
	return "", errNoChat
}

func (e *Engine) Broken() bool             { return false }
func (e *Engine) ToolFormat() tools.Format { return tools.None }
func (e *Engine) Load() (waiting, busy, slots int) {
	return int(e.waiting.Load()), int(e.busy.Load()), 1
}
func (e *Engine) PromptTokens() int64                { return 0 }
func (e *Engine) EvalTokens() int64                  { return e.frames.Load() }
func (e *Engine) Drafts() (drafted, accepted int64)  { return 0, 0 }
func (e *Engine) Reuse() (tokens int64, entries int) { return 0, 0 }
func (e *Engine) ToolGrammar() (held, redrawn int64) { return 0, 0 }
func (e *Engine) Context() (perSlot, most int)       { return 0, 0 }
func (e *Engine) CountTokens([]chat.Message, engine.GenParams) (int, error) {
	return 0, errNoChat
}

// Close frees the model once the speech under way, if any, is done.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.closed.Swap(true) {
		e.m.Free()
	}
}
