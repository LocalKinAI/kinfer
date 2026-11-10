package mlximage

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

// cacheLimit is what MLX may keep of freed buffers in a picture's runner.
const cacheLimit = 10 << 30

// Engine serves a Z-Image model as the server serves any other, for
// /v1/images/generations: it paints, and a chat sent to it is refused. One
// picture is made at a time.
type Engine struct {
	m       *Model
	mu      sync.Mutex
	waiting atomic.Int32
	busy    atomic.Int32
	steps   atomic.Int64
	closed  atomic.Bool
}

// Open loads the Z-Image model in dir on MLX.
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
	// MLX keeps the buffers a step frees for the next to reuse, by default up
	// to the machine's memory. It keeps enough for a 1024² picture's step
	// (7.5 GB beside the weights): cleared each step, every buffer is
	// allocated again and a 1024² picture took 23% longer. Paint clears it
	// once the picture is done.
	mlx.SetCacheLimit(cacheLimit)
	m, err := Load(dir)
	if err != nil {
		return nil, err
	}
	return &Engine{m: m}, nil
}

// Paints is what makes the server send it /v1/images/generations.
func (e *Engine) Paints() bool { return true }

// Paint makes the picture r asks for, as Model.Paint does.
func (e *Engine) Paint(ctx context.Context, r Request) (Result, error) {
	e.waiting.Add(1)
	e.mu.Lock()
	e.waiting.Add(-1)
	e.busy.Store(1)
	defer func() { e.busy.Store(0); e.mu.Unlock() }()
	if e.closed.Load() {
		return Result{}, errors.New("the model was unloaded")
	}
	res, err := e.m.Paint(ctx, r)
	if err == nil {
		e.steps.Add(int64(r.withDefaults().Steps))
	}
	return res, err
}

var errNoChat = engine.PictureError{Err: errors.New("this is a text to image model: send it what to paint, as /v1/images/generations does")}

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
func (e *Engine) EvalTokens() int64                  { return e.steps.Load() }
func (e *Engine) Drafts() (drafted, accepted int64)  { return 0, 0 }
func (e *Engine) Reuse() (tokens int64, entries int) { return 0, 0 }
func (e *Engine) ToolGrammar() (held, redrawn int64) { return 0, 0 }
func (e *Engine) Context() (perSlot, most int)       { return 0, 0 }
func (e *Engine) CountTokens([]chat.Message, engine.GenParams) (int, error) {
	return 0, errNoChat
}

// Close frees the model once the picture under way, if any, is done.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.closed.Swap(true) {
		e.m.Free()
	}
}
