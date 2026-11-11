package mlxembed

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
)

// maxTokens is the longest input read: what the release was validated for,
// and what Ollama reads of this model.
const maxTokens = 8192

// Embedder embeds texts with the model in a directory, one at a time: MLX
// keeps one scope open per thread of work, and a text of this model is
// milliseconds.
type Embedder struct {
	mu sync.Mutex
	m  *Model
}

// Open loads the model in dir.
func Open(dir string) (*Embedder, error) {
	if err := mlxengine.Usable(); err != nil {
		return nil, err
	}
	libs, err := mlxengine.Libraries()
	if err != nil {
		return nil, err
	}
	if err := mlx.Open(libs); err != nil {
		return nil, fmt.Errorf("MLX libraries (%s): %w", libs, err)
	}
	if err := mlx.ImageOps(); err != nil { // tanh, for GELU
		return nil, err
	}
	m, err := Load(dir)
	if err != nil {
		return nil, err
	}
	return &Embedder{m: m}, nil
}

// Embed is a vector, of length 1, for each input, and the tokens read. With
// truncate an input longer than the model reads is cut to fit, keeping its
// <eos>; without it that is an error.
func (e *Embedder) Embed(ctx context.Context, inputs []engine.EmbedInput, truncate bool) ([][]float32, int, error) {
	seqs := make([][]int32, len(inputs))
	pieces := make([][]piece, len(inputs))
	read := 0
	vocab := int32(e.m.VocabSize())
	for i, in := range inputs {
		if len(in.Media) > 0 {
			toks, ps, err := e.m.withMedia(in)
			if err != nil {
				return nil, 0, engine.InputError{Err: fmt.Errorf("input %d: %w", i, err)}
			}
			if len(toks) > maxTokens {
				return nil, 0, engine.InputError{Err: fmt.Errorf("input %d is %d tokens with its pictures and sound, more than the %d this model reads", i, len(toks), maxTokens)}
			}
			seqs[i], pieces[i] = toks, ps
			read += len(toks)
			continue
		}
		toks := in.Tokens
		if toks == nil {
			toks = e.m.Tokens(in.Text)
		}
		for _, t := range toks {
			if t < 0 || t >= vocab {
				return nil, 0, engine.InputError{Err: fmt.Errorf("input %d: token %d is not in the model's vocabulary of %d", i, t, vocab)}
			}
		}
		if len(toks) == 0 {
			return nil, 0, engine.InputError{Err: fmt.Errorf("input %d is empty", i)}
		}
		if len(toks) > maxTokens {
			if !truncate {
				return nil, 0, engine.InputError{Err: fmt.Errorf("input %d is %d tokens, more than the %d this model reads", i, len(toks), maxTokens)}
			}
			toks = append(append([]int32(nil), toks[:maxTokens-1]...), toks[len(toks)-1])
		}
		seqs[i] = toks
		read += len(toks)
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.m == nil {
		return nil, 0, errors.New("the model was unloaded")
	}
	out := make([][]float32, len(seqs))
	for i, toks := range seqs {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		v, err := e.m.Embed(toks, pieces[i])
		if err != nil {
			return nil, 0, err
		}
		out[i] = v
	}
	return out, read, nil
}

// Close frees the model.
func (e *Embedder) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.m != nil {
		e.m.Free()
		e.m = nil
	}
}
