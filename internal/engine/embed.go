package engine

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"sync"

	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// An embedding model reads a text whole in one pass — an encoder attends both
// ways, so llama.cpp will not split a sequence across passes — and a pass of
// n tokens costs memory with n. embedCtx is the longest text read, unless the
// model was trained on less; longer ones are cut there, as Ollama cuts them.
const embedCtx = 8192

// embedSeqs is how many texts one pass embeds at most.
const embedSeqs = 16

// ErrNotEmbedding is a model that writes text, asked to embed.
var ErrNotEmbedding = errors.New("not an embedding model")

// InputError is an input Embed cannot read — empty, too long, a token outside
// the vocabulary: the caller's to fix, not the model's failure.
type InputError struct{ Err error }

func (e InputError) Error() string { return e.Err.Error() }
func (e InputError) Unwrap() error { return e.Err }

// Embedder makes vectors of texts with an embedding model, one whose header
// says how to pool its tokens into a vector (store.Shape.Embeds). It holds a
// context of its own, beside any chat model's.
//
// Safe for concurrent use; calls take turns.
type Embedder struct {
	mu    sync.Mutex
	model llama.Model
	ctx   llama.Context
	vocab llama.Vocab
	batch *llama.BatchBuilder
	dims  int
	limit int // tokens one text may hold
}

// EmbedInput is one thing to embed: a text, or the tokens of one.
type EmbedInput struct {
	Text   string
	Tokens []int32
}

// OpenEmbedder loads path to embed with.
func OpenEmbedder(path string, opts Options) (*Embedder, error) {
	shape, shErr := store.ReadShape(path)
	if shErr == nil && !shape.Embeds() {
		return nil, fmt.Errorf("%s is %w", filepath.Base(path), ErrNotEmbedding)
	}
	libDir, err := nativelib.Prepare()
	if err != nil {
		return nil, err
	}
	if err := llama.Bind(libDir); err != nil {
		return nil, err
	}

	mp := llama.DefaultModelParams()
	mp.NGpuLayers = int32(opts.GPULayers)
	model := llama.LoadModel(path, mp)
	if model == 0 {
		return nil, fmt.Errorf("load %s: llama.cpp could not read the model (see its output above)", path)
	}
	limit := embedCtx
	if t := llama.TrainedCtx(model); t > 0 {
		limit = min(limit, t)
	}

	cp := llama.DefaultContextParams()
	cp.Embeddings = 1
	cp.NCtx = uint32(limit)
	cp.NBatch, cp.NUbatch = uint32(limit), uint32(limit)
	cp.NSeqMax = embedSeqs
	// The texts of one pass share the cells, however long each is.
	cp.KVUnified = 1
	if shErr == nil && shape.Pooling == 0 {
		cp.PoolingType = llama.PoolingMean // an encoder whose header names none
	}
	ctx := llama.NewContext(model, cp)
	if ctx == 0 {
		llama.FreeModel(model)
		return nil, fmt.Errorf("create an embedding context for %s (see llama.cpp's output above)", path)
	}
	switch llama.PoolingOf(ctx) {
	case llama.PoolingMean, llama.PoolingCLS, llama.PoolingLast:
	default:
		llama.FreeContext(ctx)
		llama.FreeModel(model)
		return nil, fmt.Errorf("%s pools its tokens into no vector: it is %w", filepath.Base(path), ErrNotEmbedding)
	}
	return &Embedder{
		model: model, ctx: ctx, vocab: llama.GetVocab(model),
		batch: llama.NewBatchBuilder(limit), dims: llama.EmbedDims(model), limit: limit,
	}, nil
}

// Dims is how long the vectors are.
func (e *Embedder) Dims() int { return e.dims }

// Embed makes a vector of each input, in order, as the model pools it — not
// normalized — and says how many tokens it read. An input longer than the
// model reads is cut to fit when truncate is set, and refused otherwise.
func (e *Embedder) Embed(ctx context.Context, inputs []EmbedInput, truncate bool) ([][]float32, int, error) {
	seqs := make([][]llama.Token, len(inputs))
	read := 0
	vocab := llama.NVocab(e.vocab)
	for i, in := range inputs {
		toks := in.Tokens
		if toks == nil {
			var err error
			if toks, err = llama.Tokenize(e.vocab, in.Text, true, true); err != nil {
				return nil, 0, InputError{fmt.Errorf("input %d: %w", i, err)}
			}
		}
		for _, t := range toks {
			// llama.cpp asserts on a token outside its vocabulary, which ends
			// the process; tokens given as numbers are checked first.
			if t < 0 || t >= vocab {
				return nil, 0, InputError{fmt.Errorf("input %d: token %d is not in the model's vocabulary of %d", i, t, vocab)}
			}
		}
		if len(toks) == 0 {
			return nil, 0, InputError{fmt.Errorf("input %d is empty", i)}
		}
		if len(toks) > e.limit {
			if !truncate {
				return nil, 0, InputError{fmt.Errorf("input %d is %d tokens, more than the %d this model reads", i, len(toks), e.limit)}
			}
			toks = cut(toks, e.limit, llama.IsControl(e.vocab, toks[len(toks)-1]))
		}
		seqs[i] = toks
		read += len(toks)
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([][]float32, len(inputs))
	for _, pass := range passes(seqs, e.batch.Cap(), embedSeqs) {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		e.batch.Reset()
		for s, toks := range seqs[pass[0]:pass[1]] {
			for p, t := range toks {
				// Pooling reads every token's output, so every one is asked for.
				e.batch.Add(t, int32(p), int32(s), true)
			}
		}
		err := e.batch.Decode(e.ctx)
		if err == nil {
			for s := pass[0]; s < pass[1]; s++ {
				if out[s] = llama.SeqEmbedding(e.ctx, int32(s-pass[0]), e.dims); out[s] == nil {
					err = errors.New("llama.cpp pooled no vector")
					break
				}
			}
		}
		// Each pass starts from nothing: a text embedded once is not context
		// for the next.
		llama.ClearMemory(e.ctx)
		if err != nil {
			return nil, 0, fmt.Errorf("embed: %w", err)
		}
	}
	return out, read, nil
}

// Close frees the model.
func (e *Embedder) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ctx != 0 {
		llama.FreeContext(e.ctx)
		e.ctx = 0
	}
	if e.model != 0 {
		llama.FreeModel(e.model)
		e.model = 0
	}
}

// cut shortens tokens to n from the end, keeping a closing marker — BERT's
// [SEP], a model's end of text — where the text had one, since the model was
// taught to read a text that ends with it.
func cut(tokens []llama.Token, n int, closed bool) []llama.Token {
	if !closed || n < 2 {
		return tokens[:n]
	}
	out := append([]llama.Token(nil), tokens[:n-1]...)
	return append(out, tokens[len(tokens)-1])
}

// passes groups texts, in order, into the passes that embed them: as many as
// fit one pass's tokens, up to seqs of them. A text is never split; each fits
// alone, Embed having cut it to.
func passes(texts [][]llama.Token, tokens, seqs int) [][2]int {
	var out [][2]int
	for start := 0; start < len(texts); {
		end, n := start, 0
		for end < len(texts) && end-start < seqs && (end == start || n+len(texts[end]) <= tokens) {
			n += len(texts[end])
			end++
		}
		out = append(out, [2]int{start, end})
		start = end
	}
	return out
}

// Normalize scales v to unit length, in place, as Ollama's /api/embed and
// OpenAI's vectors come: then a dot product is the cosine. A zero vector
// stays zero.
func Normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
	return v
}

// Shorten keeps the first dims values of v, the size a Matryoshka-trained
// model's vector is still good at (EmbeddingGemma's 512, 256, 128), and
// normalizes what is left. dims of 0, or not shorter, keeps v whole.
func Shorten(v []float32, dims int) []float32 {
	if dims <= 0 || dims >= len(v) {
		return v
	}
	return Normalize(v[:dims])
}
