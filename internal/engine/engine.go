// Package engine runs one loaded model.
//
// It owns the pieces that must agree with each other — the llama.cpp context,
// the chat template, the sampler, the vocabulary — and exposes a single Chat
// call. Both the CLI and the HTTP server go through it, so there is one place
// where generation is correct rather than two places that drift.
package engine

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"unicode/utf8"
	"unsafe"

	gollama "github.com/dianlight/gollama.cpp"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/sampling"
)

// Options configure how a model is loaded.
type Options struct {
	// GPULayers is how many layers to offload. 0 is pure CPU; 99 means all.
	// Note that on small models CPU often wins — measured 123 tok/s CPU vs
	// 114 tok/s Metal on a 0.5B.
	GPULayers int

	// ContextSize in tokens. 0 takes the model's own training context.
	ContextSize int

	// Template forces a chat family ("chatml", "llama3", "mistral").
	// Empty means guess from the filename.
	Template string
}

// GenParams control one generation.
type GenParams struct {
	sampling.Params

	// MaxTokens caps the reply. 0 means "until the model stops".
	MaxTokens int
}

// DefaultGenParams are sane conversational defaults.
func DefaultGenParams() GenParams {
	return GenParams{Params: sampling.DefaultParams(), MaxTokens: 512}
}

// Engine is one loaded model, ready to answer.
//
// Safe for concurrent use, but generation is serialised: a llama.cpp context
// holds mutable KV-cache state, so two conversations sharing one context would
// corrupt each other. A future version can keep a pool of contexts; for a local
// runtime, one at a time is the honest tradeoff.
type Engine struct {
	mu    sync.Mutex
	model gollama.LlamaModel
	lctx  gollama.LlamaContext
	vocab uintptr
	tpl   *chat.Template
	path  string
	nCtx  int
}

// Open loads a GGUF file.
func Open(path string, opts Options) (*Engine, error) {
	libDir, err := nativelib.Prepare()
	if err != nil {
		return nil, err
	}
	if err := nativelib.Load(); err != nil {
		return nil, err
	}
	// Bind the entry points gollama omits or gets wrong — real vocabulary size,
	// real detokenizer, real end-of-generation test.
	if err := llama.Bind(libDir); err != nil {
		return nil, err
	}

	mp := gollama.Model_default_params()
	mp.NGpuLayers = int32(opts.GPULayers)

	model, err := gollama.Model_load_from_file(path, mp)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", path, err)
	}

	cp := gollama.Context_default_params()
	if opts.ContextSize > 0 {
		setContextSize(&cp, uint32(opts.ContextSize))
	}

	lctx, err := gollama.Init_from_model(model, cp)
	if err != nil {
		gollama.Model_free(model)
		return nil, fmt.Errorf("create context: %w", err)
	}

	tpl := chat.Detect(path)
	if opts.Template != "" {
		t, err := chat.Get(opts.Template)
		if err != nil {
			gollama.Free(lctx)
			gollama.Model_free(model)
			return nil, err
		}
		tpl = t
	}

	// Trust the library, not the request: see setContextSize for why a requested
	// size can be silently ignored.
	actualCtx := llama.NCtx(uintptr(lctx))
	if opts.ContextSize > 0 && actualCtx < opts.ContextSize {
		gollama.Free(lctx)
		gollama.Model_free(model)
		return nil, fmt.Errorf("asked for a %d-token context but llama.cpp allocated %d "+
			"(gollama's context params may have shifted again — see internal/engine/ctxparams.go)",
			opts.ContextSize, actualCtx)
	}

	return &Engine{
		model: model,
		lctx:  lctx,
		vocab: llama.Vocab(uintptr(model)),
		tpl:   tpl,
		path:  path,
		nCtx:  actualCtx,
	}, nil
}

// Close releases the model and its context.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.lctx != 0 {
		gollama.Free(e.lctx)
		e.lctx = 0
	}
	if e.model != 0 {
		gollama.Model_free(e.model)
		e.model = 0
	}
}

// Path is the file this engine was loaded from.
func (e *Engine) Path() string { return e.path }

// Template reports the chat family in use.
func (e *Engine) Template() string { return e.tpl.Name }

// VocabSize is the model's true vocabulary size.
func (e *Engine) VocabSize() int { return int(llama.NVocab(e.vocab)) }

// Chat generates a reply. onToken, if non-nil, receives each fragment as it is
// produced — that is what the HTTP server streams.
//
// ctx cancels generation between tokens; a request that goes away should not
// keep the model busy.
func (e *Engine) Chat(ctx context.Context, msgs []chat.Message, p GenParams, onToken func(string)) (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Wipe the KV cache. Without this, turn N+1 silently inherits turn N's
	// state and the model answers a question nobody asked.
	gollama.Memory_clear(e.lctx, true)

	prompt := e.tpl.Render(msgs)
	tokens, err := gollama.Tokenize(e.model, prompt, true, true)
	if err != nil {
		return "", fmt.Errorf("tokenize: %w", err)
	}
	if len(tokens) >= e.nCtx {
		return "", fmt.Errorf("prompt is %d tokens but the context holds %d", len(tokens), e.nCtx)
	}

	sampler := sampling.New(p.Params)
	nVocab := llama.NVocab(e.vocab)
	cands := make([]sampling.Candidate, nVocab)

	maxTokens := p.MaxTokens
	if maxTokens <= 0 {
		maxTokens = e.nCtx - len(tokens)
	}

	var (
		out     strings.Builder
		emitted int
		cur     = tokens
	)

	for i := 0; i < maxTokens; i++ {
		if err := ctx.Err(); err != nil {
			return out.String(), err
		}
		if len(tokens)+i >= e.nCtx {
			break // context is full
		}

		if err := gollama.Decode(e.lctx, gollama.Batch_get_one(cur)); err != nil {
			return out.String(), fmt.Errorf("decode at token %d: %w", i, err)
		}

		logits := gollama.Get_logits_ith(e.lctx, -1)
		if logits == nil {
			return out.String(), fmt.Errorf("no logits at token %d", i)
		}
		row := unsafe.Slice(logits, nVocab)
		for j := range row {
			cands[j] = sampling.Candidate{ID: int32(j), Logit: row[j]}
		}

		tok := sampler.Sample(cands)
		if llama.IsEOG(e.vocab, tok) {
			break
		}

		piece := llama.TokenToPiece(e.vocab, tok, false)
		if piece == "" {
			break
		}
		out.WriteString(piece)

		// Some templates leak their stop marker as ordinary text; cut there.
		text, hitStop := e.tpl.TrimStop(out.String())

		// Emit only what has become valid UTF-8. A CJK character spans two or
		// three tokens, and half of one is not printable.
		if onToken != nil && len(text) > emitted {
			pending := text[emitted:]
			if utf8.ValidString(pending) {
				onToken(pending)
				emitted = len(text)
			}
		}

		if hitStop {
			return text, nil
		}
		cur = []gollama.LlamaToken{gollama.LlamaToken(tok)}
	}

	text, _ := e.tpl.TrimStop(out.String())
	if onToken != nil && len(text) > emitted {
		onToken(text[emitted:])
	}
	return text, nil
}
