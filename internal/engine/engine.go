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

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/sampling"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// Options configure how a model is loaded.
type Options struct {
	// GPULayers is how many layers to offload. 0 is pure CPU; 99 means all.
	// Note that on small models CPU often wins — measured 123 tok/s CPU vs
	// 114 tok/s Metal on a 0.5B.
	GPULayers int

	// ContextSize is the context ONE conversation gets, in tokens. The context
	// llama.cpp allocates is this times Slots, because it divides a context
	// between sequences.
	ContextSize int

	// Slots is how many conversations share the model, each with its own
	// sequence in the KV cache. They are served by one forward pass per step,
	// so aggregate throughput climbs far faster than per-stream speed falls.
	//
	// Pick from {8, 32, 64, 128}: step time roughly doubles between a batch of
	// 8 and 12 and then stays flat to 32, so 12-16 costs more than 8 and
	// returns less. See the table in README. 0 means 8.
	Slots int

	// PrefixSlots is how many prompt prefixes stay resident so repeat requests
	// skip prefilling them. An agent's system prompt is identical on every turn
	// it takes, and pooling it turns thousands of tokens of prefill into a
	// pointer walk.
	//
	// Each one costs a sequence's worth of KV cache, the same as a slot, so
	// this is memory traded for latency. Negative disables the pool.
	PrefixSlots int

	// Template forces a chat family ("chatml", "llama3", "mistral").
	// Empty means guess from the filename.
	Template string
}

// GenParams control one generation.
type GenParams struct {
	sampling.Params

	// MaxTokens caps the reply. 0 means "until the model stops".
	MaxTokens int

	// Tools are the functions the model may ask to have run. They are folded
	// into the prompt in the form the model was trained on; whether it knows
	// that form at all is SupportsTools.
	Tools []tools.Tool
}

// Defaults for a machine nobody has measured yet.
const (
	// DefaultSlots is the conservative end of the useful range: step time is
	// still cheap at a batch of 8, and the KV cost is eight conversations'
	// worth rather than thirty-two.
	DefaultSlots = 8

	// DefaultContextSize is the context one conversation gets.
	DefaultContextSize = 4096

	// DefaultPrefixSlots pools a few prefixes by default. Four covers a handful
	// of distinct system prompts without doubling the KV cache.
	DefaultPrefixSlots = 4

	// batchCapacity bounds one forward pass: a full prefill chunk plus a token
	// for every slot, with room to spare.
	batchCapacity = 2048

	// maxFragBuffer caps the per-request fragment queue.
	maxFragBuffer = 2048
)

// DefaultGenParams are sane conversational defaults.
func DefaultGenParams() GenParams {
	return GenParams{Params: sampling.DefaultParams(), MaxTokens: 512}
}

// Engine is one loaded model, ready to answer.
//
// Safe for concurrent use, and concurrency is the point: conversations run side
// by side as separate sequences in one KV cache, advanced together by a single
// forward pass per step. Generating a token means reading every weight in the
// model, so serving eight conversations in one pass costs barely more than
// serving one — measured on an M3 Ultra with Qwen2.5-0.5B, 300 tok/s alone
// against 1337 at a batch of eight.
//
// Chat is a thin wrapper over the scheduler, which owns the context outright.
type Engine struct {
	mu    sync.Mutex
	model llama.Model
	lctx  llama.Context
	vocab llama.Vocab
	tpl   *chat.Template
	path  string
	nCtx  int // per conversation
	slots int

	sched *scheduler
}

// Open loads a GGUF file.
func Open(path string, opts Options) (*Engine, error) {
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

	slots := opts.Slots
	if slots <= 0 {
		slots = DefaultSlots
	}
	perSeq := opts.ContextSize
	if perSeq <= 0 {
		perSeq = DefaultContextSize
	}
	prefix := opts.PrefixSlots
	if prefix == 0 {
		prefix = DefaultPrefixSlots
	}
	if prefix < 0 {
		prefix = 0
	}

	cp := llama.DefaultContextParams()
	// Pooled prefixes live in sequence ids above the slots, and each needs a
	// sequence's worth of cache.
	cp.NSeqMax = uint32(slots + prefix)
	// llama.cpp divides n_ctx between sequences, so ask for the total.
	cp.NCtx = uint32(perSeq * (slots + prefix))
	if cp.NBatch < uint32(batchCapacity) {
		cp.NBatch = uint32(batchCapacity)
	}
	if prefix > 0 {
		// A cell can only belong to several sequences in the unified buffer,
		// and sharing cells is the entire point of the pool. llama.cpp warns
		// that unified costs performance when sequences do NOT share a large
		// prefix — here they are chosen precisely because they do.
		cp.KVUnified = 1
	}

	lctx := llama.NewContext(model, cp)
	if lctx == 0 {
		llama.FreeModel(model)
		return nil, fmt.Errorf("create context for %s", path)
	}

	tpl := chat.FromModel(model, path)
	if opts.Template != "" {
		t, err := chat.Get(opts.Template)
		if err != nil {
			llama.FreeContext(lctx)
			llama.FreeModel(model)
			return nil, err
		}
		tpl = t
	}

	// Ask the library what it actually allocated rather than assuming the
	// request took effect. This check caught the bug that made -ctx a no-op for
	// months, and it stays: a struct that silently drifts out of sync with
	// llama.h would fail here instead of somewhere unrecognisable.
	actualCtx := llama.NCtx(lctx)
	wantCtx := perSeq * (slots + prefix)
	if actualCtx < wantCtx {
		llama.FreeContext(lctx)
		llama.FreeModel(model)
		return nil, fmt.Errorf("asked for a %d-token context (%d slots + %d prefix slots x %d) "+
			"but llama.cpp allocated %d "+
			"(llama_context_params may have changed shape — see internal/llama)",
			wantCtx, slots, prefix, perSeq, actualCtx)
	}

	vocab := llama.GetVocab(model)
	e := &Engine{
		model: model,
		lctx:  lctx,
		vocab: vocab,
		tpl:   tpl,
		path:  path,
		nCtx:  actualCtx / (slots + prefix),
		slots: slots,
	}
	e.sched = newScheduler(lctx, vocab, tpl, slots, prefix, e.nCtx, batchCapacity)
	return e, nil
}

// Slots is how many conversations this engine serves at once.
func (e *Engine) Slots() int { return e.slots }

// Close stops the scheduler and releases the model and its context.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.sched != nil {
		// Stop decoding before anything it decodes into is freed.
		e.sched.close()
		e.sched = nil
	}
	if e.lctx != 0 {
		llama.FreeContext(e.lctx)
		e.lctx = 0
	}
	if e.model != 0 {
		llama.FreeModel(e.model)
		e.model = 0
	}
}

// Path is the file this engine was loaded from.
func (e *Engine) Path() string { return e.path }

// Template reports the chat family in use.
func (e *Engine) Template() string { return e.tpl.Name }

// SupportsTools reports whether this model was trained on the tool-call
// convention kinfer speaks. A model that was not will answer in prose no matter
// how the functions are declared.
func (e *Engine) SupportsTools() bool { return e.tpl.SupportsTools() }

// VocabSize is the model's true vocabulary size.
func (e *Engine) VocabSize() int { return int(llama.NVocab(e.vocab)) }

// Chat generates a reply. onToken, if non-nil, receives each fragment as it is
// produced — that is what the HTTP server streams.
//
// ctx cancels generation; a request that goes away releases its slot at the
// next step rather than holding it for the full reply.
//
// Calls run concurrently. Each takes a slot and rides along in the shared
// forward pass, so the cost of the tenth caller is far below ten times the cost
// of the first.
func (e *Engine) Chat(ctx context.Context, msgs []chat.Message, p GenParams, onToken func(string)) (string, error) {
	e.mu.Lock()
	sched := e.sched
	e.mu.Unlock()

	if sched == nil {
		return "", fmt.Errorf("engine for %s is closed", e.path)
	}

	// Size the buffer to hold the whole reply. The scheduler never blocks on a
	// consumer — a client that stopped reading must not stall the other slots —
	// so a buffer that cannot hold a reply would drop the tail of it instead.
	buf := p.MaxTokens
	if buf <= 0 || buf > maxFragBuffer {
		buf = maxFragBuffer
	}

	j := &job{
		ctx:    ctx,
		msgs:   msgs,
		params: p,
		frags:  make(chan string, buf+8),
		done:   make(chan struct{}),
	}
	if err := sched.submit(j); err != nil {
		return "", err
	}

	var out strings.Builder
	for frag := range j.frags {
		out.WriteString(frag)
		if onToken != nil {
			onToken(frag)
		}
	}
	<-j.done
	return out.String(), j.err
}
