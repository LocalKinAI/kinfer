// Package engine runs one loaded model.
//
// It owns the pieces that must agree with each other — the llama.cpp context,
// the chat template, the sampler, the vocabulary — and exposes a single Chat
// call. Both the CLI and the HTTP server go through it, so there is one place
// where generation is correct rather than two places that drift.
package engine

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/sampling"
	"github.com/LocalKinAI/kinfer/internal/store"
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
	// between sequences. 0 sizes it to the model and the memory left after
	// its weights (AutoSize).
	ContextSize int

	// Slots is how many conversations share the model, each with its own
	// sequence in the KV cache. They are served by one forward pass per step,
	// so aggregate throughput climbs far faster than per-stream speed falls.
	//
	// Pick from {8, 32, 64, 128}: step time roughly doubles between a batch of
	// 8 and 12 and then stays flat to 32, so 12-16 costs more than 8 and
	// returns less. See the table in README. 0 means 8, or fewer when the
	// memory left after the weights cannot give 8 a useful context.
	Slots int

	// PrefixSlots is how many prompt prefixes stay resident so repeat requests
	// skip prefilling them. An agent's system prompt is identical on every turn
	// it takes, and pooling it turns thousands of tokens of prefill into a
	// pointer walk.
	//
	// Each one costs a sequence's worth of KV cache, the same as a slot, so
	// this is memory traded for latency. 0 sizes it with the rest — and where
	// memory is short, lets it share the slots' cells instead of giving up
	// context; negative disables the pool.
	PrefixSlots int

	// MaxGenerate bounds how long one reply may take, in wall-clock time.
	//
	// The token budget is not this bound: `num_predict` is whatever the caller
	// sent, and Ollama's convention for it includes "until the model stops". A
	// caller's own context deadline is the real authority; this covers the ones
	// that set none, which is how a single request could hold a slot for half
	// an hour on the 125B. 0 means DefaultMaxGenerate; negative disables it.
	MaxGenerate time.Duration

	// MaxWait bounds how long a request may sit in the queue before being
	// refused instead of given a slot. 0 means DefaultMaxWait; negative
	// disables it.
	MaxWait time.Duration

	// MaxQueue is how many requests may wait for a slot before the engine
	// starts refusing them.
	//
	// It is a decision about honesty rather than capacity. Unbounded queueing
	// does not serve more requests, it only hides how far behind the server is,
	// and it hides it in the worst possible form — an open connection that
	// looks like a slow model. A refusal is something a fleet can act on.
	//
	// The right depth is roughly how much work you would still want done by the
	// time the front of the queue is answered; past that the answers are stale
	// anyway. 0 means DefaultMaxQueue.
	MaxQueue int

	// Template forces a chat family ("chatml", "llama3", "mistral").
	// Empty means guess from the filename.
	Template string

	// FixedLayout keeps the slots and context the model loaded with. Without
	// it a prompt too long for a slot gets a longer one: the context is rebuilt
	// as fewer slots holding the same number of tokens while that prompt runs,
	// and rebuilt as it was once nothing needs them. See flex.go.
	FixedLayout bool

	// Batch is how many tokens of a prompt the GPU takes in one pass
	// (llama.cpp's n_ubatch). 0 sizes it the way Ollama does: 2048, 1024 or
	// 512, by the context and by how much of the memory the model and its
	// cache take. See autoUbatch.
	Batch int
}

// GenParams control one generation.
type GenParams struct {
	sampling.Params

	// MaxTokens caps the reply. 0 (the default) or negative means "until the
	// model stops or the slot's context is full" — Ollama's num_predict -1.
	MaxTokens int

	// Tools are the functions the model may ask to have run. They are folded
	// into the prompt in the form the model was trained on; whether it knows
	// that form at all is SupportsTools.
	Tools []tools.Tool

	// NoThink asks a reasoning model to answer without a think block, by
	// rendering the template's own disable switch into the prompt. It is what
	// a caller's think:false means, and it is different from merely not
	// showing the thinking: hidden thinking still costs every token and every
	// second it takes, which with a token budget leaves nothing for the reply.
	NoThink bool
}

// Stats are the token counts and timings for one reply.
//
// They exist because a runtime that cannot be measured cannot be chosen. Ollama
// reports these on every reply and everything that benchmarks a local model
// reads them — `ollama run --verbose`, dashboards, benchmark scripts — so a
// runtime that answers with zeros is not fast, it is unmeasurable, and against
// a self-reporting Ollama the only comparison left is a stopwatch.
//
// The durations are wall clock for one request, not exclusive GPU time. A
// request shares each forward pass with every other busy slot, so under load
// its eval duration counts their tokens as well as its own. That is the honest
// answer to "how long did this caller wait", and the number Ollama reports too.
type Stats struct {
	// Truncated records that the reply stopped because it ran out of budget
	// rather than because the model finished. Reporting the two the same way
	// tells a caller a cut-off answer is complete — and a reasoning model that
	// spends its whole budget thinking returns nothing at all, which is
	// baffling unless the reason is given.
	Truncated bool

	// PromptTokens is the whole rendered prompt, counting the part the prefix
	// pool supplied from cache rather than prefilled. Ollama counts
	// prompt_eval_count the same way, so a pooled prefix shows up as an
	// unchanged count against a tiny PromptEvalDuration — which is the saving,
	// stated, rather than a prompt that appears not to have existed.
	PromptTokens int

	// EvalTokens is how many tokens the model generated.
	EvalTokens int

	// PromptEvalDuration runs from the moment a slot takes the request to the
	// first sampled token — the prefill and nothing else. Time spent queueing
	// for a slot is not in it; that shows up only in the caller's own total,
	// which is the right place for it, since a queue says how busy the server
	// was rather than how fast the model is.
	PromptEvalDuration time.Duration

	// Timeout records that the reply stopped because it ran out of wall-clock
	// time. Like Truncated it is a reason rather than a failure: the text
	// produced so far is real and is returned, and what a caller must not be
	// told is that a reply cut short finished normally.
	Timeout bool

	// ReloadDuration is time spent loading a replacement model in the middle of
	// this request, after the one it started against died.
	//
	// It is separate from the caller's own load measurement because that one is
	// taken when the model is acquired, before generation begins, and this
	// happens after. Adding them is the caller's job and is not optional: on a
	// retried request the first acquire found a resident model and reports
	// microseconds, while the reload it hides can be most of a minute. Left out,
	// the components of a forty-second request all read as milliseconds — worse
	// than the missing zeros this whole set of fields removed, because a
	// plausible small number does not look wrong.
	ReloadDuration time.Duration

	// EvalDuration starts at that first sampled token, so it excludes prefill.
	// That is the boundary Ollama draws, and drawing it anywhere else would
	// make EvalTokens/EvalDuration — the tokens-per-second everything quotes —
	// mean something different here than it does there.
	EvalDuration time.Duration
}

// Defaults for a machine nobody has measured yet.
const (
	// DefaultSlots is the conservative end of the useful range: step time is
	// still cheap at a batch of 8, and the KV cost is eight conversations'
	// worth rather than thirty-two.
	DefaultSlots = 8

	// DefaultContextSize is the context one conversation gets when nothing
	// better is known — no accelerator to budget against, or a GGUF whose
	// header lacks the attention geometry. With both known, the context is
	// sized to the model and the memory instead: see AutoSize.
	DefaultContextSize = 4096

	// DefaultPrefixSlots pools a few prefixes by default. Four covers a handful
	// of distinct system prompts without doubling the KV cache.
	DefaultPrefixSlots = 4

	// DefaultMaxQueue is sixteen requests of waiting per slot.
	//
	// Deep enough that a fleet arriving all at once is absorbed rather than
	// half-refused — 150 agents against the default 8 slots fits with room to
	// spare — and shallow enough that the wait at the back is measured in
	// minutes rather than hours. Past that an answer arrives long after the
	// agent that asked for it has moved on, and the queue is storing work
	// nobody still wants.
	DefaultMaxQueue = 128

	// batchCapacity bounds one forward pass: a full prefill chunk plus a token
	// for every slot, with room to spare.
	batchCapacity = 2048

	// maxFragBuffer caps the per-request fragment queue.
	maxFragBuffer = 2048
)

// DefaultGenParams are sane conversational defaults.
//
// No token ceiling: MaxTokens 0 means the reply runs until the model stops or
// its slot's context is full, which is Ollama's default too (num_predict -1).
// The scheduler was written on that premise — what a runaway reply costs is
// time, and the wall clock bounds that (see deadline.go) — but this default
// said 512, so a caller that named no budget was cut off at 512 tokens anyway.
// Measured: a card-writing request whose JSON ran past 512 tokens came back
// truncated mid-string, done_reason "length", and parsed as zero cards.
func DefaultGenParams() GenParams {
	return GenParams{Params: sampling.DefaultParams(), MaxTokens: 0}
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
	vocab llama.Vocab
	tpl   *chat.Template
	path  string
	nCtx  int // per conversation, in the layout it loaded with
	slots int // likewise; the scheduler's layout can change, see Slots

	// sched owns the llama.cpp context: it rebuilds it when the layout
	// changes, so the context it holds when it stops is the one to free.
	sched *scheduler

	// broken is set when llama.cpp reports a fatal decode. The context cannot
	// be used again, so the server discards this engine rather than serving
	// the same error to every request that follows.
	broken atomic.Bool
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

	// Read the accelerator's free memory before anything is allocated, so the
	// model's and the context's real costs can be measured rather than guessed.
	bud := newBudget()

	model := llama.LoadModel(path, mp)
	if model == 0 {
		return nil, fmt.Errorf("load %s: llama.cpp could not read the model (see its output above)", path)
	}
	bud.afterModel = bud.sample()

	// Whatever the operator left at zero is sized here, from the model's
	// shape and the memory the weights actually left — see autosize.go.
	sized := opts.Slots <= 0 || opts.ContextSize <= 0
	shape, _ := store.ReadShape(path)

	// Slots given and the context left to us, with flex on: the total the
	// slots share is sized per model the way Ollama sizes one request's
	// context (SizeTotal), and flex gives one long prompt all of it.
	byTotal := opts.Slots > 0 && opts.ContextSize <= 0 && !opts.FixedLayout
	ctx := opts.ContextSize
	if byTotal {
		t, err := SizeTotal(shape, bud.dev.Total, bud.afterModel, opts.Slots)
		if err != nil {
			llama.FreeModel(model)
			return nil, err
		}
		ctx = t.PerSeq
		one := ollamaDefaultCtx(bud.dev.Total)
		if shape.TrainedCtx > 0 {
			one = min(one, shape.TrainedCtx)
		}
		log.Printf("sizing: %d tokens of context for %d slots of %d — Ollama would give one request %d here; "+
			"the cache should cost about %s of the %s left. -ctx overrides this.",
			t.PerSeq*t.Slots, t.Slots, t.PerSeq, one, gib(uint64(t.Estimated)), gib(bud.afterModel))
	}

	plan, err := AutoSize(shape, bud.afterModel, opts.Slots, opts.PrefixSlots, ctx)
	if err != nil {
		llama.FreeModel(model)
		return nil, err
	}
	slots, prefix, perSeq, shared := plan.Slots, plan.Prefix, plan.PerSeq, plan.SharedPool

	// The micro-batch, from the longest conversation the cache can hold and
	// the memory the model and its cache are predicted to take.
	longest := perSeq
	if !opts.FixedLayout && slots > 1 {
		longest = perSeq * slots
		if shape.TrainedCtx > 0 {
			longest = min(longest, shape.TrainedCtx)
		}
	}
	ubAuto := opts.Batch <= 0
	ub := min(max(opts.Batch, 1), batchCapacity)
	if ubAuto {
		ub = autoUbatch(longest, sub(bud.atStart, bud.afterModel)+uint64(plan.Estimated), bud.atStart)
	}
	// What was left for us to choose. An explicit flag is never overridden, so
	// each shrinking step below is taken only where the operator gave no number.
	slotsAuto, prefixAuto, ctxAuto := opts.Slots <= 0, opts.PrefixSlots == 0, opts.ContextSize <= 0
	if sized && !byTotal {
		trained := "unknown"
		if shape.TrainedCtx > 0 {
			trained = fmt.Sprint(shape.TrainedCtx)
		}
		log.Printf("sizing: %d slots + %d prefix x %d tokens each%s — the model was trained for %s, "+
			"the cache should cost about %s of the %s left. -slots and -ctx override this.",
			slots, prefix, perSeq, sharedNote(shared), trained, gib(uint64(plan.Estimated)), gib(bud.afterModel))
	} else if prefixAuto {
		log.Printf("sizing: %d prefix entries%s around the %d slots x %d tokens asked for — "+
			"the pool gets only what they leave; -prefix overrides", prefix, sharedNote(shared), slots, perSeq)
	}

	floor := ctxFloor
	if shape.TrainedCtx > 0 {
		floor = min(floor, shape.TrainedCtx)
	}

	var lctx llama.Context
sizing:
	for {
		lctx = newContext(model, slots, prefix, perSeq, shared, ub)
		if lctx == 0 {
			llama.FreeModel(model)
			return nil, fmt.Errorf("create context for %s", path)
		}
		bud.afterCtx = bud.sample()
		// The estimate is for plain attention. A hybrid model's cache holds a
		// per-sequence state the header does not describe, and on one such
		// model it cost two and a half times the estimate. When the measured
		// cost leaves less than the floor, and the size was ours to choose,
		// choose smaller: a context is seconds to rebuild, the weights stay.
		// The pool counts as ours to choose even when -slots and -ctx did not:
		// an automatic pool around fixed conversations is exactly what left the
		// box at 0.0 GiB, and this loop never ran for it.
		// A total sized like Ollama's is kept down to fitReserve, not the
		// warning line: see fitReserve for why 1.6 GiB is not too little.
		enough := uint64(thinHeadroom)
		if byTotal {
			enough = fitReserve
		}
		if !(sized || prefixAuto || (ubAuto && ub > ubatchDefault)) || !bud.known || bud.afterCtx >= enough {
			break
		}
		measured := fmt.Sprintf("sizing: %d slots + %d prefix x %d tokens%s measured %s, leaving %s",
			slots, prefix, perSeq, sharedNote(shared), gib(sub(bud.afterModel, bud.afterCtx)), gib(bud.afterCtx))
		// Smaller the way AutoSize would have chosen had its estimate been
		// right: a conversation goes before context does. This is where that
		// matters most, because a hybrid's per-sequence state is what the
		// estimate undercounts — and dropping a sequence frees a whole one of
		// them. Halving alone took the box to 4 slots of 4096.
		switch {
		// A bigger micro-batch is speed, not capacity: it goes before
		// anything a conversation would notice.
		case ubAuto && ub > ubatchDefault:
			ub = lowerUbatch(ub)
			log.Printf("%s — a micro-batch of %d instead", measured, ub)
		case perSeq/2 < floor && slotsAuto && slots > 1:
			slots /= 2
			prefix = min(prefix, max(slots/2, 1))
			log.Printf("%s — %d slots + %d prefix instead, keeping %d tokens each", measured, slots, prefix, perSeq)
		// When the context is the operator's, halving it is not ours to do, so
		// the pool goes first whatever the floor.
		case (perSeq/2 < floor || !ctxAuto) && prefixAuto && prefix > 0 && !shared:
			// The pool's own cells go before the pool does: sharing the slots'
			// cells, it still carries an agent's turns forward.
			shared = true
			log.Printf("%s — the prefix pool shares the slots' cells instead, keeping %d tokens each", measured, perSeq)
		case (perSeq/2 < floor || !ctxAuto) && prefixAuto && prefix > 0:
			prefix--
			log.Printf("%s — %d prefix instead, keeping %d tokens each", measured, prefix, perSeq)
		case ctxAuto && perSeq/2 >= 2048:
			perSeq /= 2
			log.Printf("%s — halving the context", measured)
		default:
			break sizing
		}
		llama.FreeContext(lctx)
	}
	bud.report(slots, prefix, perSeq, shared)

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
	cells := Plan{Slots: slots, Prefix: prefix, SharedPool: shared}.Cells()
	wantCtx := perSeq * cells
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
		vocab: vocab,
		tpl:   tpl,
		path:  path,
		nCtx:  actualCtx / cells,
		slots: slots,
	}
	queue := opts.MaxQueue
	if queue <= 0 {
		queue = DefaultMaxQueue
	}
	// A model with recurrent layers can adopt a pooled prompt only whole; see
	// prefixPool.exact. Said once at load, so a slower repeat is not a mystery.
	recurrent := llama.HasRecurrentState(model)
	if recurrent && prefix > 0 {
		log.Printf("prefix pool: whole prompts only — this model keeps recurrent state, which cannot be cut back to a shorter prefix")
	}

	home := layout{slots: slots, prefix: prefix, perSeq: e.nCtx, shared: shared, ubatch: ub}
	how := "sized the way Ollama sizes it"
	if !ubAuto {
		how = "as -batch asked"
	}
	log.Printf("batch: the GPU reads %d prompt tokens a pass, %s — the model and its cache are %s of %s",
		ub, how, gib(sub(bud.atStart, bud.afterCtx)), gib(bud.atStart))
	var fx *flex
	if !opts.FixedLayout {
		fx = newFlex(home, shape.TrainedCtx, contextBuilder(model, home, bud))
	}
	if fx != nil {
		log.Printf("slots: %d tokens of context as %s; a prompt that needs a longer slot gets %s while it runs "+
			"— the rest wait, and %s comes back when nothing needs the longer slots. -flex=false keeps %s.",
			fx.total, home, fx.offers(), home, home)
	}

	e.sched = newScheduler(lctx, home, vocab, tpl, batchCapacity, queue,
		deadlines{
			generate: pick(opts.MaxGenerate, DefaultMaxGenerate),
			wait:     pick(opts.MaxWait, DefaultMaxWait),
		}, recurrent, fx)
	e.sched.onFatal = func() { e.broken.Store(true) }
	return e, nil
}

// contextBuilder makes the contexts a changing layout needs, from the model
// already loaded and against the budget measured while loading it.
//
// A layout other than home is refused if it would leave less memory than home
// did (or than the warning line, if home left more), less reshapeSlack. It
// holds the same tokens, so it should cost about the same; if it does not, the
// place to find out is here, before any batch runs in it — a batch that cannot
// allocate retires the whole model (see thinHeadroom).
func contextBuilder(model llama.Model, home layout, bud *budget) func(layout) (llama.Context, uint64, error) {
	floor := sub(min(bud.afterCtx, thinHeadroom), reshapeSlack)
	return func(l layout) (llama.Context, uint64, error) {
		lctx := newContext(model, l.slots, l.prefix, l.perSeq, l.shared, l.ubatch)
		if lctx == 0 {
			return 0, 0, errors.New("llama.cpp could not create it (see its output above)")
		}
		if got, want := llama.NCtx(lctx), l.perSeq*l.cells(); got < want {
			llama.FreeContext(lctx)
			return 0, 0, fmt.Errorf("llama.cpp allocated %d tokens of the %d asked for", got, want)
		}
		free := bud.sample()
		if bud.known && l != home && free < floor {
			llama.FreeContext(lctx)
			return 0, 0, fmt.Errorf("it would leave %s free, where %s left %s", gib(free), home, gib(bud.afterCtx))
		}
		return lctx, free, nil
	}
}

// newContext asks llama.cpp for a cache holding slots+prefix sequences of
// perSeq tokens each — or, when the pool is shared, slots conversations' worth
// of tokens that the pool's sequences borrow from.
func newContext(model llama.Model, slots, prefix, perSeq int, shared bool, ubatch int) llama.Context {
	cp := llama.DefaultContextParams()
	// Pooled prefixes live in sequence ids above the slots, and each needs a
	// sequence's worth of cache unless it borrows the slots'.
	cp.NSeqMax = uint32(slots + prefix)
	// llama.cpp divides n_ctx between sequences, so ask for the total.
	cp.NCtx = uint32(perSeq * Plan{Slots: slots, Prefix: prefix, SharedPool: shared}.Cells())
	if cp.NBatch < uint32(batchCapacity) {
		cp.NBatch = uint32(batchCapacity)
	}
	if ubatch > 0 {
		cp.NUbatch = uint32(min(ubatch, batchCapacity))
	}
	if prefix > 0 {
		// A cell can only belong to several sequences in the unified buffer,
		// and sharing cells is the entire point of the pool. llama.cpp warns
		// that unified costs performance when sequences do NOT share a large
		// prefix — here they are chosen precisely because they do.
		cp.KVUnified = 1
	}
	return llama.NewContext(model, cp)
}

// sharedNote marks a sizing line whose pool borrows the slots' cells.
func sharedNote(shared bool) string {
	if shared {
		return " (the pool sharing the slots' cells)"
	}
	return ""
}

// Slots is how many conversations this engine serves at once — in the layout
// it has now, which a long prompt can change for as long as it runs.
func (e *Engine) Slots() int {
	e.mu.Lock()
	sched := e.sched
	e.mu.Unlock()
	if sched == nil {
		return e.slots
	}
	return int(sched.nSlots.Load())
}

// Broken reports that llama.cpp's backend failed fatally and this engine can no
// longer generate. The usual cause is the GPU running out of memory, after
// which every decode returns the same error until the context is recreated.
func (e *Engine) Broken() bool { return e.broken.Load() }

// Close stops the scheduler and releases the model and its context.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.sched != nil {
		// Stop decoding before anything it decodes into is freed. Once it has
		// stopped, the context it holds is no longer touched by anything.
		e.sched.close()
		if e.sched.lctx != 0 {
			llama.FreeContext(e.sched.lctx)
			e.sched.lctx = 0
		}
		e.sched = nil
	}
	if e.model != 0 {
		llama.FreeModel(e.model)
		e.model = 0
	}
}

// Path is the file this engine was loaded from.
func (e *Engine) Path() string { return e.path }

// Load is what the server is doing right now: requests waiting for a slot,
// slots generating, and slots in total. A deep queue beside idle slots is a
// different problem from slots that are always full, and telling them apart is
// the whole reason these are reported.
func (e *Engine) Load() (waiting, busy, slots int) {
	e.mu.Lock()
	sched := e.sched
	e.mu.Unlock()
	if sched == nil {
		return 0, 0, e.slots
	}
	return sched.Waiting(), int(sched.busy.Load()), int(sched.nSlots.Load())
}

// PromptTokens and EvalTokens are what this engine has processed since it
// loaded. Counters, not rates: a scraper takes the difference.
func (e *Engine) PromptTokens() int64 {
	return e.counter(func(s *scheduler) int64 { return s.promptSeen.Load() })
}
func (e *Engine) EvalTokens() int64 {
	return e.counter(func(s *scheduler) int64 { return s.evalSeen.Load() })
}

func (e *Engine) counter(read func(*scheduler) int64) int64 {
	e.mu.Lock()
	sched := e.sched
	e.mu.Unlock()
	if sched == nil {
		return 0
	}
	return read(sched)
}

// Template reports the chat family in use.
func (e *Engine) Template() string { return e.tpl.Name }

// ToolFormat is the tool-call convention this model was trained on, read from
// its own chat template. tools.None means it knows none, and will answer in
// prose no matter how the functions are declared.
func (e *Engine) ToolFormat() tools.Format { return e.tpl.ToolFormat() }

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
	text, _, err := e.ChatFull(ctx, msgs, p, onToken)
	return text, err
}

// ChatFull is Chat plus what was counted and measured while producing it: the
// token counts and timings the HTTP dialects report, and whether the reply was
// cut short by the token budget.
//
// That last distinction matters most for a reasoning model: it can spend an
// entire budget thinking and return an empty answer, and a caller told that
// completed normally has no way to tell that from a model with nothing to say.
func (e *Engine) ChatFull(ctx context.Context, msgs []chat.Message, p GenParams, onToken func(string)) (string, Stats, error) {
	e.mu.Lock()
	sched := e.sched
	e.mu.Unlock()

	if sched == nil {
		return "", Stats{}, fmt.Errorf("engine for %s is closed", e.path)
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
		return "", Stats{}, err
	}

	var out strings.Builder
	for frag := range j.frags {
		out.WriteString(frag)
		if onToken != nil {
			onToken(frag)
		}
	}
	<-j.done

	// A deadline is not a failure at this boundary. The partial reply is worth
	// more to a caller than an error with nothing attached, so it travels as a
	// finish reason — see Stats.Timeout — the same way a token budget does.
	err := j.err
	var late *TimeoutError
	if errors.As(err, &late) && !late.BeforeStarting() {
		j.stats.Timeout, err = true, nil
	}
	return out.String(), j.stats, err
}

// pick resolves a duration option: zero takes the default, negative means the
// operator turned the limit off and is respected as such.
func pick(v, def time.Duration) time.Duration {
	if v == 0 {
		return def
	}
	if v < 0 {
		return 0
	}
	return v
}
