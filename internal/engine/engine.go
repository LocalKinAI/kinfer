// Package engine runs one loaded model.
//
// It owns the pieces that must agree with each other — the llama.cpp context,
// the chat template, the sampler, the vocabulary — and exposes a single Chat
// call. Both the CLI and the HTTP server go through it, so there is one place
// where generation is correct rather than two places that drift.
package engine

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

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
	// Pick from {8, 16, 64, 128}: past a batch of 16 the step time more than
	// doubles, and the batch does not pay for it until 48 to 64, so 24 or 32
	// returns less than 16 (llama.cpp b11998, on an M3 Ultra and an M5 Max).
	// See the table in docs/performance.md. 0 means 8, or fewer when the
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
	// (llama.cpp's n_ubatch), up to maxBatch. 0 sizes it the way Ollama does:
	// 2048, 1024 or 512, by the context and by how much of the memory the
	// model and its cache take (see autoUbatch); 2048 with HostExperts, which
	// every pass reads again, unless the context leaves too little memory.
	Batch int

	// Peers are other machines' kinfer workers ("host:port"), whose GPUs
	// take some of the model's layers; see llama.RPCDevices. They come first
	// in the device list, as llama.cpp orders them, and this machine's GPU
	// last.
	Peers []string

	// FindPeers, in place of Peers, looks for workers over the Thunderbolt
	// cable for a model this GPU cannot hold, and borrows those that answer;
	// a model that fits borrows none (findPeers).
	FindPeers bool

	// Split is each device's share of the layers, in that order: the peers',
	// then this machine's. Empty lets llama.cpp share by free memory.
	Split []float32

	// HostExperts leaves the routed experts of a mixture of experts' first
	// HostExperts layers in host memory, mapped from the file, instead of on
	// the GPU; negative means every layer. See experts.go.
	HostExperts int

	// MoeCache is MiB of GPU memory caching the host experts in use, so only
	// a miss is copied; 0 is none. It applies to layers whose experts are all
	// in host memory.
	MoeCache int

	// NoPlace loads every weight on the GPU even when it cannot hold them, and
	// leaves the split across Peers to llama.cpp. Without it, given neither
	// HostExperts nor MoeCache, a mixture of experts the GPU cannot hold leaves
	// as many layers' experts in host memory as it must, with a GPU cache for
	// the ones in use (Place); and with Peers but no Split, the peers take only
	// the layers that do not fit here (PlaceDevices).
	NoPlace bool

	// CacheType is what the attention cache is kept in: "q8_0" where the
	// model allows it (see CacheQ8), or f16 — the default.
	CacheType string

	// Projector is a vision model's projector (store.Store.ProjectorFor),
	// loaded with it so that the model reads pictures; "" for text only.
	Projector string

	// Draft is how many tokens a step drafts with the model's MTP head, for
	// one pass to check (see draft.go). 0 drafts four with a dense model that
	// has a head, one with a mixture of experts, and none when its experts are
	// in host memory; negative turns it off.
	Draft int
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

	// Think is thinking the caller asked for in so many words (think:true, a
	// reasoning effort, Anthropic's thinking enabled). With neither it nor
	// NoThink the model thinks as its template does by default, as in Ollama;
	// see chat.Thinking.
	Think bool

	// Effort is how hard a reasoning model that takes a budget should think:
	// "low", "high", "max", or a number from 1 to 100. Empty is the model's
	// default. Templates without such a budget ignore it.
	Effort string

	// Resume is the start of the reply, which the caller already has: the
	// model goes on from its end, and only what follows is sent. The server
	// sets it to carry a reply over to a reloaded model when the backend went
	// down in the middle of it (ErrInterrupted).
	Resume string
}

// thinking is the reasoning a job asked for, in the form templates take.
func (p GenParams) thinking() chat.Thinking {
	return chat.Thinking{Off: p.NoThink, On: p.Think, Effort: p.Effort}
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
	// for every slot, with room to spare. A micro-batch larger than it makes
	// the pass that large instead (see batchFor).
	batchCapacity = 2048

	// maxBatch is the largest micro-batch -batch may ask for.
	maxBatch = 8192

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
	most  int // the longest one conversation can be: flex's longest slot, or nCtx

	// what this model counts toward keeping the GPU warm (warm.go), each 0 or 1
	warmSplit, warmCache int

	// sched owns the llama.cpp context: it rebuilds it when the layout
	// changes, so the context it holds when it stops is the one to free.
	sched *scheduler

	// mtmd is the projector a vision model reads pictures through; freed
	// after the scheduler stops, which is what uses it.
	mtmd llama.Mtmd

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
	shape, _ := store.ReadShape(path)
	shape.CacheQ8 = CacheQ8(shape, opts.CacheType)
	var why string
	if shape.Draft, why = Drafts(shape, opts.Draft); why != "" {
		log.Printf("draft: none for %s — %s", filepath.Base(path), why)
	}
	switch {
	case shape.CacheQ8:
		log.Printf("cache: K and V at q8_0, about half what f16 takes")
	case opts.CacheType == "q8_0":
		log.Printf("cache: K and V at f16 — q8_0 does not suit %s (heads of %d)", shape.Arch, shape.HeadDim())
	}

	peers := opts.Peers
	if opts.FindPeers && opts.HostExperts == 0 && opts.MoeCache == 0 && !opts.NoPlace && len(opts.Split) == 0 {
		peers = findPeers(shape, bud, opts, filepath.Base(path))
	}
	devs, split, err := deviceList(peers, opts.Split)
	if err != nil && opts.FindPeers && len(peers) > 0 {
		// Found is not promised: a worker that answered may be lent to
		// another machine by now. The model stays here instead.
		log.Printf("peers: %v; %s stays here", err, filepath.Base(path))
		peers = nil
		devs, split, err = deviceList(nil, opts.Split)
	}
	if err != nil {
		return nil, err
	}
	if shape.Draft > 0 && len(peers) > 0 {
		log.Printf("draft: not with the model split across machines — %s runs without drafts", filepath.Base(path))
		shape.Draft = 0
	}

	// Where the weights go, when the operator did not say: a mixture of
	// experts the GPU cannot hold leaves some layers' experts in host memory,
	// and with peers, a model goes to them only as far as it does not fit here.
	hostExperts, moeCache, firstHost := opts.HostExperts, opts.MoeCache, 0
	var placed Placed
	if hostExperts == 0 && moeCache == 0 && !opts.NoPlace && bud.known {
		switch {
		case len(peers) == 0:
			placed = placeFor(shape, bud.atStart, opts, filepath.Base(path))
		case split == nil && opts.GPULayers > shape.Layers:
			var across DevicePlacement
			across, placed = placeAcross(shape, devs, bud.atStart, opts, filepath.Base(path))
			if across.Layers != nil {
				split, firstHost = across.Split(shape.Layers), across.FirstHost
			}
		}
		hostExperts, moeCache = placed.HostExperts, placed.MoeCache
	}
	if shape.Draft > 0 && hostExperts != 0 && opts.Draft == 0 {
		// A draft's experts are read like any other token's, and these come
		// from host memory — the SSD, when it misses — so the step it saves
		// costs more than it did.
		log.Printf("draft: none for %s by default — its experts are read from host memory; -draft %d asks for it", filepath.Base(path), moeDraft)
		shape.Draft = 0
	}
	if shape.Draft > 0 {
		// The head's weights stay on disk unless asked for.
		mp.LoadMtp = 1
	}
	if devs != nil {
		mp.Devices = uintptr(unsafe.Pointer(&devs[0]))
		if split != nil {
			mp.TensorSplit = uintptr(unsafe.Pointer(&split[0]))
		}
	}

	overrides, patterns, err := hostExpertOverrides(firstHost, hostExperts)
	if err != nil {
		return nil, err
	}
	if overrides != nil {
		mp.TensorBuftOverrides = uintptr(unsafe.Pointer(&overrides[0]))
		// Left to itself llama.cpp repacks some host weights for faster CPU
		// arithmetic (CPU_REPACK), and a repacked buffer is not plain host
		// memory, so the MoE cache passes its layer over: on Flash-Next IQ3_S
		// every layer with an IQ4_NL down projection, 39 of 48. llama.cpp's
		// --no-repack, the same switch.
		mp.UseExtraBufts = 0
		which := fmt.Sprintf("the first %d layers'", hostExperts)
		switch {
		case hostExperts < 0:
			which = "every layer's"
		case firstHost > 0:
			which = fmt.Sprintf("layers %d-%d's", firstHost, firstHost+hostExperts-1)
		}
		log.Printf("experts: %s routed experts stay in host memory, mapped from the file; a %d MiB GPU cache keeps the ones in use", which, moeCache)
		// Placement always gives host experts a cache; only -host-experts on
		// its own leaves none, and DeepSeek-V4.1 then decoded at 3.6 tok/s on
		// the box where a 46 GiB cache made it 18.
		if moeCache == 0 {
			log.Printf("experts: no GPU cache was asked for, so every token reads those layers' experts from host memory; " +
				"-moe-cache MiB caches the ones in use, and with neither -host-experts nor -moe-cache kinfer places the model itself")
		}
	}

	model := llama.LoadModel(path, mp)
	// llama.cpp has copied both lists into the model by now; until then they
	// must not be collected.
	runtime.KeepAlive(devs)
	runtime.KeepAlive(split)
	runtime.KeepAlive(overrides)
	runtime.KeepAlive(patterns)
	if model == 0 {
		return nil, fmt.Errorf("load %s: llama.cpp could not read the model (see its output above)", path)
	}
	// A vision model reads pictures through its projector, loaded — and its
	// encoder warmed up — before memory is measured, so the context is sized
	// around it.
	var mtmd llama.Mtmd
	if opts.Projector != "" {
		if mtmd = llama.LoadProjector(opts.Projector, model, 0); mtmd == 0 {
			log.Printf("pictures: %s could not be loaded; the model reads text only", opts.Projector)
		} else {
			log.Printf("pictures: read through %s", filepath.Base(opts.Projector))
		}
	}
	bud.afterModel = bud.sample()

	// Whatever the operator left at zero is sized here, from the model's
	// shape and the memory the weights actually left — see autosize.go.
	sized := opts.Slots <= 0 || opts.ContextSize <= 0
	// llama.cpp throws on a MoE cache for a model with no experts, which would
	// fail every context; a server serving several models is told it once.
	if moeCache > 0 && shape.Experts == 0 {
		log.Printf("experts: %s has none, so -moe-cache does not apply to it", filepath.Base(path))
		moeCache = 0
	}
	// The MoE cache is allocated with each context, so what the context may
	// cost is what the weights left less the cache.
	left := sub(bud.afterModel, uint64(moeCache)<<20)

	// A placed model's GPU holds the room placement kept and little more: the
	// context is sized to what that room was kept for, not to half of what is
	// left, which would leave the pass to be the first thing the measurement
	// below takes back.
	ctx, slotsWant := opts.ContextSize, opts.Slots
	if placed.Reserve != 0 {
		ctx = cmp.Or(ctx, placed.Tokens)
		slotsWant = placed.Slots
	}

	// Slots given and the context left to us, with flex on: the total the
	// slots share is sized per model the way Ollama sizes one request's
	// context (SizeTotal), and flex gives one long prompt all of it.
	byTotal := opts.Slots > 0 && ctx <= 0 && !opts.FixedLayout
	if byTotal {
		t, err := SizeTotal(shape, bud.dev.Total, left, opts.Slots)
		if err != nil {
			llama.FreeProjector(mtmd)
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
			t.PerSeq*t.Slots, t.Slots, t.PerSeq, one, gib(uint64(t.Estimated)), gib(left))
	}

	plan, err := AutoSize(shape, left, slotsWant, opts.PrefixSlots, ctx)
	if err != nil {
		llama.FreeProjector(mtmd)
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
	ub := min(max(opts.Batch, 1), maxBatch)
	if ubAuto {
		ub = autoUbatch(longest, sub(bud.atStart, left)+uint64(plan.Estimated), bud.atStart)
		// Experts in host memory are read again by every pass of a prompt,
		// and a pass of 2048 reads them a quarter as often as one of 512: on
		// the MacBook Pro, Flash-Next with 24 layers' experts on the host read
		// a 7,753-token prompt at 73 tok/s in passes of 512 and 162 in passes
		// of 2048 (kinfer's patched llama.cpp makes it 502); see ubatchHost
		// for 4096. The context measured below takes it back down when it
		// costs too much.
		if hostExperts != 0 {
			ub = max(ub, ubatchHost)
		}
		// Placement kept room for a pass of its own size.
		if placed.Reserve != 0 {
			ub = max(ub, placed.Ubatch)
		}
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
			slots, prefix, perSeq, sharedNote(shared), trained, gib(uint64(plan.Estimated)), gib(left))
	} else if prefixAuto {
		log.Printf("sizing: %d prefix entries%s around the %d slots x %d tokens asked for — "+
			"the pool gets only what they leave; -prefix overrides", prefix, sharedNote(shared), slots, perSeq)
	}

	floor := ctxFloor
	if shape.TrainedCtx > 0 {
		floor = min(floor, shape.TrainedCtx)
	}
	// The expert cache placement chose may give up as much as half of itself
	// to keep the prefix pool; a cache the operator sized is theirs.
	cacheAuto := hostExperts != 0 && opts.MoeCache == 0 && moeCache > 0
	cacheFloor := moeCache / 2

	var lctx llama.Context
	var draft *drafter
sizing:
	for {
		lctx, draft = openLayout(model, layout{slots: slots, prefix: prefix, perSeq: perSeq, shared: shared, ubatch: ub},
			moeCache, shape.CacheQ8, shape.Draft)
		if lctx == 0 {
			llama.FreeProjector(mtmd)
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
		cut := cacheCut(enough, bud.afterCtx)
		switch {
		// A bigger micro-batch is speed, not capacity: it goes before
		// anything a conversation would notice — unless experts are in host
		// memory, where every pass reads them again and it goes last.
		case ubAuto && ub > ubatchDefault && hostExperts == 0:
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
		// Placement sized the expert cache before this measurement saw what a
		// pool costs: a pool needs the KV cache unified, and a pass over all
		// of it — on the box, Flash-Next's 4 x 32768 took a pass of 5.8 GiB
		// where the conversations' own caches took 2.2. The cache gives that
		// up before the pool goes: a smaller cache costs some decode speed,
		// no pool reads every turn whole again (11.8 s for 7,800 tokens there).
		case (perSeq/2 < floor || !ctxAuto) && prefixAuto && prefix > 0 && cacheAuto && moeCache-cut >= cacheFloor:
			moeCache -= cut
			log.Printf("%s — the expert cache gives up %d MiB instead, keeping %d prefix with %d MiB of cache", measured, cut, prefix, moeCache)
		case (perSeq/2 < floor || !ctxAuto) && prefixAuto && prefix > 0:
			prefix--
			log.Printf("%s — %d prefix instead, keeping %d tokens each", measured, prefix, perSeq)
		case ctxAuto && perSeq/2 >= 2048:
			perSeq /= 2
			log.Printf("%s — halving the context", measured)
		case ubAuto && ub > ubatchDefault:
			ub = lowerUbatch(ub)
			log.Printf("%s — a micro-batch of %d instead", measured, ub)
		default:
			break sizing
		}
		draft.free()
		llama.FreeContext(lctx)
	}

	// The micro-batch was chosen before the context existed, from what it was
	// predicted to cost, and that rule is Ollama's and cautious: on the box,
	// Flash-Next's model and cache came to 71.9 GiB of the 77.7 free, so it got
	// passes of 512, and the context then left 5.8 GiB. Where the measurement
	// leaves twice the headroom, a pass twice as large is tried, and kept if
	// it still leaves the headroom; on ornith-1.5:35b an 11,000-token prompt
	// read 13–18% faster in passes of 2048 than of 512. Not with experts on
	// the host, whose pass is sized for them already, nor across machines,
	// where the larger pass also grows the peers' buffers this one cannot see.
	grownFrom := 0
	for ubAuto && hostExperts == 0 && devs == nil && ub < ubatchLarge && bud.known && bud.afterCtx >= 2*thinHeadroom {
		try := min(ub*2, ubatchLarge)
		draft.free()
		llama.FreeContext(lctx)
		next, nextDraft := openLayout(model, layout{slots: slots, prefix: prefix, perSeq: perSeq, shared: shared, ubatch: try},
			moeCache, shape.CacheQ8, shape.Draft)
		left := bud.sample()
		if next != 0 && left >= thinHeadroom {
			grownFrom = cmp.Or(grownFrom, ub)
			lctx, draft, ub, bud.afterCtx = next, nextDraft, try, left
			continue
		}
		if next != 0 {
			nextDraft.free()
			llama.FreeContext(next)
		}
		if lctx, draft = openLayout(model, layout{slots: slots, prefix: prefix, perSeq: perSeq, shared: shared, ubatch: ub},
			moeCache, shape.CacheQ8, shape.Draft); lctx == 0 {
			llama.FreeProjector(mtmd)
			llama.FreeModel(model)
			return nil, fmt.Errorf("create context for %s", path)
		}
		bud.afterCtx = bud.sample()
		break
	}
	bud.report(slots, prefix, perSeq, shared)

	// With next to nothing left the first pass may not allocate, and a reload
	// with the same settings fails the same way: say so now, and again in the
	// error if it happens, with the settings that were the operator's to give
	// back (see noRoom).
	var roomHint string
	if bud.known && bud.afterCtx < noRoom {
		roomHint = noRoomHint(filepath.Base(path), bud.afterCtx, opts)
		log.Printf("memory: %s", roomHint)
	}
	if ctxAuto && perSeq < floor && (opts.HostExperts != 0 || opts.MoeCache > 0) {
		log.Printf("context: %d tokens a conversation is all that fits beside -host-experts/-moe-cache as given, "+
			"and a longer prompt is refused; a smaller -moe-cache leaves more, and with neither flag kinfer places the model itself", perSeq)
	}

	tpl := chat.FromModel(model, path)
	if opts.Template != "" {
		t, err := chat.Get(opts.Template)
		if err != nil {
			draft.free()
			llama.FreeContext(lctx)
			llama.FreeProjector(mtmd)
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
		draft.free()
		llama.FreeContext(lctx)
		llama.FreeProjector(mtmd)
		llama.FreeModel(model)
		return nil, fmt.Errorf("asked for a %d-token context (%d slots + %d prefix slots x %d) "+
			"but llama.cpp allocated %d "+
			"(llama_context_params may have changed shape — see internal/llama)",
			wantCtx, slots, prefix, perSeq, actualCtx)
	}

	vocab := llama.GetVocab(model)
	e := &Engine{
		model: model,
		mtmd:  mtmd,
		vocab: vocab,
		tpl:   tpl,
		path:  path,
		nCtx:  actualCtx / cells,
		slots: slots,
	}
	if hostExperts != 0 {
		e.warmCache = 1
	} else if devs != nil {
		e.warmSplit = 1
	}
	warmAdd(e.warmSplit, e.warmCache)
	if devs != nil && hostExperts != 0 {
		log.Printf("experts: with experts in host memory the GPU is not kept warm between tokens — the MoE cache waits on it too often")
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
	switch {
	case !ubAuto:
		how = "as -batch asked"
	case hostExperts != 0 && ub < ubatchHost:
		// The headroom is not this process's to spend: on a shared machine
		// it is what other processes' memory crosses the line with.
		how = fmt.Sprintf("not %d: the host experts are read again by every pass, and %d would leave less than %s free "+
			"— a smaller -moe-cache makes room, or -batch %d takes it from that", ubatchHost, ubatchHost, gib(thinHeadroom), ubatchHost)
	case hostExperts != 0:
		how = "sized for experts in host memory, which every pass reads again"
	case grownFrom != 0:
		how = fmt.Sprintf("grown from the %d Ollama's rule gives, since the context left room", grownFrom)
	case placed.Reserve != 0:
		how = "the pass placement kept room for"
	}
	log.Printf("batch: the GPU reads %d prompt tokens a pass, %s — the model and its cache are %s of %s",
		ub, how, gib(sub(bud.atStart, bud.afterCtx)), gib(bud.atStart))
	var fx *flex
	if !opts.FixedLayout {
		fx = newFlex(home, shape.TrainedCtx, contextBuilder(model, home, bud, moeCache, shape.CacheQ8, shape.Draft))
	}
	e.most = e.nCtx
	if fx != nil {
		e.most = fx.most
		log.Printf("slots: %d tokens of context as %s; a prompt that needs a longer slot gets %s while it runs "+
			"— the rest wait, and %s comes back when nothing needs the longer slots. -flex=false keeps %s.",
			fx.total, home, fx.offers(), home, home)
	}

	profile := newExpertProfile(path, moeCache)
	profile.load(lctx)

	if draft != nil {
		log.Printf("draft: up to %d tokens a step with the model's MTP head, for one pass to check — -draft -1 turns it off", shape.Draft)
	}
	e.sched = newScheduler(lctx, draft, home, vocab, tpl, batchFor(ub), queue,
		deadlines{
			generate: pick(opts.MaxGenerate, DefaultMaxGenerate),
			wait:     pick(opts.MaxWait, DefaultMaxWait),
		}, recurrent, fx)
	e.sched.profile = profile
	e.sched.mtmd = mtmd
	e.sched.onFatal = func() { e.broken.Store(true) }
	e.sched.roomHint = roomHint
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
func contextBuilder(model llama.Model, home layout, bud *budget, moeCache int, q8 bool, drafts int) func(layout) (llama.Context, *drafter, uint64, error) {
	floor := sub(min(bud.afterCtx, thinHeadroom), reshapeSlack)
	return func(l layout) (llama.Context, *drafter, uint64, error) {
		lctx, draft := openLayout(model, l, moeCache, q8, drafts)
		if lctx == 0 {
			return 0, nil, 0, errors.New("llama.cpp could not create it (see its output above)")
		}
		if got, want := llama.NCtx(lctx), l.perSeq*l.cells(); got < want {
			draft.free()
			llama.FreeContext(lctx)
			return 0, nil, 0, fmt.Errorf("llama.cpp allocated %d tokens of the %d asked for", got, want)
		}
		free := bud.sample()
		if bud.known && l != home && free < floor {
			draft.free()
			llama.FreeContext(lctx)
			return 0, nil, 0, fmt.Errorf("it would leave %s free, where %s left %s", gib(free), home, gib(bud.afterCtx))
		}
		return lctx, draft, free, nil
	}
}

// openLayout makes the contexts a layout runs on: the model's, and with
// drafts the MTP head's beside it (draft.go). A head that cannot be made is
// reported and left out; the model's context serves without it.
func openLayout(model llama.Model, l layout, moeCache int, q8 bool, drafts int) (llama.Context, *drafter) {
	lctx := newContext(model, l.slots, l.prefix, l.perSeq, l.shared, l.ubatch, moeCache, q8, drafts)
	if lctx == 0 || drafts == 0 {
		return lctx, nil
	}
	dctx := draftContext(model, l, q8)
	if dctx == 0 {
		log.Printf("draft: llama.cpp could not make the MTP head's context (see its output above) — no drafts")
		return lctx, nil
	}
	draft, err := newDrafter(lctx, dctx, model, l.slots+l.prefix, drafts)
	if err != nil {
		llama.FreeContext(dctx)
		log.Printf("draft: %v — no drafts", err)
		return lctx, nil
	}
	return lctx, draft
}

// draftContext makes the MTP head's context for a layout of the model's: the
// same sequences, as long, read in passes as large.
func draftContext(model llama.Model, l layout, q8 bool) llama.Context {
	cp := contextParams(l.slots, l.prefix, l.perSeq, l.shared, l.ubatch, 0, q8)
	cp.CtxType = llama.ContextTypeMTP
	// A pass asks for one row of logits per sequence drafted for, and a
	// context keeps room for as many as it may be asked for: by default a
	// whole micro-batch's, n_vocab floats each.
	cp.NOutputsMax = cp.NSeqMax
	return llama.NewContext(model, cp)
}

// batchFor is how many tokens one forward pass may hold with a micro-batch of
// ubatch: batchCapacity, or a whole micro-batch when that is larger, so that a
// prompt is read a micro-batch at a time.
func batchFor(ubatch int) int {
	return max(batchCapacity, ubatch)
}

// newContext asks llama.cpp for a cache holding slots+prefix sequences of
// perSeq tokens each — or, when the pool is shared, slots conversations' worth
// of tokens that the pool's sequences borrow from — kept at q8_0 when q8. With
// drafts, each sequence of a hybrid model keeps that many snapshots of its
// recurrent state, to go back to when drafts are rejected.
func newContext(model llama.Model, slots, prefix, perSeq int, shared bool, ubatch, moeCache int, q8 bool, drafts int) llama.Context {
	cp := contextParams(slots, prefix, perSeq, shared, ubatch, moeCache, q8)
	if drafts > 0 && llama.HasRecurrentState(model) {
		cp.NRsSeq = uint32(drafts)
	}
	return llama.NewContext(model, cp)
}

// contextParams lays out a context as newContext describes.
func contextParams(slots, prefix, perSeq int, shared bool, ubatch, moeCache int, q8 bool) llama.ContextParams {
	cp := llama.DefaultContextParams()
	cp.MoeCacheSize = uint64(moeCache) << 20
	if q8 {
		// llama.cpp turns flash attention on for it, which a quantized V
		// cache needs and Metal has for every head size CacheQ8 lets by.
		cp.TypeK, cp.TypeV = llama.TypeQ8_0, llama.TypeQ8_0
	}
	// Pooled prefixes live in sequence ids above the slots, and each needs a
	// sequence's worth of cache unless it borrows the slots'.
	cp.NSeqMax = uint32(slots + prefix)
	// llama.cpp divides n_ctx between sequences, so ask for the total.
	cp.NCtx = uint32(perSeq * Plan{Slots: slots, Prefix: prefix, SharedPool: shared}.Cells())
	if cp.NBatch < uint32(batchFor(ubatch)) {
		cp.NBatch = uint32(batchFor(ubatch))
	}
	if ubatch > 0 {
		cp.NUbatch = uint32(ubatch)
	}
	if prefix > 0 {
		// A cell can only belong to several sequences in the unified buffer,
		// and sharing cells is the entire point of the pool. llama.cpp warns
		// that unified costs performance when sequences do NOT share a large
		// prefix — here they are chosen precisely because they do.
		cp.KVUnified = 1
	}
	return cp
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

// CountTokens is how many tokens msgs come to as a prompt, rendered with the
// tools and thinking p asks for: what Anthropic's count_tokens answers, and
// what an agent sizes its history by. A picture counts as the cells the
// projector lays it out in, as it does when the prompt is read. Tokenizing
// with libmtmd prepares the pictures on the CPU only; the encoder, which the
// scheduler alone runs, is not touched.
func (e *Engine) CountTokens(msgs []chat.Message, p GenParams) (int, error) {
	text := e.tpl.RenderWith(msgs, p.Tools, p.thinking())
	if !showsPictures(msgs) {
		toks, err := llama.Tokenize(e.vocab, text, true, true)
		return len(toks), err
	}
	if e.mtmd == 0 {
		return 0, PictureError{ErrNoPictures}
	}
	var pictures [][]byte
	for _, m := range msgs {
		pictures = append(pictures, m.Images...)
	}
	mp, err := llama.TokenizeMedia(e.mtmd, text, pictures)
	if err != nil {
		return 0, PictureError{err}
	}
	defer mp.Free()
	return mp.Tokens, nil
}

// Context is the context each conversation has in the layout the model loaded
// with, and the most one conversation can have: flex gives a longer prompt a
// longer slot while it runs, and the others wait. An agent compacting its
// history should know the second, or it is cut off where it thought it had room.
func (e *Engine) Context() (perSlot, most int) { return e.nCtx, e.most }

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
		if d := e.sched.drafted.Load(); d > 0 {
			a := e.sched.draftsAccepted.Load()
			log.Printf("draft: the model accepted %d of the %d tokens its MTP head drafted for %s (%.0f%%)",
				a, d, filepath.Base(e.path), 100*float64(a)/float64(d))
		}
		if e.sched.lctx != 0 {
			e.sched.profile.save(e.sched.lctx)
			e.sched.draft.free()
			e.sched.draft = nil
			llama.FreeContext(e.sched.lctx)
			e.sched.lctx = 0
		}
		e.sched = nil
	}
	llama.FreeProjector(e.mtmd)
	e.mtmd = 0
	if e.model != 0 {
		llama.FreeModel(e.model)
		e.model = 0
	}
	if e.warmSplit != 0 || e.warmCache != 0 {
		warmAdd(-e.warmSplit, -e.warmCache)
		e.warmSplit, e.warmCache = 0, 0
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

// Drafts is how many tokens the model's MTP head drafted since it loaded, and
// how many of them the model accepted; counters too.
func (e *Engine) Drafts() (drafted, accepted int64) {
	return e.counter(func(s *scheduler) int64 { return s.drafted.Load() }),
		e.counter(func(s *scheduler) int64 { return s.draftsAccepted.Load() })
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
