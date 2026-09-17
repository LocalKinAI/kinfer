package engine

import (
	"fmt"
	"log"

	"github.com/LocalKinAI/kinfer/internal/llama"
)

// Memory budgeting for the accelerator.
//
// One thing to know before reading any number here: ggml-metal computes free as
// recommendedMaxWorkingSetSize minus currentAllocatedSize, and
// currentAllocatedSize is a property of this process's MTLDevice. So the
// headroom below is what THIS process has left of its budget, not what the
// machine has left. Verified: with a 73.4 GiB model resident in another kinfer,
// a second one loading a 0.5B still reported 76.9 GiB of 77.8 GiB free.
//
// That is the uncomfortable part. Every out-of-memory failure observed on this
// machine happened while another process held GPU memory, and this reading is
// blind to exactly that.
//
// The failure this exists to prevent: kinfer accepts `-ctx 16384 -slots 4`,
// spends 34 seconds loading 73.5 GiB, answers a few requests, and then dies
// under load with
//
//	error: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)
//	llama_decode: failed to decode, ret = -3
//
// which retires the model and fails every in-flight and queued request at once.
// Nothing in that sequence tells the operator what was wrong, and the obvious
// reading — "too many concurrent requests" — is wrong: only `slots` sequences
// ever decode at a time. The real cause is that `-ctx` is per-conversation and
// gets multiplied by the slot count, so raising slots from 2 to 4 silently
// doubled the KV cache on a machine that had no room for it.
//
// A promise that fails later is the thing kinfer exists to not do. So measure
// the budget while the model is loading, and refuse before serving.

// thinHeadroom is the point below which kinfer says out loud how little room is
// left for anything else on the machine.
//
// The failure it warns about is understood, and the account is the one thing on
// this machine worth writing down. ggml-metal derives free from
// recommendedMaxWorkingSetSize minus currentAllocatedSize, and
// currentAllocatedSize belongs to the calling process. Every process is
// therefore told how much of the budget IT has spent, and nothing about the
// others — while the hardware limit they are all spending against is shared.
// When the sum crosses, an allocation fails, and no process could see it coming.
//
// Measured, with each process logging its own figure and the sums added
// afterwards, against a 77.8 GiB budget and a 73.4 GiB model:
//
//	processes                 sum       margin   outcome
//	alone                     76.2 GiB  -1.6     passes
//	+ one 0.5B under load     77.3 GiB  -0.5     32/32 answered, three runs
//	+ two 0.5B under load     78.4 GiB  +0.7     fails, both runs
//	+ a second copy of itself 151.3 GiB +73.5    fails immediately
//
// The only difference between the third row and the second is 1.1 GiB belonging
// to someone else, and the sum crosses exactly there. It explains what weeks of
// looking inside one process could not: why an exclusive machine never
// reproduces the failure at any concurrency, prompt length or batch size; why a
// second big model reproduces it instantly; and why the original outage looked
// intermittent — it sat within a gigabyte of the line.
//
// Two consequences for this file. The figure it prints is exact for this
// process and says nothing about the machine, which is why it is labelled that
// way. And the line below is not a prediction that this configuration will
// fail: it is the point at which there is no longer room for company, which on
// a shared machine is the thing that decides it.
const thinHeadroom = 2 << 30 // 2 GiB

// ThinHeadroom is the same number, for tools that plan a configuration the
// server will then judge. They must use one number or fit will recommend
// something serve immediately complains about.
func ThinHeadroom() int64 { return thinHeadroom }

// budget tracks the accelerator's free memory across the stages of a load.
type budget struct {
	dev   llama.Device
	known bool

	atStart    uint64
	afterModel uint64
	afterCtx   uint64
}

func newBudget() *budget {
	b := &budget{}
	if d, ok := llama.Accelerator(); ok {
		b.dev, b.known, b.atStart = d, true, d.Free
	}
	return b
}

// sample re-reads free memory. Returns 0 when there is no accelerator.
func (b *budget) sample() uint64 {
	if !b.known {
		return 0
	}
	d, ok := llama.Accelerator()
	if !ok {
		return 0
	}
	b.dev = d
	return d.Free
}

// report logs what the load actually cost and, when the margin is thin, what
// that risks. It never refuses: see thinHeadroom for why measuring beats
// enforcing here.
//
// slots, perSeq and prefix are echoed because the arithmetic between them is
// the part operators get wrong — `-ctx` reads like a total and is not one.
func (b *budget) report(slots, prefix, perSeq int, shared bool) {
	if !b.known {
		return
	}

	model := sub(b.atStart, b.afterModel)
	ctx := sub(b.afterModel, b.afterCtx)
	free := b.afterCtx
	total := b.dev.Total

	// The multiplication is spelled out because it is the part that surprises
	// people: -ctx reads like a total and is per-conversation.
	cells := Plan{Slots: slots, Prefix: prefix, SharedPool: shared}.Cells()
	layout := fmt.Sprintf("x %d sequences (%d slots + %d prefix)", cells, slots, prefix)
	if shared {
		layout = fmt.Sprintf("x %d slots, with %d prefix entries in whatever the slots leave free", slots, prefix)
	}
	log.Printf("memory: %s   model %s + context %s -> %s free of %s (this process only)\n"+
		"          context is %d tokens: -ctx %d per conversation %s",
		b.dev.Name, gib(model), gib(ctx), gib(free), gib(total),
		perSeq*cells, perSeq, layout)

	if total == 0 || free >= thinHeadroom {
		return
	}

	log.Printf("memory: only %s of %s is left after loading.\n"+
		"          This figure is this process's own budget and cannot see other\n"+
		"          processes — anything else using the GPU eats into it unseen.\n"+
		"          A batch that then cannot allocate is not recoverable in place: the\n"+
		"          Metal backend latches the error, llama.cpp reports it fatally, and\n"+
		"          kinfer retires the model, ending every in-flight and queued request\n"+
		"          at once. Configurations this tight have run for hours and have also\n"+
		"          died, so leave room for whatever else runs here.\n"+
		"          -ctx %d -slots %d, or -ctx %d -slots %d, would leave more.",
		gib(free), gib(total),
		suggestCtx(b, cells, 0, perSeq), slots,
		perSeq, max(1, slots/2))
}

// freeAt estimates the headroom a different per-conversation context would
// leave, by scaling the cache cost that was just measured.
//
// It is an upper bound, not a promise. Measured on the hybrid model above: it
// predicted 4.0 GiB for -ctx 2048 and the real figure was 3.1, because a
// per-sequence recurrent state is counted in the cache and does not shrink with
// the context. That is why the warning no longer quotes a number back.
func freeAt(b *budget, newPerSeq, slots, prefix, perSeq int) uint64 {
	ctx := sub(b.afterModel, b.afterCtx)
	if perSeq <= 0 {
		return b.afterCtx
	}
	scaled := uint64(float64(ctx) * float64(newPerSeq) / float64(perSeq))
	return sub(b.afterModel, scaled)
}

// suggestCtx is the largest per-conversation context that would have cleared
// the floor, rounded down to a power of two so the answer is one an operator
// would have picked themselves.
//
// It assumes the KV cache scales linearly with the token count. That holds for
// ordinary attention and overstates the saving for hybrid architectures, whose
// cache includes a per-sequence recurrent state of fixed size — so the context
// it suggests frees less than the arithmetic implies, and the warning says so
// rather than quoting a figure.
//
// How far out that assumption goes, measured on qwen3.8-flash-next at the same
// 32768 total tokens:
//
//	 2 sequences x 16384   1.7 GiB
//	32 sequences x  1024   4.3 GiB
//
// Identical token counts, two and a half times the memory, entirely from the
// number of sequences. Scaling a cache cost by tokens alone underestimated the
// second by 2.6 GiB and left a 73.4 GiB model with no headroom at all — twice,
// on two different days, by someone who had already written this paragraph. The
// per-sequence term dominates once the sequence count is large; when sizing
// -slots, that is the term to think about, not -ctx.
func suggestCtx(b *budget, slots, prefix, perSeq int) int {
	seqs := slots + prefix
	ctxCost := sub(b.afterModel, b.afterCtx)
	if seqs <= 0 || ctxCost == 0 {
		return perSeq / 2
	}
	spare := sub(b.afterModel, thinHeadroom) // what the context may cost at all
	perToken := float64(ctxCost) / float64(perSeq*seqs)
	fit := int(float64(spare) / perToken / float64(seqs))

	pow := 1024
	for pow*2 <= fit {
		pow *= 2
	}
	if pow >= perSeq {
		// The context is not what is too big — the model is.
		return perSeq / 2
	}
	return pow
}

func sub(a, b uint64) uint64 {
	if a < b {
		return 0
	}
	return a - b
}

func gib(n uint64) string {
	return fmt.Sprintf("%.1f GiB", float64(n)/(1<<30))
}
