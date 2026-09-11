package engine

import (
	"fmt"
	"log"

	"github.com/LocalKinAI/kinfer/internal/llama"
)

// Memory budgeting for the accelerator.
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

// thinHeadroom is how much of the accelerator's budget should still be free
// once the model, its KV cache and the reserved compute graph are in place.
//
// A heuristic from three observations, all on one 96 GB Mac Studio whose Metal
// budget is 77.8 GiB, serving one 73.4 GiB model:
//
//	free 1.6 GiB  (-ctx 16384 -slots 4)  died mid-batch, taking the model with it
//	free 2.7 GiB  (-ctx 16384 -slots 2)  served for hours
//	free 3.1 GiB  (-ctx  2048 -slots 4)  survived 8 concurrent streams
//
// So the line sits at 2 GiB, between the one that died and the ones that did
// not. Absolute rather than a fraction of the budget: what fails is a transient
// allocation for one batch, and Metal sizes that command buffer by the batch,
// not by the machine. Three points on one machine is a rule of thumb, not a
// law — which is the other reason this only warns.
//
// The reading is also taken at the wrong moment to be complete. It happens once
// the context exists, and the allocation most likely to exhaust what is left is
// the prefill batch, which has not been built yet. A sibling session
// demonstrated the point cleanly by running two 0.5B models — 1.6 GiB of
// weights between them — into the same OOM with 3000-token prompts. kinfer
// chunks prefill at 512 tokens per slot so its batch is bounded, but bounded is
// not accounted for, and the ceiling rises with -slots. A configuration that
// clears this check can still be too tight for its own prompts.
const thinHeadroom = 2 << 30 // 2 GiB

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
func (b *budget) report(slots, prefix, perSeq int) {
	if !b.known {
		return
	}

	model := sub(b.atStart, b.afterModel)
	ctx := sub(b.afterModel, b.afterCtx)
	free := b.afterCtx
	total := b.dev.Total

	// The multiplication is spelled out because it is the part that surprises
	// people: -ctx reads like a total and is per-conversation.
	log.Printf("memory: %s   model %s + context %s -> %s free of %s\n"+
		"          context is %d tokens: -ctx %d per conversation x %d sequences (%d slots + %d prefix)",
		b.dev.Name, gib(model), gib(ctx), gib(free), gib(total),
		perSeq*(slots+prefix), perSeq, slots+prefix, slots, prefix)

	if total == 0 || free >= thinHeadroom {
		return
	}

	log.Printf("memory: WARNING — only %s of %s is left, under the %s this needs.\n"+
		"          A batch that cannot allocate fails the whole model, not one request:\n"+
		"          llama.cpp reports it fatally, kinfer retires the model, and every\n"+
		"          in-flight and queued request ends at once.\n"+
		"          Try -ctx %d -slots %d, or keep -ctx %d and drop to -slots %d.\n"+
		"          The cache will shrink by less than the ratio suggests: part of it is\n"+
		"          per-sequence state that does not scale with the context length.\n"+
		"          Free memory also counts other processes, so this margin moves.",
		gib(free), gib(total), gib(thinHeadroom),
		suggestCtx(b, slots, prefix, perSeq), slots,
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
