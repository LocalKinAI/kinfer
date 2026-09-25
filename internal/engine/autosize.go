package engine

import (
	"fmt"

	"github.com/LocalKinAI/kinfer/internal/store"
)

// Sizing a server when the operator gave no numbers.
//
// Ollama needs none: it reads the context the model was trained for, looks at
// the memory left after the weights, and picks a concurrency that fits. kinfer
// asked for -ctx, -slots and -prefix, and the person choosing them — the one
// who wrote the arithmetic below — chose wrong twice in one evening: a 32768
// context that refused a 45k-token prompt Ollama took without comment, and a
// disabled prefix pool that re-prefilled that prompt on every turn. Neither
// was the engine failing. Both looked exactly like it, from the outside.
//
// So the numbers are now derived, and the flags override. The policy:
//
//   - Spend at most half of what is left after the weights on the cache. The
//     other half is for company: the free-memory figure is per process (see
//     budget.go) and a machine that serves a fleet is rarely running one thing.
//   - Give each conversation the context it was trained for, rounded down to a
//     power of two, and only as far as the budget allows.
//   - When that comes out below what a real prompt needs, give up slots before
//     giving up context. A server with eight slots of 4096 answers nothing an
//     agent sends; one slot of 32768 answers it slowly. Slow is a number and
//     nothing is a bug report.
//   - Even one slot keeps a pool entry. An agent's next turn is the request
//     that reuses a prefix most, and a machine with room for one conversation
//     is exactly where prefilling it again every turn hurts.
//   - When one slot with a pool of its own is still short of the floor, the
//     pool gives up its cells before the conversation gives up context. Its
//     entries then live in whatever the slot is not using, and are dropped
//     when the slot needs the room: an agent's own next turn is a prefix of
//     what the slot already holds, so it costs nothing extra, and anything
//     else simply finds the pool empty. On the box that is one conversation of
//     32768 with its turns reused, instead of 16384 or no pool.

// ctxFloor is the context below which slots are sacrificed instead. Agents'
// prompts on the fleet this was built for run 3k–25k tokens; a person with a
// long system prompt sent 45k; a coding agent's opening prompt alone is 7,428
// tokens (Codex) and about 20k (Claude Code), before a single tool result has
// come back. It was 8192: on the 96 GB box with a 73.4 GiB model that sized
// four slots of 8192, halved to 4096 once measured, and Codex's first prompt
// was refused. A model trained for less is floored at what it was trained for.
const ctxFloor = 32768

// Plan is what serve runs with.
type Plan struct {
	Slots, Prefix, PerSeq int
	// Estimated is the cache cost the plan was sized against; zero when the
	// shape was unknown and the numbers are the static defaults.
	Estimated int64
	// Capped reports that the context is the model's trained maximum, not
	// what memory would have allowed.
	Capped bool
	// SharedPool is a pool with no cells of its own: its entries use what the
	// slots leave free, so the cache holds Slots conversations, not
	// Slots+Prefix.
	SharedPool bool
}

// Cells is how many conversations' worth of tokens the cache holds.
func (p Plan) Cells() int {
	if p.SharedPool {
		return p.Slots
	}
	return p.Slots + p.Prefix
}

// AutoSize completes a configuration: any of slots, prefix and perSeq that is
// zero is derived from the model's shape and the memory left after its weights.
// prefix < 0 means no pool.
//
// An unknown shape (a GGUF whose header lacks the attention geometry) yields
// the static defaults rather than a guess dressed up as a measurement.
func AutoSize(sh store.Shape, free uint64, slots, prefix, perSeq int) (Plan, error) {
	prefixAuto := prefix == 0
	if prefix < 0 {
		prefix = 0
	} else if prefix == 0 {
		prefix = DefaultPrefixSlots
	}
	known := sh.CacheBytes(1024) != 0 && free != 0
	if slots > 0 && perSeq > 0 {
		if prefixAuto && known {
			return poolAround(sh, spendableOf(free), slots, prefix, perSeq), nil
		}
		return Plan{Slots: slots, Prefix: prefix, PerSeq: perSeq,
			Estimated: sh.CacheBytesFor(perSeq*(slots+prefix), slots+prefix)}, nil
	}
	if !known {
		if slots <= 0 {
			slots = DefaultSlots
		}
		if perSeq <= 0 {
			perSeq = DefaultContextSize
		}
		return Plan{Slots: slots, Prefix: prefix, PerSeq: perSeq}, nil
	}

	spendable := spendableOf(free)
	capCtx := sh.TrainedCtx
	if capCtx <= 0 {
		capCtx = 32768
	}
	floor := min(ctxFloor, capCtx)

	// A fixed context: find the most slots that carry it.
	if perSeq > 0 {
		for s := DefaultSlots; s >= 1; s /= 2 {
			p := min(prefix, max(s/2, 1))
			if sh.CacheBytesFor(perSeq*(s+p), s+p) <= spendable {
				return Plan{Slots: s, Prefix: p, PerSeq: perSeq,
					Estimated: sh.CacheBytesFor(perSeq*(s+p), s+p)}, nil
			}
		}
		if prefixAuto && prefix > 0 && sh.CacheBytesFor(perSeq, 2) <= spendable {
			return Plan{Slots: 1, Prefix: 1, PerSeq: perSeq, SharedPool: true,
				Estimated: sh.CacheBytesFor(perSeq, 2)}, nil
		}
		return Plan{}, fmt.Errorf("a %d-token context does not fit even for one conversation: "+
			"the cache would need %s and %s is left after the weights",
			perSeq, store.HumanSize(sh.CacheBytesFor(perSeq, 1)), store.HumanSize(int64(free)))
	}

	// Fixed slots, or the default: the largest context that fits them, and if
	// that is too small to be useful, fewer slots.
	start := slots
	if start <= 0 {
		start = DefaultSlots
	}
	var last Plan
	for s := start; s >= 1; s /= 2 {
		p := min(prefix, max(s/2, 1))
		plan := sizedPlan(sh, spendable, capCtx, s, p, false)
		// The last conversation left, or the operator's own count, still short
		// of the floor: the pool shares its cells rather than cost context.
		if plan.PerSeq < floor && p > 0 && prefixAuto && (s == 1 || slots > 0) {
			if shared := sizedPlan(sh, spendable, capCtx, s, p, true); shared.PerSeq > plan.PerSeq {
				plan = shared
			}
		}
		if plan.PerSeq == 0 {
			continue
		}
		last = plan
		if plan.PerSeq >= floor || slots > 0 {
			return last, nil
		}
	}
	if last.PerSeq > 0 {
		return last, nil // one slot, a short context: it will answer something
	}
	return Plan{}, fmt.Errorf("no room for a context: %s left after the weights", store.HumanSize(int64(free)))
}

// Sizing the total a fixed number of slots share, the way Ollama sizes one
// request's context.
//
// Ollama's rule, from its server/routes.go: a request that names no context
// gets 262144 tokens on a machine with at least 47 GiB of GPU memory, 32768
// with 23, 4096 below that, never more than the model was trained for. That is
// why ornith-1.5-35b and qwen3.8 both loaded at 262144 on the box. The number
// is per machine, not per model. It only worked for those two because they
// are small; Flash-Next at 262144 would need 7 GiB of cache where its weights
// leave 4.4.
//
// kinfer takes the same number as the total its slots share. At home each slot
// gets an equal part, and flex gives one long prompt the whole of it (see
// flex.go). Where the total does not fit it is halved until it does, leaving
// fitReserve free. On the box: ornith gets 262144 as 4 x 65536, the memory
// Ollama gives its single request, with four conversations at once. Flash-Next
// gets 65536 as 4 x 16384, which is what -ctx 16384 -slots 4 used to write down
// for every model.

// ollamaDefaultCtx is the context Ollama gives a request that names none, for a
// machine with this much GPU memory.
func ollamaDefaultCtx(budget uint64) int {
	switch {
	case budget >= 47<<30:
		return 262144
	case budget >= 23<<30:
		return 32768
	}
	return 4096
}

// fitReserve is the least an automatically sized total leaves free. Below
// thinHeadroom the server still says so. Flash-Next at 4 x 16384 leaves 1.6
// GiB and ran the deep-study swarm that way for a day. Every failure seen on
// the box came from another process's memory, which no reserve kept here can
// see.
const fitReserve = 1 << 30

// minSlotCtx is the least a home slot is given when Ollama's number is small:
// on a 16 GB machine 4096 split four ways answers nothing. flex still gives a
// long prompt the whole total.
const minSlotCtx = 4096

// SizeTotal is the plan for slots conversations sharing a context sized as
// above, for a machine whose GPU budget is budget with free left after the
// weights. PerSeq is each slot's share at home; Slots x PerSeq is the total.
func SizeTotal(sh store.Shape, budget, free uint64, slots int) (Plan, error) {
	if slots <= 0 {
		slots = 1
	}
	one := ollamaDefaultCtx(budget)
	if sh.TrainedCtx > 0 {
		one = min(one, sh.TrainedCtx)
	}
	total := max(one, slots*minSlotCtx)
	if sh.TrainedCtx > 0 {
		total = min(total, slots*sh.TrainedCtx)
	}
	share := func(t int) int { return t / slots / 256 * 256 }
	if sh.CacheBytes(1024) == 0 || free == 0 {
		// No shape or no accelerator to measure against: Ollama's number, as is.
		return Plan{Slots: slots, PerSeq: share(total)}, nil
	}
	spend := int64(sub(free, fitReserve))
	for ; share(total) >= 1024; total /= 2 {
		if c := sh.CacheBytesFor(share(total)*slots, slots); c <= spend {
			return Plan{Slots: slots, PerSeq: share(total), Estimated: c,
				Capped: sh.TrainedCtx > 0 && share(total)*slots >= sh.TrainedCtx}, nil
		}
	}
	return Plan{}, fmt.Errorf("%d slots do not fit even at 1024 tokens each: %s left after the weights",
		slots, store.HumanSize(int64(free)))
}

// The micro-batch, sized the way Ollama sizes it.
//
// llama.cpp reads a prompt n_ubatch tokens at a time, 512 unless told
// otherwise, and a bigger pass keeps the GPU busier. Ollama started the box's
// ornith-1.5-35b with -ub 2048 and read an 11,479-token prompt in 5.9 s, where
// kinfer at 512 took 7.7. Its rule is automaticGenerationBatch in
// server/sched.go. It starts at 2048 for a context over 32768 and at 1024 over
// 4096, and steps down while the model and its cache (predicted) are too much
// of the memory (available): 2048 needs them under 60% of it and 2 GiB to
// spare within 80%; 1024 needs 75% and 768 MiB. Flash-Next's 76 GiB of 77.8
// gets 512, as before; ornith's 25.6 gets 2048.
const (
	ubatchDefault = 512
	ubatchMedium  = 1024
	ubatchLarge   = 2048
)

// autoUbatch is Ollama's micro-batch for a model whose longest conversation is
// longest tokens.
func autoUbatch(longest int, predicted, available uint64) int {
	b := ubatchDefault
	switch {
	case longest > 32768:
		b = ubatchLarge
	case longest > 4096:
		b = ubatchMedium
	}
	for b > ubatchDefault && !ubatchFits(b, predicted, available) {
		b = lowerUbatch(b)
	}
	return b
}

func ubatchFits(b int, predicted, available uint64) bool {
	if predicted == 0 || available == 0 {
		return true
	}
	within := available * 80 / 100
	if predicted > within {
		return false
	}
	switch {
	case b >= ubatchLarge:
		return predicted <= available*60/100 && 2<<30 <= within-predicted
	case b >= ubatchMedium:
		return predicted <= available*75/100 && 768<<20 <= within-predicted
	}
	return true
}

func lowerUbatch(b int) int {
	if b > ubatchMedium {
		return ubatchMedium
	}
	return ubatchDefault
}

// spendableOf is what the cache may cost: half of what the weights left, and
// never the last thinHeadroom of it. The other half is for company — see the
// policy at the top of this file.
func spendableOf(free uint64) int64 {
	return int64(min(free/2, sub(free, thinHeadroom)))
}

// poolAround sizes an automatic prefix pool around conversations the operator
// fixed with -slots and -ctx. Their numbers are theirs and stand; the pool is
// ours, and it gets only what they leave.
//
// It used to get DefaultPrefixSlots entries whatever that cost, because a
// fixed -slots and -ctx returned before any arithmetic. On the box,
// `-ctx 16384 -slots 4` with the pool left automatic asked for eight sequences
// of 16384: 0.0 GiB left after loading, and the first request failed inside
// llama.cpp (code -3). Now the pool takes cells of its own if they fit, shares
// the slots' cells if only that fits, and is left out if neither does — the
// operator's conversations never pay for a pool nobody asked for. The measured
// check in Open backs this up, because a hybrid model's cache can cost more
// than this estimate.
func poolAround(sh store.Shape, spendable int64, slots, want, perSeq int) Plan {
	most := min(want, max(slots/2, 1))
	for p := most; p >= 1; p-- {
		if c := sh.CacheBytesFor(perSeq*(slots+p), slots+p); c <= spendable {
			return Plan{Slots: slots, Prefix: p, PerSeq: perSeq, Estimated: c}
		}
	}
	for p := most; p >= 1; p-- {
		if c := sh.CacheBytesFor(perSeq*slots, slots+p); c <= spendable {
			return Plan{Slots: slots, Prefix: p, PerSeq: perSeq, SharedPool: true, Estimated: c}
		}
	}
	return Plan{Slots: slots, PerSeq: perSeq, Estimated: sh.CacheBytesFor(perSeq*slots, slots)}
}

// sizedPlan is the largest context s slots and p pool entries fit in, capped
// at what the model was trained for; a zero PerSeq when none does.
func sizedPlan(sh store.Shape, spendable int64, capCtx, s, p int, shared bool) Plan {
	plan := Plan{Slots: s, Prefix: p, SharedPool: shared}
	ctx := largestCtx(sh, spendable, plan.Cells(), s+p)
	if ctx == 0 {
		return Plan{}
	}
	if ctx >= capCtx {
		ctx, plan.Capped = capCtx, true
	}
	plan.PerSeq = ctx
	plan.Estimated = sh.CacheBytesFor(ctx*plan.Cells(), s+p)
	return plan
}

// LargestCtx is the biggest per-conversation context whose cache for seqs
// sequences fits in spendable, rounded down to a power of two.
func LargestCtx(sh store.Shape, spendable int64, seqs int) int {
	return largestCtx(sh, spendable, seqs, seqs)
}

// largestCtx is LargestCtx for a cache holding cells conversations' worth of
// tokens across seqs sequences; they differ when the pool shares cells.
func largestCtx(sh store.Shape, spendable int64, cells, seqs int) int {
	if spendable <= 0 || cells <= 0 {
		return 0
	}
	for ctx := 1 << 20; ctx >= 512; ctx /= 2 {
		if sh.CacheBytesFor(ctx*cells, seqs) <= spendable {
			return ctx
		}
	}
	return 0
}
