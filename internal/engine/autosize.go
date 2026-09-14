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

// ctxFloor is the context below which slots are sacrificed instead. Agents'
// prompts on the fleet this was built for run 3k–25k tokens; a person with a
// long system prompt sent 45k. Below this the server refuses what it is for.
const ctxFloor = 8192

// Plan is what serve runs with.
type Plan struct {
	Slots, Prefix, PerSeq int
	// Estimated is the cache cost the plan was sized against; zero when the
	// shape was unknown and the numbers are the static defaults.
	Estimated int64
	// Capped reports that the context is the model's trained maximum, not
	// what memory would have allowed.
	Capped bool
}

// AutoSize completes a configuration: any of slots, prefix and perSeq that is
// zero is derived from the model's shape and the memory left after its weights.
// prefix < 0 means no pool.
//
// An unknown shape (a GGUF whose header lacks the attention geometry) yields
// the static defaults rather than a guess dressed up as a measurement.
func AutoSize(sh store.Shape, free uint64, slots, prefix, perSeq int) (Plan, error) {
	if prefix < 0 {
		prefix = 0
	} else if prefix == 0 {
		prefix = DefaultPrefixSlots
	}
	if slots > 0 && perSeq > 0 {
		return Plan{Slots: slots, Prefix: prefix, PerSeq: perSeq,
			Estimated: sh.CacheBytesFor(perSeq*(slots+prefix), slots+prefix)}, nil
	}
	if sh.CacheBytes(1024) == 0 || free == 0 {
		if slots <= 0 {
			slots = DefaultSlots
		}
		if perSeq <= 0 {
			perSeq = DefaultContextSize
		}
		return Plan{Slots: slots, Prefix: prefix, PerSeq: perSeq}, nil
	}

	spendable := int64(min(free/2, sub(free, thinHeadroom)))
	capCtx := sh.TrainedCtx
	if capCtx <= 0 {
		capCtx = 32768
	}

	// A fixed context: find the most slots that carry it.
	if perSeq > 0 {
		for s := DefaultSlots; s >= 1; s /= 2 {
			p := min(prefix, s/2)
			if sh.CacheBytesFor(perSeq*(s+p), s+p) <= spendable {
				return Plan{Slots: s, Prefix: p, PerSeq: perSeq,
					Estimated: sh.CacheBytesFor(perSeq*(s+p), s+p)}, nil
			}
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
		p := min(prefix, s/2)
		ctx := LargestCtx(sh, spendable, s+p)
		if ctx == 0 {
			continue
		}
		capped := ctx >= capCtx
		if capped {
			ctx = capCtx
		}
		last = Plan{Slots: s, Prefix: p, PerSeq: ctx, Capped: capped,
			Estimated: sh.CacheBytesFor(ctx*(s+p), s+p)}
		if ctx >= ctxFloor || slots > 0 {
			return last, nil
		}
	}
	if last.PerSeq > 0 {
		return last, nil // one slot, a short context: it will answer something
	}
	return Plan{}, fmt.Errorf("no room for a context: %s left after the weights", store.HumanSize(int64(free)))
}

// LargestCtx is the biggest per-conversation context whose cache for seqs
// sequences fits in spendable, rounded down to a power of two.
func LargestCtx(sh store.Shape, spendable int64, seqs int) int {
	if spendable <= 0 || seqs <= 0 {
		return 0
	}
	for ctx := 1 << 20; ctx >= 512; ctx /= 2 {
		if sh.CacheBytesFor(ctx*seqs, seqs) <= spendable {
			return ctx
		}
	}
	return 0
}
