package engine

import (
	"testing"

	"github.com/LocalKinAI/kinfer/internal/store"
)

// The 35B hybrid MoE this was tuned on, as its header describes it: 40
// layers of which every fourth attends, 2 KV heads of 256, and a recurrent
// state on the other 30. 20 KiB of cache per token plus ~63 MiB per sequence
// — llama.cpp's own figures at load time.
var moe = store.Shape{Layers: 40, HeadsKV: 2, Heads: 16, EmbedDim: 4096, KeyLen: 256, ValLen: 256,
	TrainedCtx: 262144, Experts: 256, AttnInterval: 4, SSMConv: 4, SSMState: 128, SSMGroups: 16, SSMInner: 4096}

const GiB = 1 << 30

// The evening this exists because of: 45.7 GiB left after the weights, an
// operator who typed -ctx 32768, and a 45k-token prompt refused. Unasked, the
// server must give each of eight conversations more than that.
func TestAutoSizeGivesAgentsRoom(t *testing.T) {
	p, err := AutoSize(moe, 457*GiB/10, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots != DefaultSlots || p.Prefix != DefaultPrefixSlots {
		t.Errorf("plenty of memory, got %d slots + %d prefix, want %d + %d", p.Slots, p.Prefix, DefaultSlots, DefaultPrefixSlots)
	}
	// 65536 is the number that worked that night: 12 x 65536 x 20 KiB is
	// 15 GiB; the next power of two is 30 and over the half it may spend.
	if p.PerSeq != 65536 {
		t.Errorf("per-conversation context %d, want 65536 — a 45k prompt must fit, and 131072 would spend %s", p.PerSeq, "30 GiB of 45.7")
	}
	if p.Estimated > 457*GiB/20 {
		t.Errorf("cache estimate %d spends more than half of what is left", p.Estimated)
	}
}

// With memory to spare the context stops at what the model was trained for.
func TestAutoSizeCapsAtTrainedContext(t *testing.T) {
	small := store.Shape{Layers: 24, HeadsKV: 2, Heads: 14, EmbedDim: 896, TrainedCtx: 32768}
	p, err := AutoSize(small, 60*GiB, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if p.PerSeq != 32768 || !p.Capped {
		t.Errorf("got %d tokens capped=%v, want the trained 32768, capped", p.PerSeq, p.Capped)
	}
}

// When the weights leave little, slots go before context does. Eight slots
// of 2048 answer nothing an agent sends; one of 16384 answers it.
func TestAutoSizeSheddsSlotsBeforeContext(t *testing.T) {
	p, err := AutoSize(moe, 44*GiB/10, 0, 0, 0) // the 125B: 4.4 GiB left of 77.8
	if err != nil {
		t.Fatal(err)
	}
	if p.PerSeq < ctxFloor {
		t.Errorf("context %d is below the floor %d while %d slots remain — should have shed slots", p.PerSeq, ctxFloor, p.Slots)
	}
	if p.Slots >= DefaultSlots {
		t.Errorf("kept %d slots on 4.4 GiB; expected fewer", p.Slots)
	}
	if p.Prefix > max(p.Slots/2, 1) {
		t.Errorf("prefix %d for %d slots — the pool must not outnumber the conversations it serves", p.Prefix, p.Slots)
	}
}

// A machine with room for one conversation of agent size keeps that one — and
// a pool entry for it, since its next turn is what reuses a prefix most. The
// old floor of 8192 kept two slots of 16384 here; the old prefix rule gave one
// slot no pool at all.
func TestAutoSizeKeepsOneAgentSizedSlotWithAPoolEntry(t *testing.T) {
	p, err := AutoSize(moe, 36*GiB/10, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots != 1 || p.Prefix != 1 || p.PerSeq < ctxFloor {
		t.Errorf("got %d slots + %d prefix x %d; want 1 + 1 x at least %d", p.Slots, p.Prefix, p.PerSeq, ctxFloor)
	}
}

// Less again, and one slot with a pool of its own is short of the floor: the
// pool borrows the slot's cells rather than halve the context. This is the
// box with Qwen3.8-Flash-Next, which sized 1 + 1 x 16384 without it — below
// Claude Code's opening prompt.
func TestAutoSizeSharesThePoolsCellsBeforeGivingUpContext(t *testing.T) {
	p, err := AutoSize(moe, 3*GiB, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots != 1 || p.Prefix != 1 || !p.SharedPool || p.PerSeq < ctxFloor {
		t.Errorf("got %d slots + %d prefix x %d, shared=%v; want 1 + 1 x at least %d, shared", p.Slots, p.Prefix, p.PerSeq, p.SharedPool, ctxFloor)
	}
	if p.Cells() != 1 {
		t.Errorf("a shared pool's cache holds %d conversations; want the slot's 1", p.Cells())
	}
	// An operator who named a pool size gets that pool, with cells of its own.
	q, err := AutoSize(moe, 3*GiB, 0, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if q.SharedPool {
		t.Errorf("-prefix 2 was given and the pool was made to share anyway: %+v", q)
	}
}

// A model trained for less than the floor cannot reach it by shedding slots,
// so it is not asked to: the floor is what it was trained for.
func TestAutoSizeFloorsAShortModelAtItsTrainedContext(t *testing.T) {
	short := store.Shape{Layers: 24, HeadsKV: 2, Heads: 14, EmbedDim: 896, TrainedCtx: 16384}
	p, err := AutoSize(short, 60*GiB, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots != DefaultSlots || p.PerSeq != 16384 {
		t.Errorf("got %d slots x %d; want all %d slots at the trained 16384", p.Slots, p.PerSeq, DefaultSlots)
	}
}

// An explicit -ctx keeps its value; slots are chosen around it.
func TestAutoSizeHonoursAFixedContext(t *testing.T) {
	p, err := AutoSize(moe, 457*GiB/10, 0, 0, 131072)
	if err != nil {
		t.Fatal(err)
	}
	if p.PerSeq != 131072 {
		t.Errorf("context %d, want the 131072 that was asked for", p.PerSeq)
	}
	if p.Slots < 1 || p.Slots > DefaultSlots {
		t.Errorf("slots %d out of range", p.Slots)
	}
	if _, err := AutoSize(moe, 2*GiB, 0, 0, 1<<20); err == nil {
		t.Error("a 1M context in 2 GiB was accepted; it cannot fit and must say so")
	}
}

// Explicit numbers are never second-guessed: the operator measured, or is
// reproducing something.
func TestAutoSizeLeavesExplicitNumbersAlone(t *testing.T) {
	p, err := AutoSize(moe, 1*GiB, 16, -1, 32768)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots != 16 || p.Prefix != 0 || p.PerSeq != 32768 {
		t.Errorf("got %+v, want 16 slots, no prefix, 32768", p)
	}
}

// A header without the attention geometry gets the static defaults, labelled
// as such by a zero estimate, not a number invented from nothing.
func TestAutoSizeFallsBackWithoutAShape(t *testing.T) {
	p, err := AutoSize(store.Shape{}, 45*GiB, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots != DefaultSlots || p.PerSeq != DefaultContextSize || p.Estimated != 0 {
		t.Errorf("got %+v, want the defaults with no estimate", p)
	}
}

// freeFor is the memory after the weights that leaves exactly spendable for
// the cache (see spendableOf).
func freeFor(spendable int64) uint64 {
	if spendable >= thinHeadroom {
		return uint64(2 * spendable)
	}
	return uint64(spendable + thinHeadroom)
}

// An automatic pool around conversations the operator fixed gets only what
// they leave. It used to get DefaultPrefixSlots entries whatever that cost:
// `-ctx 16384 -slots 4` on the box asked for eight sequences and left 0.0 GiB.
func TestAutoSizeFitsAnAutomaticPoolAroundFixedConversations(t *testing.T) {
	const slots, ctx = 4, 16384
	base := moe.CacheBytesFor(ctx*slots, slots)
	shared := moe.CacheBytesFor(ctx*slots, slots+2)
	dedicated := moe.CacheBytesFor(ctx*(slots+2), slots+2)
	if !(base < shared && shared < moe.CacheBytesFor(ctx*(slots+1), slots+1)) {
		t.Fatalf("the shape no longer separates the cases: base %d shared %d", base, shared)
	}

	for _, c := range []struct {
		name       string
		spendable  int64
		wantPrefix int
		wantShared bool
	}{
		{"room for cells of its own", dedicated, 2, false},
		{"room only to share the slots' cells", shared, 2, true},
		{"no room at all", base - 1, 0, false},
	} {
		p, err := AutoSize(moe, freeFor(c.spendable), slots, 0, ctx)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if p.Slots != slots || p.PerSeq != ctx {
			t.Errorf("%s: the operator's %d slots x %d became %d x %d", c.name, slots, ctx, p.Slots, p.PerSeq)
		}
		if p.Prefix != c.wantPrefix || p.SharedPool != c.wantShared {
			t.Errorf("%s: pool %d (shared %v), want %d (shared %v)", c.name, p.Prefix, p.SharedPool, c.wantPrefix, c.wantShared)
		}
		// The operator's own conversations cost what they cost; the pool must
		// not push the plan past that or past the budget, whichever is more.
		if limit := max(c.spendable, base); p.Estimated > limit {
			t.Errorf("%s: plan costs %d, over the %d that the slots or the budget allow", c.name, p.Estimated, limit)
		}
	}

	// A count the operator typed is theirs, however tight.
	p, err := AutoSize(moe, freeFor(base-1), slots, 3, ctx)
	if err != nil || p.Prefix != 3 {
		t.Errorf("-prefix 3 became %d (%v); explicit numbers are never second-guessed", p.Prefix, err)
	}
}
