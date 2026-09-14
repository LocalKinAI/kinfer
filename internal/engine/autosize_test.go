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
	if p.Prefix > p.Slots/2 {
		t.Errorf("prefix %d for %d slots — the pool must not outnumber the conversations it serves", p.Prefix, p.Slots)
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
