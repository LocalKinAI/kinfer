package sampling

import (
	"testing"

	"github.com/LocalKinAI/kinfer/internal/llama"
)

// The sampling pipeline itself now lives in llama.cpp, which has its own tests.
// What is still kinfer's to get wrong is the translation: which stages go into
// the chain, in what order, and how "0 means random" maps onto llama.cpp's
// sentinel. Building a chain needs the library bound, so the shape of the chain
// is covered by cmd/probe end to end; these cover the pure decisions.

func TestDefaultParamsAreUsable(t *testing.T) {
	p := DefaultParams()
	if p.Temperature <= 0 {
		t.Error("default temperature must be > 0, or every reply is greedy and loops")
	}
	if p.TopK <= 0 && (p.TopP <= 0 || p.TopP >= 1) {
		t.Error("defaults narrow nothing: the full 151,936-token distribution would be drawn from")
	}
	if PenaltiesOn(p) {
		t.Errorf("repeat penalty %v by default: it is off unless a request asks for it", p.RepeatPenalty)
	}
	if p.RepeatLastN <= 0 {
		t.Error("a request that asks for a repeat penalty would have no history to apply it to")
	}
}

func TestSeedZeroMeansRandom(t *testing.T) {
	if got := seedOf(0); got != llama.DefaultSeed {
		t.Errorf("seedOf(0) = %#x, want LLAMA_DEFAULT_SEED %#x", got, llama.DefaultSeed)
	}
	if got := seedOf(42); got != 42 {
		t.Errorf("seedOf(42) = %d, want 42 — a fixed seed must reach llama.cpp unchanged", got)
	}
}
