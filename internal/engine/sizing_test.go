package engine

import "testing"

// The box, 2026-11-10 21:33: qwen3.8-27b-q4_k_m in a runner beside ornith and
// the speech models, with the service's -slots 4 -prefix auto and the context
// left to choose. Measured, 4 slots + 2 prefix x 32768 left 0.0 GiB, and the
// pool went to 1; 4 + 1 left 0.8, and it went to 0 — every turn of kinhome's
// test home read whole again, 43–50 s to a first token. Now the context is
// halved first, and the last entry stays.
func TestShrinkKeepsThePoolsLastEntry(t *testing.T) {
	box := sizeRules{slotsAuto: false, prefixAuto: true, ctxAuto: true, floor: ctxFloor}
	s := sizeState{slots: 4, prefix: 2, perSeq: 32768, ub: ubatchDefault, shared: true}

	s, why, ok := shrink(s, box)
	if !ok || s.prefix != 1 || s.perSeq != 32768 {
		t.Fatalf("first step: %+v (%s), want 1 prefix x 32768", s, why)
	}
	s, why, ok = shrink(s, box)
	if !ok || s.prefix != 1 || s.perSeq != 16384 {
		t.Fatalf("second step: %+v (%s), want the context halved and the entry kept", s, why)
	}
	// Below poolCtxMin the context is not halved again for the pool: the
	// entry goes, as it did.
	s, why, ok = shrink(s, box)
	if !ok || s.prefix != 0 || s.perSeq != 16384 {
		t.Fatalf("third step: %+v (%s), want the pool gone at 16384", s, why)
	}
}

func TestShrinkLeavesFixedLayoutsAndOperatorsContextAlone(t *testing.T) {
	s := sizeState{slots: 4, prefix: 1, perSeq: 32768, ub: ubatchDefault, shared: true}
	for name, r := range map[string]sizeRules{
		"-flex=false": {prefixAuto: true, ctxAuto: true, fixedLayout: true, floor: ctxFloor},
		"-ctx given":  {prefixAuto: true, ctxAuto: false, floor: ctxFloor},
	} {
		got, why, ok := shrink(s, r)
		if !ok || got.perSeq != 32768 || got.prefix != 0 {
			t.Errorf("%s: %+v (%s), want the pool's entry to go and the context kept", name, got, why)
		}
	}
}

// What came before the rule is unchanged: slots that are the server's to
// choose go before the pool, and a micro-batch above the default before
// either.
func TestShrinkOrderBeforeThePool(t *testing.T) {
	s := sizeState{slots: 4, prefix: 1, perSeq: 32768, ub: ubatchLarge, shared: true}
	r := sizeRules{ubAuto: true, slotsAuto: true, prefixAuto: true, ctxAuto: true, floor: ctxFloor}
	s, _, _ = shrink(s, r)
	if s.ub != lowerUbatch(ubatchLarge) || s.slots != 4 {
		t.Fatalf("got %+v, want the micro-batch lowered first", s)
	}
	s.ub = ubatchDefault
	s, _, _ = shrink(s, r)
	if s.slots != 2 || s.prefix != 1 || s.perSeq != 32768 {
		t.Fatalf("got %+v, want 2 slots before the pool or the context", s)
	}
}
