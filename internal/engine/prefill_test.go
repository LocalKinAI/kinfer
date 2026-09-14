package engine

import (
	"testing"
	"time"
)

// The order prompts are fed is the order they arrived. Getting this wrong is
// not a fairness nicety: fair sharing made every request wait for the largest
// one, and a 3k-token prompt saw its first token no sooner than a 25k one.
func TestPrefillOrderIsArrivalOrder(t *testing.T) {
	t0 := time.Now()
	late := &slot{seq: 0, admitted: t0.Add(2 * time.Second)}
	first := &slot{seq: 1, admitted: t0}
	mid := &slot{seq: 2, admitted: t0.Add(time.Second)}

	got := prefillOrder([]*slot{late, first, mid})
	if got[0] != first || got[1] != mid || got[2] != late {
		t.Errorf("order = seq %d,%d,%d; want 1,2,0 (arrival order, not slot order)",
			got[0].seq, got[1].seq, got[2].seq)
	}
}

// Two arrivals in the same instant keep their slot order, so the result is
// deterministic rather than dependent on sort internals.
func TestPrefillOrderIsStableForTies(t *testing.T) {
	t0 := time.Now()
	a := &slot{seq: 0, admitted: t0}
	b := &slot{seq: 1, admitted: t0}
	got := prefillOrder([]*slot{b, a})
	if got[0] != b || got[1] != a {
		t.Error("equal arrival times were reordered")
	}
}
