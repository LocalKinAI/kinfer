package mlxtts

import (
	"math"
	"testing"
)

// The window a pass may attend to is the reference decoder's whatever the
// pieces the frames come in: a frame attends to itself and the W-1 before it.
func TestWindowIsTheReferenceMask(t *testing.T) {
	const W, T = 72, 200
	for _, pieces := range [][]int{{200}, {50, 50, 50, 50}, {25, 1, 1, 10, 25, 1, 30, 1, 6, 100}, {71, 1, 1, 127}, {72, 128}, {73, 127}} {
		offset := 0
		for _, L := range pieces {
			start, mask := window(offset, L, W)
			S := offset + L - start
			for i := range L {
				q := offset + i
				for k := range offset + L {
					want := k <= q && k > q-W
					got := k >= start
					if got && mask != nil {
						got = !math.IsInf(float64(mask[i*S+(k-start)]), -1)
					} else if got && L > 1 {
						got = k <= q // no mask given: the fused causal one
					}
					if got != want {
						t.Fatalf("pieces %v: frame %d attends to %d: %v, want %v (offset %d, L %d, start %d, mask %v)", pieces, q, k, got, want, offset, L, start, mask != nil)
					}
				}
			}
			offset += L
		}
		if offset != T {
			t.Fatalf("pieces %v add up to %d", pieces, offset)
		}
	}
	if start, mask := window(500, 50, 0); start != 0 || mask != nil {
		t.Errorf("no window set: start %d, mask %v", start, mask != nil)
	}
}
