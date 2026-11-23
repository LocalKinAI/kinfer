package mlxengine

import (
	"math/rand"
	"slices"
	"testing"
)

// run has d choose the depth of n passes of a text whose i-th draft is kept
// at rate keep[i-1] when those before it were, each pass costing base ms and
// step more for each draft, and returns how often each depth was chosen in
// the last half of them.
func run(d *depths, n int, keep []float64, base, step float64, r *rand.Rand) []int {
	chosen := make([]int, d.max+1)
	for i := 0; i < n; i++ {
		depth := d.next()
		kept := 0
		for kept < depth && r.Float64() < keep[kept] {
			kept++
		}
		d.observe(depth, depth, kept, base+step*float64(depth)+r.Float64())
		if i >= n/2 {
			chosen[depth]++
		}
	}
	return chosen
}

func most(chosen []int) int {
	best := 0
	for d, n := range chosen {
		if n > chosen[best] {
			best = d
		}
	}
	return best
}

// The depth settles where the text pays: shallow for prose, whose drafts are
// rarely kept past the second, deep for text whose drafts nearly all are —
// and moves when the text changes.
func TestDepthFollowsTheText(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	prose := []float64{0.65, 0.45, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4}
	code := []float64{0.95, 0.95, 0.93, 0.93, 0.9, 0.9, 0.9, 0.9}
	d := newDepths(8)
	if got := most(run(d, 600, prose, 30, 3.5, r)); got < 1 || got > 2 {
		t.Errorf("prose: mostly %d drafts a pass, want 1 or 2 (costs %.0f, kept %.2f)", got, d.cost[1:], d.rate[1:])
	}
	if got := most(run(d, 600, code, 30, 3.5, r)); got < 6 {
		t.Errorf("code after prose: mostly %d drafts a pass, want 6 or more (costs %.0f, kept %.2f)", got, d.cost[1:], d.rate[1:])
	}
	if got := most(run(d, 600, prose, 30, 3.5, r)); got > 3 {
		t.Errorf("prose after code: mostly %d drafts a pass, want 3 or fewer (costs %.0f, kept %.2f)", got, d.cost[1:], d.rate[1:])
	}
}

// With -draft N every pass drafts N.
func TestDepthFixed(t *testing.T) {
	d := newDepths(4)
	d.fixed = 4
	r := rand.New(rand.NewSource(2))
	if c := run(d, 100, []float64{0.1, 0.1, 0.1, 0.1}, 30, 5, r); c[4] != 50 {
		t.Errorf("a fixed depth of 4 chose %v", c)
	}
}

// The outcomes recorded ahead are the likeliest: for prose a pass keeping
// none of its drafts or one, for code one keeping all of them.
func TestLikelyOutcomes(t *testing.T) {
	d := newDepths(5)
	for i := 0; i < 200; i++ {
		d.observe(5, 5, []int{0, 0, 1, 0, 1, 2, 0, 1, 0, 0}[i%10], 40)
	}
	if got := d.likely(3)[:2]; !slices.Equal(got, []int{0, 1}) {
		t.Errorf("drafts mostly kept none or one of: likeliest outcomes %v, want [0 1]", got)
	}
	d = newDepths(5)
	for i := 0; i < 200; i++ {
		d.observe(5, 5, []int{5, 5, 5, 5, 5, 5, 5, 5, 5, 0}[i%10], 40)
	}
	if got := d.likely(5)[0]; got != 5 {
		t.Errorf("drafts nearly all kept: likeliest outcome %d, want 5", got)
	}
	if got := d.likely(0); !slices.Equal(got, []int{0}) {
		t.Errorf("a pass of no drafts: outcomes %v, want [0]", got)
	}
}

// Text that turns from prose to code is drafted deep within a few dozen
// passes, however long the prose went on: a reply is a hundred passes.
func TestDepthTurnsQuickly(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	prose := []float64{0.65, 0.45, 0.4, 0.4, 0.4}
	code := []float64{0.95, 0.95, 0.93, 0.93, 0.9}
	d := newDepths(5)
	// A pass of five rows is dearer than one of six here, as measured.
	cost := []float64{0, 35, 40, 48, 60, 54}
	step := func(keep []float64) int {
		depth := d.next()
		kept := 0
		for kept < depth && r.Float64() < keep[kept] {
			kept++
		}
		d.observe(depth, depth, kept, cost[depth]+r.Float64())
		return depth
	}
	for i := 0; i < 2000; i++ {
		step(prose)
	}
	deep, tokens := 0, 0
	for i := 0; i < 40; i++ {
		if step(code) == 5 {
			deep++
		}
	}
	for i := 0; i < 60; i++ {
		if n := step(code); n == 5 {
			tokens++
		}
	}
	if deep < 10 || tokens < 50 {
		t.Errorf("code after long prose: five drafts in %d of the first 40 passes and %d of the next 60 (costs %.0f, kept %.2f, tried every %d)",
			deep, tokens, d.cost[1:], d.rate[1:], d.every)
	}
	shallow := 0
	for i := 0; i < 100; i++ {
		if n := step(prose); i >= 40 && n <= 2 {
			shallow++
		}
	}
	if shallow < 50 {
		t.Errorf("prose after code: one or two drafts in only %d of passes 40 to 100 (costs %.0f, kept %.2f)", shallow, d.cost[1:], d.rate[1:])
	}
}
