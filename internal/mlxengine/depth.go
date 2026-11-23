package mlxengine

import "sort"

// depths chooses how many tokens a reply writing alone drafts in a pass.
//
// More drafts are more tokens when the model agrees with them and a dearer
// pass whether it does or not, and how often it agrees is the text's doing:
// code and worked arithmetic are agreed with five and six drafts deep, prose
// rarely past the second. So the number is the one that writes the most
// tokens a millisecond by what has been measured: what a pass of each depth
// costs here, and how often the draft at each place is kept when those
// before it were — both smoothed, so they follow the text as it changes.
// (Ollama's MLX runner chooses its depth the same way; this is not its
// code.)
//
// The first passes of a model's life try every depth in turn, three passes
// each, for the costs. After that the depth one deeper than the best is
// tried every few passes — a rate nobody measures again is never corrected,
// and text turns from prose to code without notice — and less and less
// often while trying it changes nothing. A draft's rate that has not been
// measured for a while is no longer believed: the place takes the rate of
// the one before it, so a depth given up on while the text was prose is not
// held against code. An engine keeps one for as long as its model is loaded.
type depths struct {
	max  int
	cost []float64 // [d] ms a pass of d drafts; 0 before it is measured
	rate []float64 // [i] the i-th draft kept, of the passes that kept those before it; i from 1
	seen []int     // [i] passes that got as far as the i-th draft
	at   []int     // [i] the pass that last did
	pass int       // passes observed

	cur   int // the depth in use
	hold  int // passes still to run at cur before choosing again
	seed  int // depths still to measure for the first time
	fixed int // when set, the depth of every pass (-draft N)

	chosen int // the best depth when it was last worked out
	since  int // passes since a deeper one was last tried
	every  int // passes between such tries

	// vars is how many of a pass's outcomes are recorded ahead (solo).
	vars int
}

const (
	costAlpha  = 0.3  // weight of a pass's time in its depth's cost
	costClamp  = 0.25 // most one pass moves the cost, either way
	rateAlpha  = 0.1  // weight of one pass in a draft's rate
	rateTrust  = 5    // passes before a draft's own rate is believed
	rateStale  = 48   // passes after which it no longer is
	ratePrior  = 0.6  // the first draft's rate before any is measured
	holdPasses = 3    // passes each depth is first tried for
	tryFirst   = 4    // passes between tries of a deeper depth, at first
	tryLast    = 256  // and at most
)

func newDepths(max int) *depths {
	return &depths{max: max, cost: make([]float64, max+1), rate: make([]float64, max+1), seen: make([]int, max+1),
		at: make([]int, max+1), seed: max, every: tryFirst, vars: 2}
}

// next is the number of drafts for the pass about to be recorded.
func (d *depths) next() int {
	if d.fixed > 0 {
		d.cur = d.fixed
		return d.fixed
	}
	if d.hold > 0 {
		d.hold--
		return d.cur
	}
	// Every depth once, deepest first: its passes also say how often each
	// place's draft is kept.
	if d.seed > 0 {
		d.cur, d.hold = d.seed, holdPasses-1
		d.seed--
		return d.cur
	}
	best := d.best()
	if best != d.chosen {
		d.chosen, d.since, d.every = best, 0, tryFirst
	}
	d.cur = best
	if d.since++; d.since >= d.every && best < d.max {
		d.since = 0
		d.every = min(2*d.every, tryLast)
		d.cur = best + 1
	}
	return d.cur
}

// best is the depth that writes the most tokens a millisecond.
func (d *depths) best() int {
	best, most := d.max, 0.0
	for n := 1; n <= d.max; n++ {
		if d.cost[n] <= 0 {
			continue
		}
		if v := d.expected(n) / d.cost[n]; v > most {
			best, most = n, v
		}
	}
	return best
}

// kept is how often the i-th draft is kept when those before it were: its
// own rate once it has been seen often enough, till then the rate of the
// draft before it.
func (d *depths) kept(i int) float64 {
	for ; i >= 1; i-- {
		if d.seen[i] >= rateTrust && d.pass-d.at[i] <= rateStale {
			return d.rate[i]
		}
	}
	return ratePrior
}

// expected is the tokens a pass of n drafts writes: the reply's own, and
// each draft as often as it and all before it are kept.
func (d *depths) expected(n int) float64 {
	total, all := 1.0, 1.0
	for i := 1; i <= n; i++ {
		all *= d.kept(i)
		total += all
	}
	return total
}

// observe takes in a pass: n drafts asked for, drafted checked, kept of them
// kept, ms from the pass before it being read to its own.
func (d *depths) observe(n, drafted, kept int, ms float64) {
	d.pass++
	if n >= 0 && n <= d.max && ms > 0 {
		switch c := d.cost[n]; {
		case c <= 0:
			d.cost[n] = ms
		case d.fixed == 0 && d.pass <= d.max*holdPasses+1:
			// A depth's first passes: the first of them also makes the
			// kernels and buffers of its shape, so the quickest counts.
			d.cost[n] = min(c, ms)
		default:
			d.cost[n] = c + costAlpha*max(-costClamp*c, min(costClamp*c, ms-c))
		}
	}
	for i := 1; i <= drafted && i <= d.max && kept >= i-1; i++ {
		got := 0.0
		if kept >= i {
			got = 1
		}
		if d.seen[i] == 0 || d.pass-d.at[i] > rateStale {
			d.rate[i] = got
			d.seen[i] = 0
		} else {
			d.rate[i] += rateAlpha * (got - d.rate[i])
		}
		d.seen[i]++
		d.at[i] = d.pass
	}
}

// likely is the numbers of drafts a pass of n may keep, the likeliest
// first: 0 to n.
func (d *depths) likely(n int) []int {
	p := make([]float64, n+1)
	all := 1.0
	for i := 0; i < n; i++ {
		k := d.kept(i + 1)
		p[i] = all * (1 - k)
		all *= k
	}
	p[n] = all
	order := make([]int, n+1)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return p[order[a]] > p[order[b]] })
	return order
}
