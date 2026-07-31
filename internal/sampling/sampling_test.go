package sampling

import (
	"math"
	"testing"
)

func cands(logits ...float32) []Candidate {
	out := make([]Candidate, len(logits))
	for i, l := range logits {
		out[i] = Candidate{ID: int32(i), Logit: l}
	}
	return out
}

// Temperature 0 must be deterministic greedy.
func TestSample_GreedyAtZeroTemperature(t *testing.T) {
	s := New(Params{Temperature: 0})
	for i := 0; i < 20; i++ {
		if got := s.Sample(cands(1.0, 5.0, 2.0)); got != 1 {
			t.Fatalf("greedy picked %d, want 1 (highest logit)", got)
		}
	}
}

// The whole reason this package exists: a loop must be breakable.
func TestRepeatPenalty_BreaksALoop(t *testing.T) {
	p := DefaultParams()
	p.Temperature = 0 // greedy, so only the penalty can change the outcome
	p.RepeatPenalty = 1.5
	p.RepeatLastN = 8
	s := New(p)

	// Token 0 dominates. Greedy alone would return it forever.
	first := s.Sample(cands(10.0, 9.0, 8.0))
	if first != 0 {
		t.Fatalf("first pick = %d, want 0", first)
	}

	// After penalising, 10.0/1.5 = 6.67 falls below 9.0.
	second := s.Sample(cands(10.0, 9.0, 8.0))
	if second == 0 {
		t.Error("repeat penalty failed to break out of the loop")
	}
}

// Dividing a negative logit would make it larger, rewarding repetition.
func TestRepeatPenalty_NegativeLogitsGetSmaller(t *testing.T) {
	p := Params{Temperature: 0, RepeatPenalty: 2.0, RepeatLastN: 4}
	s := New(p)
	s.remember(0)

	c := cands(-1.0, -2.0)
	s.applyRepeatPenalty(c)

	if c[0].Logit != -2.0 {
		t.Errorf("penalised negative logit = %v, want -2.0 (multiplied, not divided)", c[0].Logit)
	}
	if c[1].Logit != -2.0 {
		t.Errorf("unseen token was modified: %v", c[1].Logit)
	}
}

// top-k must confine the draw to the k best tokens.
func TestTopK_LimitsChoices(t *testing.T) {
	p := Params{Temperature: 1.0, TopK: 2, RepeatPenalty: 1}
	s := New(p)

	seen := map[int32]bool{}
	for i := 0; i < 300; i++ {
		seen[s.Sample(cands(5.0, 4.9, 0.1, 0.05))] = true
	}
	if seen[2] || seen[3] {
		t.Errorf("top-k=2 still drew outside the top 2: %v", seen)
	}
	if !seen[0] || !seen[1] {
		t.Errorf("top-k=2 did not use both allowed tokens: %v", seen)
	}
}

// A dominant token should make nucleus sampling collapse to just that token.
func TestTopP_CollapsesOnDominantToken(t *testing.T) {
	p := Params{Temperature: 1.0, TopP: 0.9, RepeatPenalty: 1}
	s := New(p)

	for i := 0; i < 200; i++ {
		if got := s.Sample(cands(20.0, 0.0, 0.0)); got != 0 {
			t.Fatalf("nucleus drew %d despite a dominant token", got)
		}
	}
}

func TestSoftmax_SumsToOne(t *testing.T) {
	probs := softmax(cands(3.0, 2.0, 1.0))
	var sum float64
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("softmax sums to %v, want 1", sum)
	}
	if !(probs[0] > probs[1] && probs[1] > probs[2]) {
		t.Errorf("softmax lost ordering: %v", probs)
	}
}

// Large logits must not overflow to Inf/NaN.
func TestSoftmax_NumericStability(t *testing.T) {
	probs := softmax(cands(1000.0, 999.0))
	for i, p := range probs {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			t.Fatalf("probs[%d] = %v on large logits", i, p)
		}
	}
}

// Same seed, same output — required for reproducible tests and bug reports.
func TestSample_SeedIsReproducible(t *testing.T) {
	run := func() []int32 {
		p := DefaultParams()
		p.Seed = 42
		s := New(p)
		var out []int32
		for i := 0; i < 30; i++ {
			out = append(out, s.Sample(cands(3.0, 2.5, 2.0, 1.0)))
		}
		return out
	}

	a, b := run(), run()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed diverged at %d: %d vs %d", i, a[i], b[i])
		}
	}
}

func TestSample_EmptyCandidates(t *testing.T) {
	if got := New(DefaultParams()).Sample(nil); got != -1 {
		t.Errorf("Sample(nil) = %d, want -1", got)
	}
}

// The recent-token window must stay bounded, or memory grows without limit on
// long generations.
func TestRecentWindowIsBounded(t *testing.T) {
	p := DefaultParams()
	p.RepeatLastN = 5
	s := New(p)
	for i := 0; i < 100; i++ {
		s.remember(int32(i))
	}
	if len(s.recent) != 5 {
		t.Errorf("recent window = %d, want 5", len(s.recent))
	}
	if s.recent[4] != 99 {
		t.Errorf("window lost the newest token: %v", s.recent)
	}
}
