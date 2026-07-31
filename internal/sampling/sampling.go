// Package sampling picks the next token from a model's logits.
//
// It exists because gollama.cpp binds exactly one sampler — greedy — and greedy
// decoding degenerates. Observed on Qwen2.5-0.5B: "财是人的命根，是人的命脉，
// 是人的命关。" repeated until the token budget ran out. Always taking the
// highest-probability token means that once the model enters a loop, nothing can
// break it out. A usable runtime needs temperature, nucleus sampling, and a
// repetition penalty, so kinfer implements them over the raw logits.
//
// The pipeline order matches llama.cpp's default:
//
//	repetition penalty → temperature → top-k → top-p → softmax → draw
package sampling

import (
	"math"
	"math/rand"
	"sort"
)

// Candidate is one token under consideration.
type Candidate struct {
	ID    int32
	Logit float32
}

// Params configures a Sampler. The zero value is not useful; start from
// DefaultParams.
type Params struct {
	// Temperature flattens (>1) or sharpens (<1) the distribution.
	// Zero means greedy: always take the most likely token.
	Temperature float32

	// TopK keeps only the K most likely tokens. Zero disables.
	TopK int

	// TopP (nucleus) keeps the smallest set of tokens whose probabilities sum
	// to P. Values >= 1 disable it.
	TopP float32

	// RepeatPenalty divides the logit of recently seen tokens. 1.0 disables.
	// Around 1.1 is the usual choice — high enough to break loops, low enough
	// that natural repetition ("the … the") still survives.
	RepeatPenalty float32

	// RepeatLastN is how many recent tokens the penalty considers.
	RepeatLastN int

	// Seed makes a run reproducible. Zero picks a random seed.
	Seed int64
}

// DefaultParams are conservative settings that produce coherent prose without
// the loops greedy decoding falls into.
func DefaultParams() Params {
	return Params{
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.95,
		RepeatPenalty: 1.1,
		RepeatLastN:   64,
	}
}

// Sampler draws tokens and remembers what it has drawn, so the repetition
// penalty has history to work with. It is not safe for concurrent use; give
// each generation its own.
type Sampler struct {
	params Params
	rng    *rand.Rand
	recent []int32
}

// New creates a Sampler.
func New(p Params) *Sampler {
	seed := p.Seed
	if seed == 0 {
		seed = rand.Int63()
	}
	return &Sampler{
		params: p,
		rng:    rand.New(rand.NewSource(seed)),
		recent: make([]int32, 0, max(p.RepeatLastN, 1)),
	}
}

// Sample returns the chosen token id. It mutates cands in place (sorting and
// rescaling), so pass a slice the caller does not need afterwards.
func (s *Sampler) Sample(cands []Candidate) int32 {
	if len(cands) == 0 {
		return -1
	}

	s.applyRepeatPenalty(cands)

	// Temperature 0 means greedy — no distribution to draw from.
	if s.params.Temperature <= 0 {
		best := 0
		for i := 1; i < len(cands); i++ {
			if cands[i].Logit > cands[best].Logit {
				best = i
			}
		}
		return s.remember(cands[best].ID)
	}

	for i := range cands {
		cands[i].Logit /= s.params.Temperature
	}

	// Sort once, descending. Both top-k and top-p need ordered candidates, and
	// softmax below relies on cands[0] being the maximum for numeric stability.
	sort.Slice(cands, func(i, j int) bool { return cands[i].Logit > cands[j].Logit })

	if k := s.params.TopK; k > 0 && k < len(cands) {
		cands = cands[:k]
	}

	probs := softmax(cands)

	if p := s.params.TopP; p > 0 && p < 1 {
		cands, probs = nucleus(cands, probs, p)
	}

	return s.remember(cands[s.draw(probs)].ID)
}

// applyRepeatPenalty pushes down the logits of tokens seen recently.
//
// Positive and negative logits are handled differently on purpose: dividing a
// negative logit would make it LARGER (less negative), rewarding repetition
// instead of punishing it. This asymmetry is what llama.cpp does.
func (s *Sampler) applyRepeatPenalty(cands []Candidate) {
	penalty := s.params.RepeatPenalty
	if penalty == 1 || penalty <= 0 || len(s.recent) == 0 {
		return
	}

	seen := make(map[int32]struct{}, len(s.recent))
	for _, id := range s.recent {
		seen[id] = struct{}{}
	}

	for i := range cands {
		if _, ok := seen[cands[i].ID]; !ok {
			continue
		}
		if cands[i].Logit > 0 {
			cands[i].Logit /= penalty
		} else {
			cands[i].Logit *= penalty
		}
	}
}

// softmax converts logits to probabilities. cands must be sorted descending so
// that subtracting the first element keeps exp() from overflowing.
func softmax(cands []Candidate) []float64 {
	maxLogit := float64(cands[0].Logit)

	probs := make([]float64, len(cands))
	var sum float64
	for i, c := range cands {
		e := math.Exp(float64(c.Logit) - maxLogit)
		probs[i] = e
		sum += e
	}
	if sum == 0 {
		// Degenerate distribution; fall back to uniform rather than dividing by
		// zero and producing NaNs.
		for i := range probs {
			probs[i] = 1 / float64(len(probs))
		}
		return probs
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

// nucleus keeps the shortest prefix whose probabilities reach p.
func nucleus(cands []Candidate, probs []float64, p float32) ([]Candidate, []float64) {
	var cum float64
	for i, pr := range probs {
		cum += pr
		if cum >= float64(p) {
			return cands[:i+1], probs[:i+1]
		}
	}
	return cands, probs
}

// draw picks an index with probability proportional to probs. probs need not
// sum to exactly 1 — after nucleus truncation it sums to slightly more than p.
func (s *Sampler) draw(probs []float64) int {
	var total float64
	for _, p := range probs {
		total += p
	}

	r := s.rng.Float64() * total
	var cum float64
	for i, p := range probs {
		cum += p
		if r < cum {
			return i
		}
	}
	return len(probs) - 1 // float rounding
}

// remember records a token for the repetition penalty and returns it unchanged,
// so callers can write `return s.remember(id)`.
func (s *Sampler) remember(id int32) int32 {
	n := s.params.RepeatLastN
	if n <= 0 {
		return id
	}
	s.recent = append(s.recent, id)
	if len(s.recent) > n {
		s.recent = s.recent[len(s.recent)-n:]
	}
	return id
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
