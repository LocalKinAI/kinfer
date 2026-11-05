package mlxengine

import (
	"math"
	"math/rand"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/sampling"
)

// sampleRows picks one token for each row of logits [B, vocab], every row by
// its own sampling.Params, on the GPU: copying a batch's logits to the host
// would move a quarter of a million floats per row per token.
//
// The stages and their order are llama.cpp's chain as internal/sampling
// builds it, so a model samples the same way on either backend: the repeat
// penalty over the row's last RepeatLastN generated tokens, top-k, top-p over
// the probabilities at temperature 1, then temperature, then a draw. Rows
// that share their parameters are sampled together.
func sampleRows(logits mlx.Array, params []sampling.Params, history [][]int32, rngs []*rand.Rand) []int32 {
	picks := sampleGraph(logits, params, history, rngs)
	return picks.Int32s()
}

// sampleGraph is sampleRows without running anything: it returns the tokens
// as a [B] int32 array still to be evaluated, so a decode step can be
// evaluated with its sampling in one go — and the result can feed the next
// step before the host has seen it.
func sampleGraph(logits mlx.Array, params []sampling.Params, history [][]int32, rngs []*rand.Rand) mlx.Array {
	B := len(params)
	lg := mlx.Astype(logits, mlx.Float32)
	lg = penalize(lg, params, history)

	type group struct {
		p    sampling.Params
		rows []int
	}
	var groups []*group
	for r, p := range params {
		var g *group
		for _, h := range groups {
			// A seeded row draws from its own generator, so it samples alone.
			if h.p == p && p.Seed == 0 {
				g = h
				break
			}
		}
		if g == nil {
			g = &group{p: p}
			groups = append(groups, g)
		}
		g.rows = append(g.rows, r)
	}
	if len(groups) == 1 {
		return mlx.Astype(pick(lg, groups[0].p, rngs[0]), mlx.Int32)
	}

	// Sample each group, then put the tokens back in row order.
	picks := make([]mlx.Array, len(groups))
	inv := make([]int32, B)
	k := 0
	for i, g := range groups {
		idx := make([]int32, len(g.rows))
		for j, r := range g.rows {
			idx[j] = int32(r)
			inv[r] = int32(k)
			k++
		}
		sub := mlx.Take(lg, mlx.FromInt32(idx, len(idx)), 0)
		picks[i] = mlx.Astype(pick(sub, g.p, rngs[g.rows[0]]), mlx.Int32)
	}
	return mlx.Take(mlx.Concatenate(0, picks...), mlx.FromInt32(inv, B), 0)
}

// penalize applies llama.cpp's repeat penalty: a recently generated token's
// logit is divided by the penalty when positive and multiplied when negative.
func penalize(lg mlx.Array, params []sampling.Params, history [][]int32) mlx.Array {
	B := len(params)
	n := 0
	for r, p := range params {
		if on(p) {
			n = max(n, min(len(history[r]), lastN(p, len(history[r]))))
		}
	}
	if n == 0 {
		return lg
	}
	idx := make([]int32, B*n)
	pen := make([]float32, B)
	for r, p := range params {
		h := history[r]
		k := 0
		if on(p) {
			k = min(len(h), lastN(p, len(h)))
		}
		pen[r] = 1
		if k > 0 {
			pen[r] = p.RepeatPenalty
		}
		recent := h[len(h)-k:]
		for j := 0; j < n; j++ {
			// Short rows repeat an id they have (the same value written twice
			// is harmless); rows with none point at token 0 with penalty 1.
			if k > 0 {
				idx[r*n+j] = recent[j%k]
			}
		}
	}
	ids := mlx.FromInt32(idx, B, n)
	p := mlx.FromFloat32(pen, B, 1)
	got := mlx.TakeAlongAxis(lg, ids, 1)
	adj := mlx.Where(mlx.Greater(got, mlx.Scalar(0)), mlx.Divide(got, p), mlx.Multiply(got, p))
	return mlx.PutAlongAxis(lg, ids, adj, 1)
}

func on(p sampling.Params) bool {
	return p.RepeatPenalty != 1 && p.RepeatPenalty > 0 && p.RepeatLastN != 0
}

// lastN is how much history the penalty reads: RepeatLastN, or all of it for
// -1 (llama.cpp's "the whole context").
func lastN(p sampling.Params, have int) int {
	if p.RepeatLastN < 0 {
		return have
	}
	return p.RepeatLastN
}

// pick draws one token per row of lg [n, vocab] float32.
func pick(lg mlx.Array, p sampling.Params, rng *rand.Rand) mlx.Array {
	if p.Temperature <= 0 {
		return mlx.Argmax(lg, -1, false)
	}
	n, V := lg.Shape()[0], lg.Shape()[1]

	// cand [n, k] are the candidate logits and ids [n, k] their token ids;
	// without top-k every token is a candidate.
	cand, ids := lg, mlx.Array(0)
	if p.TopK > 0 && p.TopK < V {
		part := mlx.Argpartition(mlx.Negative(lg), p.TopK-1, -1)
		ids = mlx.Slice(part, []int{0, 0}, []int{n, p.TopK})
		cand = mlx.TakeAlongAxis(lg, ids, -1)
	}

	if p.TopP > 0 && p.TopP < 1 {
		// Sort descending; keep the smallest prefix whose probability (at
		// temperature 1) reaches TopP, always at least the first.
		order := mlx.Argsort(mlx.Negative(cand), -1)
		sorted := mlx.TakeAlongAxis(cand, order, -1)
		probs := mlx.Softmax(sorted, -1)
		before := mlx.Subtract(mlx.Cumsum(probs, -1, false, true), probs)
		sorted = mlx.Where(mlx.Less(before, mlx.Scalar(p.TopP)), sorted, mlx.Scalar(float32(math.Inf(-1))))
		if ids != 0 {
			ids = mlx.TakeAlongAxis(ids, order, -1)
		} else {
			ids = order
		}
		cand = sorted
	}

	cand = mlx.MulScalar(cand, 1/p.Temperature)
	choice := mlx.Reshape(mlx.Categorical(cand, -1, mlx.RandomKey(rng.Uint64())), n, 1)
	if ids == 0 {
		return mlx.Reshape(choice, n)
	}
	return mlx.Reshape(mlx.TakeAlongAxis(ids, mlx.Astype(choice, mlx.Int32), -1), n)
}
