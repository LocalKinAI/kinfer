package mlxtts

import (
	"math"
	"sync"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// Sampling as mlx-audio's _sample_token does it, step for step: the special
// codes suppressed, a repetition penalty over the last 64 first codes, the
// temperature, the top k, then a draw with the speech's next key.

// recentCodes is how far back the repetition penalty looks.
const recentCodes = 64

// sampling is how one kind of code is drawn, with the scalars it needs made
// once: newSampling's arrays belong to the scope it is called in.
type sampling struct {
	Temperature float32 // 0: the most likely code
	TopK        int
	Penalty     float32 // on the first code's recent values; 1 for none
	EOS         int32   // the talker's end, the one special code drawn

	V             int
	negInf, temp  mlx.Array
	pen, eos      mlx.Array
	penaltyLaunch *mlx.Launch
	keys          *keys // the draws' keys, shared with the other codes'
}

// keys is a sequence of PRNG keys of the speech's own, as MLX's global one
// goes on — split the key, draw with the second half, keep the first — so
// that a seed gives the same draws mx.random.seed gives mlx-audio, whichever
// thread the Go code runs on: MLX keeps its global key for each thread.
type keys struct {
	key   mlx.Array
	owned bool // key was carried out of a frame's scope, and is ours to free
}

func newKeys(seed uint64) *keys { return &keys{key: mlx.RandomKey(seed)} }

func (k *keys) next() mlx.Array {
	a, b := mlx.SplitKey(k.key)
	k.key = a
	return b
}

// carry keeps the key a frame's draws ended on past its scope, freeing the
// one the frame started from if that was carried in.
func (k *keys) carry(sc *mlx.Scope, before mlx.Array, beforeOwned bool) {
	if k.key == before {
		return // nothing drawn
	}
	sc.Escape(k.key)
	if beforeOwned {
		before.Free()
	}
	k.owned = true
}

func (k *keys) free() {
	if k.owned {
		k.key.Free()
		k.owned = false
	}
}

func newSampling(temperature float32, topK int, penalty float32, eos int32, V int, dt mlx.Dtype) *sampling {
	return &sampling{Temperature: temperature, TopK: topK, Penalty: penalty, EOS: eos, V: V,
		negInf: mlx.ScalarAs(float32(math.Inf(-1)), dt), temp: mlx.ScalarAs(temperature, dt),
		pen: mlx.ScalarAs(penalty, dt), eos: mlx.Int(int(eos))}
}

func (s *sampling) free() {
	if s.penaltyLaunch != nil {
		s.penaltyLaunch.Free()
		s.penaltyLaunch = nil
	}
}

// draw draws a code from logits [1, V]: [1, 1] uint32.
func (s *sampling) draw(logits mlx.Array) mlx.Array {
	if s.Temperature <= 0 {
		return mlx.Reshape(mlx.Argmax(logits, -1, false), 1, 1)
	}
	if s.Temperature != 1 {
		logits = mlx.Divide(logits, s.temp)
	}
	if s.TopK > 0 && s.TopK < s.V {
		drop := mlx.Slice(mlx.Argpartition(mlx.Negative(logits), s.TopK-1, -1), []int{0, s.TopK}, []int{1, s.V})
		logits = mlx.PutAlongAxis(logits, drop, s.negInf, -1)
	}
	return mlx.Reshape(mlx.Categorical(logits, -1, s.keys.next()), 1, 1)
}

// penaltySource is the first code's suppression and repetition penalty in
// one kernel, as mlx-audio's _sample_token does them with put_along_axis:
// the top 1,024 codes but the end are -inf, and a code among the recent ones
// (window, -1 where there are none yet) is multiplied by the penalty if it
// is below 0 and divided by it if not, each rounded to T as the ops round.
// The recent codes never reach the suppressed ones, so the order is moot.
const penaltySource = `
	uint v = thread_position_in_grid.x;
	T l = logits[v];
	bool seen = false;
	for (int j = 0; j < W; j++) {
		seen = seen || window[j] == int(v);
	}
	T y;
	if (int(v) >= V - 1024 && int(v) != eos) {
		y = T(-INFINITY);
	} else if (seen) {
		y = float(l) < 0.0f ? T(float(l) * float(penalty)) : T(float(l) / float(penalty));
	} else {
		y = l;
	}
	out[v] = y;
`

var (
	penaltyOnce   sync.Once
	penaltyKernel *mlx.Kernel
)

// penalize is logits [1, V] suppressed and penalized for window's codes
// ([recentCodes] int32).
func (s *sampling) penalize(logits, window mlx.Array) mlx.Array {
	penaltyOnce.Do(func() {
		penaltyKernel = mlx.NewKernel("kinfer_tts_penalty", []string{"logits", "window", "penalty", "eos"}, []string{"out"}, penaltySource)
	})
	if s.penaltyLaunch == nil {
		dt := logits.Dtype()
		s.penaltyLaunch = penaltyKernel.Prepare([]mlx.KernelOutput{{Shape: []int{1, s.V}, Dtype: dt}},
			[]mlx.KernelTemplate{{Name: "T", Dtype: dt}, {Name: "V", Int: s.V, IsInt: true}, {Name: "W", Int: recentCodes, IsInt: true}},
			[3]int{s.V, 1, 1}, [3]int{min(256, s.V), 1, 1})
	}
	return s.penaltyLaunch.Run(logits, window, s.pen, s.eos)[0]
}
