package mlxembed

import (
	"fmt"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// What the picture and sound towers share: their projections, which the
// release keeps in bfloat16 with clamps beside some of them, and the
// placement of a tower's output in the text the model reads.

// tlinear is one tower projection: x @ w.T + b, its input and output clamped
// where the checkpoint gives bounds (Gemma 4's clipped linears).
type tlinear struct {
	w, b                         mlx.Array
	inMin, inMax, outMin, outMax mlx.Array
}

func (l tlinear) apply(x mlx.Array) mlx.Array {
	if l.inMin != 0 {
		x = clip(x, l.inMin, l.inMax)
	}
	y := mlx.Matmul(x, mlx.T(l.w))
	if l.b != 0 {
		y = mlx.Add(y, l.b)
	}
	if l.outMin != 0 {
		y = clip(y, l.outMin, l.outMax)
	}
	return y
}

func clip(x, lo, hi mlx.Array) mlx.Array { return mlx.Minimum(mlx.Maximum(x, lo), hi) }

// towerWeights reads a tower's tensors by name, remembering the first one
// missing so a loader can go on and report it once.
type towerWeights struct {
	t       map[string]mlx.Array
	missing string
}

func (w *towerWeights) arr(name string) mlx.Array {
	a, ok := w.t[name]
	if !ok && w.missing == "" {
		w.missing = name
	}
	return a
}

// linear reads name's weight — under a .linear infix, as the clipped ones
// keep it, or not — its bias if it has one, and all four clamps or none.
func (w *towerWeights) linear(name string) tlinear {
	var l tlinear
	for _, k := range []string{name + ".linear.weight", name + ".weight"} {
		if a, ok := w.t[k]; ok {
			l.w = a
			break
		}
	}
	if l.w == 0 {
		if w.missing == "" {
			w.missing = name + ".weight"
		}
		return l
	}
	for _, k := range []string{name + ".linear.bias", name + ".bias"} {
		if a, ok := w.t[k]; ok {
			l.b = a
			break
		}
	}
	l.inMin, l.inMax = w.t[name+".input_min"], w.t[name+".input_max"]
	l.outMin, l.outMax = w.t[name+".output_min"], w.t[name+".output_max"]
	if n := count(l.inMin, l.inMax, l.outMin, l.outMax); n != 0 && n != 4 {
		panic(&mlx.Error{Msg: fmt.Sprintf("%s has %d of the four clamps", name, n)})
	}
	return l
}

func count(as ...mlx.Array) int {
	n := 0
	for _, a := range as {
		if a != 0 {
			n++
		}
	}
	return n
}

// pad adds lo zeros before and hi after x along axis.
func pad(x mlx.Array, axis, lo, hi int) mlx.Array {
	if lo == 0 && hi == 0 {
		return x
	}
	parts := []mlx.Array{}
	zeros := func(n int) mlx.Array {
		sh := append([]int(nil), x.Shape()...)
		sh[axis] = n
		return mlx.Zeros(x.Dtype(), sh...)
	}
	if lo > 0 {
		parts = append(parts, zeros(lo))
	}
	parts = append(parts, x)
	if hi > 0 {
		parts = append(parts, zeros(hi))
	}
	return mlx.Concatenate(axis, parts...)
}

// media is one picture or sound in an input, prepared: the token run it
// takes in the text (begin marker, soft tokens, end marker) and how to make
// the features that stand in its soft tokens.
type media struct {
	tokens []int32
	// soft are the positions in tokens, relative to the run's start, of each
	// piece's soft tokens, and encode makes that piece's [n, hidden] features.
	pieces []piece
}

type piece struct {
	at, n  int
	encode func() mlx.Array
}
