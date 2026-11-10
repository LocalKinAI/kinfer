package mlx

import (
	"fmt"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/ebitengine/purego"
)

// The ops a diffusion model needs beside a transformer's: the noise it starts
// from, drawn from a key, and the tanh its gates are squashed with. Bound
// apart, so that a library without them still serves the rest.

var (
	opRandomNormal func(res *Array, shape *int32, n uintptr, dt Dtype, loc, scale float32, key Array, s Stream) int32
	opTanh         unaryFn

	imageErr error = fmt.Errorf("mlx: image ops not bound")
)

func bindImage(lib uintptr) {
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&opRandomNormal, "mlx_random_normal"}, {&opTanh, "mlx_tanh"},
	} {
		if _, err := dl.Sym(lib, b.name); err != nil {
			imageErr = fmt.Errorf("mlx-c has no %s: %w", b.name, err)
			return
		}
		purego.RegisterLibFunc(b.fn, lib, b.name)
	}
	imageErr = nil
}

// ImageOps says whether the library has what a diffusion model needs.
func ImageOps() error { return imageErr }

// RandomNormal draws standard normal samples of the given shape and dtype from
// key, as mx.random.normal(shape, dtype, key=key) does.
func RandomNormal(key Array, dt Dtype, shape ...int) Array {
	s := make([]int32, len(shape))
	for i, v := range shape {
		s[i] = int32(v)
	}
	r := arrayNew()
	var p *int32
	if len(s) > 0 {
		p = &s[0]
	}
	return out(opRandomNormal(&r, p, uintptr(len(s)), dt, 0, 1, key, gpu), r)
}

// Tanh is the hyperbolic tangent, elementwise.
func Tanh(a Array) Array { return un(opTanh, a) }
