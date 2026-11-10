package mlx

import (
	"fmt"
	"math"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/ebitengine/purego"
)

// The ops an audio encoder needs beside a transformer's — a 2-D convolution,
// a layer norm with a bias, the exact GELU's erf — and a spectrogram's: a real
// FFT and log10. Bound apart from bindOps so that a library without them still
// serves text; Conv2d and the rest then report it.

var (
	opConv2d    func(res *Array, in, w Array, s0, s1, p0, p1, d0, d1, groups int32, s Stream) int32
	opLayerNorm func(res *Array, x, w, b Array, eps float32, s Stream) int32
	opErf       unaryFn
	opLog10     unaryFn
	opReal      unaryFn
	opImag      unaryFn
	opRFFT      func(res *Array, a Array, n, axis, norm int32, s Stream) int32 // norm: mlx_fft_norm

	// A vocoder's: the snake activation's sine, rotary tables' cosine, the
	// upsampling transposed convolution — and the global seed, which sampling
	// with no key draws from, as mlx-audio's does.
	opSin             unaryFn
	opCos             unaryFn
	opConvTranspose1d func(res *Array, in, w Array, stride, padding, dilation, outputPadding, groups int32, s Stream) int32
	opRandomSeed      func(seed uint64) int32
	opAddmm           func(res *Array, c, a, b Array, alpha, beta float32, s Stream) int32

	audioErr error = fmt.Errorf("mlx: audio ops not bound")
)

func bindAudio(lib uintptr) {
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&opConv2d, "mlx_conv2d"}, {&opLayerNorm, "mlx_fast_layer_norm"}, {&opErf, "mlx_erf"},
		{&opLog10, "mlx_log10"}, {&opRFFT, "mlx_fft_rfft"}, {&opReal, "mlx_real"}, {&opImag, "mlx_imag"},
		{&opSin, "mlx_sin"}, {&opCos, "mlx_cos"}, {&opConvTranspose1d, "mlx_conv_transpose1d"},
		{&opRandomSeed, "mlx_random_seed"}, {&opAddmm, "mlx_addmm"},
	} {
		if _, err := dl.Sym(lib, b.name); err != nil {
			audioErr = fmt.Errorf("mlx-c has no %s: %w", b.name, err)
			return
		}
		purego.RegisterLibFunc(b.fn, lib, b.name)
	}
	audioErr = nil
}

// AudioOps says whether the library has what an audio encoder needs.
func AudioOps() error { return audioErr }

// Conv2d convolves in [N, H, W, C] with w [Cout, KH, KW, C/groups], as
// mx.conv2d, the same stride and padding on both axes.
func Conv2d(in, w Array, stride, padding int) Array {
	r := arrayNew()
	s, p := int32(stride), int32(padding)
	return out(opConv2d(&r, in, w, s, s, p, p, 1, 1, 1, gpu), r)
}

// LayerNorm normalizes the last axis, then scales by w and shifts by b; either
// may be 0.
func LayerNorm(x, w, b Array, eps float32) Array {
	r := arrayNew()
	return out(opLayerNorm(&r, x, w, b, eps, gpu), r)
}

// Erf is the error function, elementwise.
func Erf(a Array) Array { return un(opErf, a) }

// Log10 is the base-10 logarithm, elementwise.
func Log10(a Array) Array { return un(opLog10, a) }

// GELU is the exact GELU, x·½(1 + erf(x/√2)), as mlx.nn.gelu computes it.
func GELU(x Array) Array {
	half := MulScalar(x, 0.5)
	return Multiply(half, AddScalar(Erf(MulScalar(x, float32(1/math.Sqrt2))), 1))
}

// Power2 is |z|², elementwise, of a complex array, as float32.
func Power2(z Array) Array {
	re, im := un(opReal, z), un(opImag, z)
	return Add(Multiply(re, re), Multiply(im, im))
}

// Sin and Cos are elementwise.
func Sin(a Array) Array { return un(opSin, a) }
func Cos(a Array) Array { return un(opCos, a) }

// ConvTranspose1d is mx.conv_transpose1d: in [N, L, Cin], w [Cout, K, Cin].
func ConvTranspose1d(in, w Array, stride int) Array {
	r := arrayNew()
	return out(opConvTranspose1d(&r, in, w, int32(stride), 0, 1, 0, 1, gpu), r)
}

// Addmm is c + a·b in one kernel, as mlx.nn.Linear adds its bias.
func Addmm(c, a, b Array) Array {
	r := arrayNew()
	return out(opAddmm(&r, c, a, b, 1, 1, gpu), r)
}

// Seed seeds the global key that Categorical draws from when it is given
// none, as mx.random.seed does.
func Seed(seed uint64) {
	check(opRandomSeed(seed))
}

// RFFT is the real FFT of size n along axis, complex64.
func RFFT(a Array, n, axis int) Array {
	r := arrayNew()
	const backward = 0 // MLX_FFT_NORM_BACKWARD: no scaling, as numpy's default
	return out(opRFFT(&r, a, int32(n), int32(axis), backward, gpu), r)
}
