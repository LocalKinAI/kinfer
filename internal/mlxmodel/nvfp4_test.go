package mlxmodel

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// An NVFP4 weight packed by hand — 4-bit floats eight to a uint32, lowest
// nibble first, an FP8 scale for each 16 and a scale of its own — multiplies
// and is looked up as the weight it stands for.
func TestNVFP4Linear(t *testing.T) {
	lib := os.Getenv("KINFER_MLX_LIB")
	if lib == "" {
		t.Skip("KINFER_MLX_LIB not set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	const out, in, global = 3, 32, 0.001
	fp4 := []float64{0, 0.5, 1, 1.5, 2, 3, 4, 6} // E2M1; the fourth bit is the sign
	fp8 := map[int32]float64{0x30: 0.5, 0x38: 1, 0x40: 2}
	scales := []int32{0x38, 0x40, 0x30, 0x38, 0x40, 0x30}
	rng := rand.New(rand.NewSource(1))
	want := make([]float64, out*in)
	packed := make([]int32, out*in/8)
	for r := 0; r < out; r++ {
		for c := 0; c < in; c++ {
			code := rng.Intn(16)
			if c%8 == 7 {
				code %= 8 // the top nibble positive, so the word fits an int32
			}
			v := fp4[code%8]
			if code >= 8 {
				v = -v
			}
			want[r*in+c] = v * fp8[scales[r*in/16+c/16]] * global
			packed[(r*in+c)/8] |= int32(code) << (4 * (c % 8))
		}
	}
	s := mlx.NewScope()
	defer s.Close()
	l := Linear{
		W:      mlx.Astype(mlx.FromInt32(packed, out, in/8), mlx.Uint32),
		Scales: mlx.Astype(mlx.FromInt32(scales, out, in/16), mlx.Uint8),
		Group:  16, Bits: 4, Mode: "nvfp4", Global: global,
	}

	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	y := linear(mlx.FromFloat32(x, 1, in), l).Float32s()
	for r := 0; r < out; r++ {
		var w float64
		for c := 0; c < in; c++ {
			w += float64(x[c]) * want[r*in+c]
		}
		if math.Abs(float64(y[r])-w) > 1e-5 {
			t.Errorf("row %d: x·w = %g, want %g", r, y[r], w)
		}
	}

	// Looked up, a row comes out as bfloat16, like any embedding's.
	got := mlx.Astype(l.rows(mlx.FromInt32([]int32{2, 0}, 1, 2)), mlx.Float32).Float32s()
	for i, r := range []int{2, 0} {
		for c := 0; c < in; c++ {
			if g, w := float64(got[i*in+c]), want[r*in+c]; math.Abs(g-w) > math.Abs(w)/256 {
				t.Fatalf("row %d, column %d looked up as %g, want %g", r, c, g, w)
			}
		}
	}
}
