package mlxmodel

import (
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// The ragged kernel gives what MLX's attention gives with each row's padding
// masked out, rows of every length, a chunk boundary or none.
func TestDecodeAttentionMatchesMaskedSDPA(t *testing.T) {
	lib := os.Getenv("KINFER_MLX_LIB")
	if lib == "" {
		t.Skip("KINFER_MLX_LIB not set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	const B, H, Hkv, D, capacity, end = 5, 8, 2, 256, 1300, 1200
	starts := []int32{0, 300, end - 1, 700, 1000}
	rng := rand.New(rand.NewSource(1))
	random := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	s := mlx.NewScope()
	defer s.Close()
	q := mlx.Astype(mlx.FromFloat32(random(B*H*D), B, H, 1, D), mlx.Bfloat16)
	k := mlx.Astype(mlx.FromFloat32(random(B*Hkv*capacity*D), B, Hkv, capacity, D), mlx.Bfloat16)
	v := mlx.Astype(mlx.FromFloat32(random(B*Hkv*capacity*D), B, Hkv, capacity, D), mlx.Bfloat16)
	st := mlx.FromInt32(starts, B)
	scale := float32(1 / math.Sqrt(D))

	got := mlx.Astype(decodeAttention(q, k, v, st, end, scale), mlx.Float32).Float32s()

	cols := mlx.Arange(0, end, 1, mlx.Int32)
	mask := mlx.Reshape(mlx.Greater(mlx.Reshape(cols, 1, end), mlx.AddScalar(mlx.Reshape(st, B, 1), -1)), B, 1, 1, end)
	ks := mlx.Slice(k, []int{0, 0, 0, 0}, []int{B, Hkv, end, D})
	vs := mlx.Slice(v, []int{0, 0, 0, 0}, []int{B, Hkv, end, D})
	ref := mlx.SDPA(mlx.Astype(q, mlx.Float32), mlx.Astype(ks, mlx.Float32), mlx.Astype(vs, mlx.Float32), scale, "", mask)
	want := mlx.Reshape(mlx.Transpose(ref, 0, 2, 1, 3), B, H*D).Float32s()

	worst := 0.0
	for i := range want {
		worst = math.Max(worst, math.Abs(float64(got[i]-want[i])))
	}
	// The kernel rounds its output to bf16, a part in 256.
	if worst > 0.02 {
		t.Errorf("the kernel and masked SDPA differ by up to %g", worst)
	}
	t.Logf("largest difference from masked SDPA in float32: %.4g", worst)
}

// How long a 64-row step takes with the ragged kernel and with masked SDPA:
// one row at KINFER_MLX_LONG tokens and the rest at KINFER_MLX_SHORT.
func TestDecodeAttentionStepTime(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_MLX_MODEL")
	long, _ := strconv.Atoi(os.Getenv("KINFER_MLX_LONG"))
	short, _ := strconv.Atoi(os.Getenv("KINFER_MLX_SHORT"))
	if lib == "" || dir == "" || long == 0 || short == 0 {
		t.Skip("KINFER_MLX_LIB, KINFER_MLX_MODEL, KINFER_MLX_LONG and KINFER_MLX_SHORT not all set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadQwen35(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	const B, steps = 64, 32
	b := m.NewBatch()
	defer b.Free()
	prompt := func(seed, n int) []int32 {
		p := make([]int32, n)
		for i := range p {
			p[i] = int32(1000 + ((seed+1)*7919+i*104729)%50000)
		}
		return p
	}
	for r := 0; r < B; r++ {
		n := short
		if r == 0 {
			n = long
		}
		c := m.NewCache()
		toks := prompt(r, n)
		for i := 0; i < n; i += 2048 {
			lg, err := m.Forward([][]int32{toks[i:min(i+2048, n)]}, c, nil)
			if err != nil {
				t.Fatal(err)
			}
			lg.Free()
		}
		if _, err := b.Add(c, B); err != nil {
			t.Fatal(err)
		}
	}
	for _, ragged := range []bool{true, false, true, false} {
		raggedDecode = ragged
		y := mlx.FromInt32(make([]int32, B), B, 1)
		start := time.Now()
		for i := 0; i < steps; i++ {
			s := mlx.NewScope()
			next := mlx.Reshape(mlx.Astype(mlx.Argmax(b.Step(y), -1, false), mlx.Int32), B, 1)
			s.Keep(next)
			s.Close()
			mlx.AsyncEval(append([]mlx.Array{next}, b.Arrays()...)...)
			y.Free()
			y = next
		}
		y.Int32s()
		y.Free()
		el := time.Since(start)
		t.Logf("ragged %-5v one row at %d, %d at %d: %.1f ms a step, %.0f tok/s", ragged, long, B-1, short,
			float64(el.Microseconds())/1000/steps, float64(B*steps)/el.Seconds())
	}
	raggedDecode = true
}
