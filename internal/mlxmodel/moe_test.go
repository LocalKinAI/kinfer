package mlxmodel

import (
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

func openMLX(t *testing.T) {
	lib := os.Getenv("KINFER_MLX_LIB")
	if lib == "" {
		t.Skip("KINFER_MLX_LIB not set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
}

func randomArray(rng *rand.Rand, scale float64, shape ...int) mlx.Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64() * scale)
	}
	return mlx.FromFloat32(v, shape...)
}

// The experts' block gives, for each token, what the router's top k experts
// and the shared expert give it one at a time, whether the token-expert pairs
// are few enough to be gathered as they come or so many they are sorted.
func TestMoEBlockMatchesExpertByExpert(t *testing.T) {
	openMLX(t)
	const E, k, D, I = 8, 2, 32, 16
	rng := rand.New(rand.NewSource(3))
	s := mlx.NewScope()
	defer s.Close()
	e := &moeBlock{
		router: Linear{W: randomArray(rng, 1, E, D)}, sharedGate: Linear{W: randomArray(rng, 0.2, 1, D)},
		gate: Linear{W: randomArray(rng, 0.2, E, I, D)}, up: Linear{W: randomArray(rng, 0.2, E, I, D)},
		down:  Linear{W: randomArray(rng, 0.2, E, D, I)},
		sGate: Linear{W: randomArray(rng, 0.2, I, D)}, sUp: Linear{W: randomArray(rng, 0.2, I, D)},
		sDown: Linear{W: randomArray(rng, 0.2, D, I)},
		k:     k, experts: E, normed: true,
	}
	// The reference is plain Go in float64, from the weights' values.
	vals := func(l Linear) []float32 { return l.W.Float32s() }
	router, sg := vals(e.router), vals(e.sharedGate)
	eg, eu, ed := vals(e.gate), vals(e.up), vals(e.down)
	sgw, suw, sdw := vals(e.sGate), vals(e.sUp), vals(e.sDown)
	dot := func(w []float32, row, n int, x []float64) float64 {
		var t float64
		for i := 0; i < n; i++ {
			t += float64(w[row*n+i]) * x[i]
		}
		return t
	}
	glu := func(g, u, d []float32, x []float64) []float64 { // g, u [I, D], d [D, I]
		h := make([]float64, I)
		for i := range h {
			a := dot(g, i, D, x)
			h[i] = a / (1 + math.Exp(-a)) * dot(u, i, D, x)
		}
		y := make([]float64, D)
		for o := range y {
			y[o] = dot(d, o, I, h)
		}
		return y
	}
	for _, L := range []int{3, 40} { // 6 pairs, then 80: past sortAt
		var worst, scale float64
		x := randomArray(rng, 1, 1, L, D)
		got, xs := e.forward(x).Float32s(), x.Float32s()
		for n := 0; n < L; n++ {
			xn := make([]float64, D)
			for i := range xn {
				xn[i] = float64(xs[n*D+i])
			}
			p, top := make([]float64, E), 0.0
			for j := range p {
				p[j] = dot(router, j, D, xn)
				top = math.Max(top, p[j])
			}
			order := make([]int, E)
			for j := range p {
				p[j], order[j] = math.Exp(p[j]-top), j
			}
			sort.Slice(order, func(a, b int) bool { return p[order[a]] > p[order[b]] })
			sum := 0.0
			for _, j := range order[:k] {
				sum += p[j]
			}
			want := glu(sgw, suw, sdw, xn)
			gate := 1 / (1 + math.Exp(-dot(sg, 0, D, xn)))
			for o := range want {
				want[o] *= gate
			}
			for _, j := range order[:k] {
				y := glu(eg[j*I*D:(j+1)*I*D], eu[j*I*D:(j+1)*I*D], ed[j*D*I:(j+1)*D*I], xn)
				for o := range want {
					want[o] += p[j] / sum * y[o]
				}
			}
			for o := range want {
				worst = math.Max(worst, math.Abs(float64(got[n*D+o])-want[o]))
				scale = math.Max(scale, math.Abs(want[o]))
			}
		}
		// The block's own matmuls, the router's and the shared expert's, are
		// Metal's float32 ones, which on an M5 Max round to about one part in
		// 1,400 (0.0019 of 2.75 for a 32-wide dot product); the gathered
		// experts' are exact to 1e-7. A wrong expert or weight is off by
		// the size of the output.
		t.Logf("%d tokens: largest difference %.2g of outputs up to %.2g", L, worst, scale)
		if worst > 5e-3*scale {
			t.Errorf("%d tokens: the block is %g from expert by expert", L, worst)
		}
	}
}

// GatherQMM multiplies each row by the quantized weight it names, as
// QuantizedMatmul does with that weight alone: the binding passes MLX its
// arguments where it wants them.
func TestGatherQMMIsQuantizedMatmulOfEachExpert(t *testing.T) {
	openMLX(t)
	const E, out, in, group = 4, 8, 64, 32
	rng := rand.New(rand.NewSource(4))
	s := mlx.NewScope()
	defer s.Close()
	packed := make([]int32, E*out*in/8)
	for i := range packed {
		packed[i] = rng.Int31()
	}
	w := mlx.Astype(mlx.FromInt32(packed, E, out, in/8), mlx.Uint32)
	scales := randomArray(rng, 0.01, E, out, in/group)
	biases := randomArray(rng, 0.01, E, out, in/group)
	x := randomArray(rng, 1, 3, 1, in)
	idx := mlx.FromInt32([]int32{2, 0, 3}, 3)
	got := mlx.GatherQMM(x, w, scales, biases, idx, group, 4, "affine", false).Float32s()
	for r, j := range []int{2, 0, 3} {
		one := func(a mlx.Array) mlx.Array {
			sh := a.Shape()
			return mlx.Reshape(mlx.Slice(a, []int{j, 0, 0}, []int{j + 1, sh[1], sh[2]}), sh[1], sh[2])
		}
		xr := mlx.Reshape(mlx.Slice(x, []int{r, 0, 0}, []int{r + 1, 1, in}), 1, in)
		want := mlx.QuantizedMatmul(xr, one(w), one(scales), one(biases), group, 4, "affine").Float32s()
		for o := 0; o < out; o++ {
			if g := got[r*out+o]; math.Abs(float64(g-want[o])) > 1e-4*math.Max(1, math.Abs(float64(want[o]))) {
				t.Fatalf("row %d (expert %d), %d: %g, QuantizedMatmul %g", r, j, o, g, want[o])
			}
		}
	}
}

// A pull asks of an MoE repo's tensor names whether its experts are stacked.
func TestExpertsReadable(t *testing.T) {
	for _, c := range []struct {
		names []string
		ok    bool
	}{
		{nil, true},
		{[]string{"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", "language_model.model.layers.0.mlp.gate.weight"}, true},
		{[]string{"model.language_model.layers.0.mlp.experts.gate_up_proj", "model.language_model.layers.0.mlp.gate.weight"}, false},
		{[]string{"model.layers.0.mlp.experts.3.gate_proj.weight"}, false},
		{[]string{"model.layers.0.mlp.gate_proj.weight"}, true},
	} {
		if err := ExpertsReadable(c.names); (err == nil) != c.ok {
			t.Errorf("%v: %v", c.names, err)
		}
	}
}
