package mlxmodel

import (
	"math"
	"os"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// noise is n numbers in [-scale, scale], a few of them zero and a few far
// out, where a sigmoid or a softplus takes its other branch.
func noise(seed uint32, n int, scale float32) []float32 {
	v := make([]float32, n)
	x := seed*2654435761 + 12345
	for i := range v {
		x = x*1664525 + 1013904223
		v[i] = (float32(x>>8)/float32(1<<23) - 1) * scale
		switch x >> 28 {
		case 0:
			v[i] = 0
		case 1:
			v[i] *= 40
		}
	}
	return v
}

func bitsOf(t *testing.T, a mlx.Array) []uint32 {
	t.Helper()
	f := mlx.Astype(a, mlx.Float32).Float32s()
	b := make([]uint32, len(f))
	for i, x := range f {
		b[i] = math.Float32bits(x)
	}
	return b
}

// Each fused kernel gives, to the bit, what the operations it stands for
// give: in bfloat16, as a model's activations are, in float16 and in float32.
func TestFusedKernelsMatchOps(t *testing.T) {
	lib := os.Getenv("KINFER_MLX_LIB")
	if lib == "" {
		t.Skip("KINFER_MLX_LIB not set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	defer func() { opsSmall = false }()
	const B, L, in, hid, Hv, Dv = 2, 3, 64, 96, 8, 12
	for _, dt := range []mlx.Dtype{mlx.Bfloat16, mlx.Float16, mlx.Float32} {
		for _, biasT := range []mlx.Dtype{dt, mlx.Float32} {
			s := mlx.NewScope()
			arr := func(seed uint32, scale float32, shape ...int) mlx.Array {
				return mlx.Astype(mlx.FromFloat32(noise(seed, size(shape), scale), shape...), dt)
			}
			// A weight quantized as MLX keeps one — any bits are one — with a
			// scale of its own besides, as an NVFP4 weight has.
			lin := func(seed uint32, out int, global float32) Linear {
				packed := make([]int32, out*in/8)
				x := seed*7919 + 1
				for i := range packed {
					x = x*1664525 + 1013904223
					packed[i] = int32(x >> 1)
				}
				return Linear{W: mlx.Astype(mlx.FromInt32(packed, out, in/8), mlx.Uint32),
					Scales: arr(seed+100, 0.05, out, in/64), Biases: arr(seed+200, 0.3, out, in/64),
					Group: 64, Bits: 4, Mode: "affine", Global: global}
			}
			plain := func(seed uint32, out int) Linear { return Linear{W: arr(seed, 0.6, out, in)} }
			x, h := arr(1, 2, B, L, in), arr(2, 3, B, L, hid)
			o, gate := arr(3, 3, B, L, hid), arr(4, 6, B, L, hid)
			n := arr(5, 2, B, L, Hv, Dv)
			w := &linearAttn{
				inA: lin(10, Hv, 0.37), inB: lin(11, Hv, 1.9), inZ: lin(12, Hv*Dv, 0.011),
				dtBias: mlx.Astype(mlx.FromFloat32(noise(13, Hv, 2), Hv), biasT),
				aLog:   mlx.Astype(mlx.FromFloat32(noise(14, Hv, 1), Hv), dt),
			}
			w.nexp = mlx.Negative(mlx.Exp(mlx.Astype(w.aLog, mlx.Float32)))
			cases := map[string]func() []mlx.Array{
				"project":                func() []mlx.Array { return []mlx.Array{project(x, lin(20, hid, 0.0123))} },
				"addProjected":           func() []mlx.Array { return []mlx.Array{addProjected(h, x, lin(21, hid, 0.0123))} },
				"addProjected, no scale": func() []mlx.Array { return []mlx.Array{addProjected(h, x, lin(21, hid, 0))} },
				"addProjected, plain":    func() []mlx.Array { return []mlx.Array{addProjected(h, x, plain(24, hid))} },
				"swiGLUIn, plain":        func() []mlx.Array { return []mlx.Array{swiGLUIn(x, plain(25, hid), plain(26, hid))} },
				"swiGLUIn":               func() []mlx.Array { return []mlx.Array{swiGLUIn(x, lin(22, hid, 0.7), lin(23, hid, 2.3))} },
				"swiGLUIn, no scale":     func() []mlx.Array { return []mlx.Array{swiGLUIn(x, lin(22, hid, 0), lin(23, hid, 0))} },
				"gated":                  func() []mlx.Array { return []mlx.Array{gated(o, gate)} },
				"deltaGates": func() []mlx.Array {
					_, b, a := w.gateInputs(x)
					g, beta := w.deltaGates(a, b)
					return []mlx.Array{g, beta}
				},
				"deltaOut": func() []mlx.Array {
					z, _, _ := w.gateInputs(x)
					return []mlx.Array{w.deltaOut(n, z, dt), w.deltaOut(mlx.Astype(n, mlx.Float32), z, dt)}
				},
			}
			for name, f := range cases {
				opsSmall = true
				want := f()
				opsSmall = false
				got := f()
				for i := range want {
					if want[i].Dtype() != got[i].Dtype() {
						t.Errorf("%v (bias %v) %s: output %d is %v, the operations' is %v", dt, biasT, name, i, got[i].Dtype(), want[i].Dtype())
						continue
					}
					wb, gb := bitsOf(t, want[i]), bitsOf(t, got[i])
					if len(wb) != len(gb) {
						t.Errorf("%v (bias %v) %s: output %d has %d numbers, the operations' %d", dt, biasT, name, i, len(gb), len(wb))
						continue
					}
					bad, finite := 0, 0
					for j := range wb {
						if f := math.Float32frombits(wb[j]); f != 0 && !math.IsNaN(float64(f)) && !math.IsInf(float64(f), 0) {
							finite++
						}
						if wb[j] != gb[j] {
							if bad == 0 {
								t.Errorf("%v (bias %v) %s: output %d differs at %d: %g, the operations give %g", dt, biasT, name, i, j,
									math.Float32frombits(gb[j]), math.Float32frombits(wb[j]))
							}
							bad++
						}
					}
					if bad > 0 {
						t.Errorf("%v (bias %v) %s: output %d differs in %d of %d numbers", dt, biasT, name, i, bad, len(wb))
					}
					if finite*3 < len(wb) {
						t.Errorf("%v (bias %v) %s: output %d has only %d ordinary numbers of %d: the test's data proves little", dt, biasT, name, i, finite, len(wb))
					}
				}
			}
			s.Close()
			if dt == mlx.Float32 {
				break
			}
		}
	}
}

// A model run with the fused kernels gives, to the bit, the logits it gives
// run with the operations they stand for: reading a prompt, stepping, in a
// pass that checks drafts, drafting, and stepping in a batch.
func TestQwen35FusedMatchesOps(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_MLX_MODEL")
	if lib == "" || dir == "" {
		t.Skip("KINFER_MLX_LIB and KINFER_MLX_MODEL not both set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadQwen35(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	defer func() { opsSmall = false }()
	prompt := make([]int32, 40)
	for i := range prompt {
		prompt[i] = int32(1000 + (i*104729)%50000)
	}
	run := func() map[string][]uint32 {
		out := map[string][]uint32{}
		c := m.NewCache()
		defer c.Free()
		c.Drafting = m.CanDraft()
		lg, err := m.Forward([][]int32{prompt}, c, nil)
		if err != nil {
			t.Fatal(err)
		}
		out["prompt"] = bitsOf(t, lg)
		lg.Free()
		for i, tok := range []int32{2001, 2002} {
			lg, err := m.Forward([][]int32{{tok}}, c, nil)
			if err != nil {
				t.Fatal(err)
			}
			out[[]string{"step 1", "step 2"}[i]] = bitsOf(t, lg)
			lg.Free()
		}
		var drafts mlx.Array
		if m.CanDraft() {
			if drafts, err = m.Draft(c, 2003, 3); err != nil {
				t.Fatal(err)
			}
			out["drafts"] = bitsOf(t, drafts)
		} else {
			drafts = mlx.FromInt32([]int32{2004, 2005, 2006}, 1, 3)
		}
		mark := c.Mark()
		logits, hidden, err := m.Verify(2003, drafts, c)
		if err != nil {
			t.Fatal(err)
		}
		mlx.Eval(append([]mlx.Array{logits, hidden}, c.Arrays()...)...)
		out["pass"] = bitsOf(t, logits)
		if err := c.Keep(mark, 1, hidden); err != nil {
			t.Fatal(err)
		}
		logits.Free()
		hidden.Free()
		drafts.Free()
		lg, err = m.Forward([][]int32{{2007}}, c, nil)
		if err != nil {
			t.Fatal(err)
		}
		out["after the pass"] = bitsOf(t, lg)
		lg.Free()

		b := m.NewBatch()
		defer b.Free()
		for r := 0; r < 2; r++ {
			rc := m.NewCache()
			lg, err := m.Forward([][]int32{prompt[:20+10*r]}, rc, nil)
			if err != nil {
				t.Fatal(err)
			}
			lg.Free()
			if _, err := b.Add(rc, 2); err != nil {
				t.Fatal(err)
			}
		}
		y := mlx.FromInt32([]int32{2008, 2009}, 2, 1)
		for i := 0; i < 2; i++ {
			lg := b.Step(y)
			mlx.Eval(append([]mlx.Array{lg}, b.Arrays()...)...)
			out[[]string{"batch step 1", "batch step 2"}[i]] = bitsOf(t, lg)
			lg.Free()
		}
		y.Free()
		return out
	}
	opsSmall = true
	want := run()
	opsSmall = false
	got := run()
	for name, w := range want {
		g := got[name]
		bad := 0
		for i := range w {
			if i >= len(g) || w[i] != g[i] {
				bad++
			}
		}
		if bad > 0 || len(g) != len(w) {
			t.Errorf("%s: %d of %d numbers differ from the operations'", name, bad, len(w))
		} else {
			t.Logf("%s: the same %d numbers", name, len(w))
		}
	}
}
