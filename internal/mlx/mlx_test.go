package mlx

import (
	"math"
	"os"
	"testing"
	"time"
)

// The tests need the MLX libraries, so they run where those are:
// KINFER_MLX_LIB names the directory holding libmlxc.dylib, libmlx.dylib and
// mlx.metallib. Without it they skip.
func openForTest(t *testing.T) {
	t.Helper()
	dir := os.Getenv("KINFER_MLX_LIB")
	if dir == "" {
		t.Skip("KINFER_MLX_LIB not set")
	}
	if err := Open(dir); err != nil {
		t.Fatal(err)
	}
}

// A matmul on the GPU agrees with the same product worked out in Go.
func TestMatmulOnGPU(t *testing.T) {
	openForTest(t)
	s := NewScope()
	defer s.Close()

	const m, k, n = 64, 128, 32
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(math.Sin(float64(i)))
	}
	for i := range b {
		b[i] = float32(math.Cos(float64(i) * 0.7))
	}
	got := Matmul(FromFloat32(a, m, k), FromFloat32(b, k, n)).Float32s()

	worst := 0.0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var want float64
			for x := 0; x < k; x++ {
				want += float64(a[i*k+x]) * float64(b[x*n+j])
			}
			worst = math.Max(worst, math.Abs(float64(got[i*n+j])-want))
		}
	}
	if worst > 1e-3 {
		t.Fatalf("largest difference from Go's product %g", worst)
	}
	t.Logf("%dx%d @ %dx%d on the GPU: largest difference from Go %.2g", m, k, k, n, worst)
}

// A shape mistake comes back as an error instead of MLX's exit(-1).
func TestErrorDoesNotExit(t *testing.T) {
	openForTest(t)
	s := NewScope()
	defer s.Close()

	err := func() (err error) {
		defer Recover(&err)
		Matmul(FromFloat32(make([]float32, 6), 2, 3), FromFloat32(make([]float32, 8), 2, 4))
		return nil
	}()
	if err == nil {
		t.Fatal("a 2x3 @ 2x4 matmul did not fail")
	}
	t.Logf("2x3 @ 2x4: %v", err)
}

// Arrays made in a scope are freed when it closes; a kept one survives. This
// also runs on a different goroutine (and most likely thread) from the test
// that opened the library, which an ordinary MLX stream refuses.
func TestScopeKeep(t *testing.T) {
	openForTest(t)
	outer := NewScope()
	defer outer.Close()

	inner := NewScope()
	x := Add(FromFloat32([]float32{1, 2}, 2), FromFloat32([]float32{10, 20}, 2))
	inner.Keep(x)
	inner.Close()

	if got := x.Float32s(); got[0] != 11 || got[1] != 22 {
		t.Fatalf("kept array reads %v", got)
	}
}

// What one small operation costs in a long chain of them: a decode step of a
// large model is a few thousand, most on arrays of a few thousand numbers,
// beside the products that read its weights.
func TestSmallOpCost(t *testing.T) {
	openForTest(t)
	if os.Getenv("KINFER_MLX_OPCOST") == "" {
		t.Skip("KINFER_MLX_OPCOST not set")
	}
	v := make([]float32, 5120)
	for i := range v {
		v[i] = float32(i%7) - 3
	}
	x := Astype(FromFloat32(v, 1, 1, 5120), Bfloat16)
	x.Eval()
	k := NewKernel("kinfer_test_scale", []string{"y", "s"}, []string{"out"}, `
        uint i = thread_position_in_grid.x;
        out[i] = static_cast<InT>(static_cast<float>(y[i]) * s);
`)
	l := k.Prepare([]KernelOutput{{Shape: []int{1, 1, 5120}, Dtype: Bfloat16}}, []KernelTemplate{{Name: "InT", Dtype: Bfloat16}},
		[3]int{5120, 1, 1}, [3]int{256, 1, 1})
	for _, kind := range []string{"operations", "kernel launches"} {
		for _, n := range []int{250, 1000, 4000} {
			var best time.Duration
			for rep := 0; rep < 6; rep++ {
				s := NewScope()
				start := time.Now()
				y := x
				for i := 0; i < n; i++ {
					if kind == "operations" {
						y = Multiply(y, ScalarAs(1.0001, Bfloat16))
					} else {
						y = l.Run(y, Scalar(1.0001))[0]
					}
				}
				built := time.Since(start)
				y.Eval()
				el := time.Since(start)
				s.Close()
				if rep == 0 || el < best {
					best = el
				}
				if rep == 5 {
					t.Logf("%d small %s: %.2f ms (%.1f µs each); recording them took %.2f ms", n, kind,
						float64(best.Microseconds())/1000, float64(best.Microseconds())/float64(n), float64(built.Microseconds())/1000)
				}
			}
		}
	}
}

// A graph recorded over a slot reads what the slot holds when the graph is
// evaluated, not what it held when the graph was recorded; and what was
// evaluated keeps its value when the slot is set again.
func TestSlotIsReadWhenEvaluated(t *testing.T) {
	openForTest(t)
	table := make([]float32, 64)
	for i := range table {
		table[i] = float32(i) * 1.5
	}
	tab := Astype(FromFloat32(table, 64, 1), Bfloat16)
	tab.Eval()
	for round := 0; round < 50; round++ {
		s := NewSlot(1, 2)
		a := Take(tab, s.Array, 0)                    // [1, 2, 1]
		b := Add(Astype(s.Array, Float32), Scalar(1)) // through arithmetic too
		x, y := int32(round%64), int32((round*7+3)%64)
		s.Set(x, y)
		AsyncEval(a, b)
		got, plus := Astype(a, Float32).Float32s(), b.Float32s()
		if got[0] != table[x] || got[1] != table[y] || plus[0] != float32(x)+1 || plus[1] != float32(y)+1 {
			t.Fatalf("round %d: slot set to %d, %d read as rows %v and numbers %v", round, x, y, got, plus)
		}
		s.Set(63, 62)
		if again := Astype(a, Float32).Float32s(); again[0] != table[x] || again[1] != table[y] {
			t.Fatalf("round %d: what was evaluated changed when the slot was set again: %v", round, again)
		}
		a.Free()
		b.Free()
		s.Free()
	}
}
