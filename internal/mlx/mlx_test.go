package mlx

import (
	"math"
	"os"
	"testing"
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
