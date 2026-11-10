package mlx

import (
	"fmt"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/ebitengine/purego"
)

var (
	memActive     func(res *uintptr) int32
	memCache      func(res *uintptr) int32
	memPeak       func(res *uintptr) int32
	memClearCache func() int32
	memCacheLimit func(res *uintptr, limit uintptr) int32
	memResetPeak  func() int32
)

func bindMemory(lib uintptr) error {
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&memActive, "mlx_get_active_memory"}, {&memCache, "mlx_get_cache_memory"},
		{&memPeak, "mlx_get_peak_memory"}, {&memClearCache, "mlx_clear_cache"},
		{&memCacheLimit, "mlx_set_cache_limit"}, {&memResetPeak, "mlx_reset_peak_memory"},
	} {
		if _, err := dl.Sym(lib, b.name); err != nil {
			return fmt.Errorf("mlx-c has no %s: %w", b.name, err)
		}
		purego.RegisterLibFunc(b.fn, lib, b.name)
	}
	return nil
}

// Memory is MLX's own account of the GPU memory it holds: Active in live
// arrays, Cache in freed buffers it keeps for reuse, Peak the most Active has
// been.
type Memory struct{ Active, Cache, Peak int64 }

// MemoryUse reads MLX's memory counters.
func MemoryUse() Memory {
	var a, c, p uintptr
	memActive(&a)
	memCache(&c)
	memPeak(&p)
	return Memory{int64(a), int64(c), int64(p)}
}

// SetCacheLimit caps the freed buffers MLX keeps for reuse, returning the
// previous cap. MLX's default is its whole memory limit, which a server whose
// batch and context sizes keep changing fills with buffers of shapes it
// rarely meets again.
func SetCacheLimit(bytes int64) int64 {
	var prev uintptr
	memCacheLimit(&prev, uintptr(bytes))
	return int64(prev)
}

// ResetPeak starts Peak over from what is active now.
func ResetPeak() { memResetPeak() }

// ClearCache returns MLX's cached buffers to the system.
func ClearCache() { memClearCache() }
