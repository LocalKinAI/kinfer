package mlx

import (
	"fmt"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/ebitengine/purego"
)

// optDtype mirrors mlx_optional_dtype (8 bytes).
type optDtype struct {
	Value Dtype
	Has   uint8
	_     [3]byte
}

var (
	opRandomKey   func(res *Array, seed uint64) int32
	opCategorical func(res *Array, logits Array, axis int32, key Array, s Stream) int32
	opCumsum      func(res *Array, a Array, axis int32, reverse, inclusive bool, dt optDtype, s Stream) int32
)

func bindRandom(lib uintptr) error {
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&opRandomKey, "mlx_random_key"}, {&opCategorical, "mlx_random_categorical"},
		{&opCumsum, "mlx_cumsum_axis"},
	} {
		if _, err := dl.Sym(lib, b.name); err != nil {
			return fmt.Errorf("mlx-c has no %s: %w", b.name, err)
		}
		purego.RegisterLibFunc(b.fn, lib, b.name)
	}
	return nil
}

// RandomKey is a PRNG key for seed.
func RandomKey(seed uint64) Array {
	r := arrayNew()
	return out(opRandomKey(&r, seed), r)
}

// Categorical draws one index per row from the distribution softmax(logits)
// along axis, using key.
func Categorical(logits Array, axis int, key Array) Array {
	r := arrayNew()
	return out(opCategorical(&r, logits, int32(axis), key, gpu), r)
}

// Cumsum is the running sum along axis.
func Cumsum(a Array, axis int, reverse, inclusive bool) Array {
	r := arrayNew()
	return out(opCumsum(&r, a, int32(axis), reverse, inclusive, optDtype{}, gpu), r)
}
