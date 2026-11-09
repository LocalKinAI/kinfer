package mlx

import (
	"fmt"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/ebitengine/purego"
)

type (
	strMap  uintptr // mlx_map_string_to_array
	strStr  uintptr // mlx_map_string_to_string
	mapIter struct{ ctx, mapCtx uintptr }
)

var (
	mapNew      func() strMap
	mapFree     func(strMap) int32
	mapStrNew   func() strStr
	mapStrFree  func(strStr) int32
	mapIterNew  func(strMap) mapIter
	mapIterFree func(mapIter) int32
	mapIterNext func(key **byte, val *Array, it mapIter) int32
	loadST      func(arrays *strMap, meta *strStr, file string, s Stream) int32
)

func bindIO(lib uintptr) error {
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&mapNew, "mlx_map_string_to_array_new"}, {&mapFree, "mlx_map_string_to_array_free"},
		{&mapStrNew, "mlx_map_string_to_string_new"}, {&mapStrFree, "mlx_map_string_to_string_free"},
		{&mapIterNew, "mlx_map_string_to_array_iterator_new"},
		{&mapIterFree, "mlx_map_string_to_array_iterator_free"},
		{&mapIterNext, "mlx_map_string_to_array_iterator_next"},
		{&loadST, "mlx_load_safetensors"},
	} {
		if _, err := dl.Sym(lib, b.name); err != nil {
			return fmt.Errorf("mlx-c has no %s: %w", b.name, err)
		}
		purego.RegisterLibFunc(b.fn, lib, b.name)
	}
	return nil
}

// LoadSafetensors maps every tensor in a .safetensors file. MLX reads the
// data lazily, on first evaluation. The arrays belong to no scope: the caller
// frees them, or keeps them for the life of the model.
func LoadSafetensors(path string) (tensors map[string]Array, err error) {
	defer Recover(&err)
	m, meta := mapNew(), mapStrNew()
	defer mapFree(m)
	defer mapStrFree(meta)
	check(loadST(&m, &meta, path, cpu))

	tensors = make(map[string]Array)
	it := mapIterNew(m)
	defer mapIterFree(it)
	for {
		var key *byte
		v := arrayNew()
		rc := mapIterNext(&key, &v, it)
		if rc == 2 {
			arrayFree(v)
			return tensors, nil
		}
		check(rc)
		tensors[goString(key)] = v
	}
}
