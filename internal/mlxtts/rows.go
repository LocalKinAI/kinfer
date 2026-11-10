package mlxtts

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// rowTable is an embedding table read a row at a time from its safetensors
// file. The talker's text embedding is 151,936 rows of 2,048 — 622 MB in
// bfloat16 — of which a sentence reads a few dozen, once: kept on disk, the
// model takes that much less memory, and the rows are the same bits.
type rowTable struct {
	f        *os.File
	off      int64 // where row 0 starts in the file
	rows     int
	dim      int
	dtype    string // safetensors': BF16, F16 or F32
	rowBytes int
}

// openRows finds tensor name among dir's safetensors.
func openRows(dir, name string) (*rowTable, error) {
	files, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if b, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json")); err == nil {
		var idx struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if json.Unmarshal(b, &idx) == nil && idx.WeightMap[name] != "" {
			files = []string{filepath.Join(dir, idx.WeightMap[name])}
		}
	}
	for _, path := range files {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		var n uint64
		if err := binary.Read(f, binary.LittleEndian, &n); err != nil || n > 100<<20 {
			f.Close()
			continue
		}
		head := make([]byte, n)
		if _, err := f.ReadAt(head, 8); err != nil {
			f.Close()
			continue
		}
		var tensors map[string]json.RawMessage
		if json.Unmarshal(head, &tensors) != nil {
			f.Close()
			continue
		}
		raw, ok := tensors[name]
		if !ok {
			f.Close()
			continue
		}
		var t struct {
			Dtype   string  `json:"dtype"`
			Shape   []int   `json:"shape"`
			Offsets []int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &t); err != nil || len(t.Shape) != 2 || len(t.Offsets) != 2 {
			f.Close()
			return nil, fmt.Errorf("%s in %s: not a table", name, path)
		}
		width := map[string]int{"BF16": 2, "F16": 2, "F32": 4}[t.Dtype]
		if width == 0 {
			f.Close()
			return nil, fmt.Errorf("%s in %s is %s, not a float table", name, path, t.Dtype)
		}
		return &rowTable{f: f, off: 8 + int64(n) + t.Offsets[0], rows: t.Shape[0], dim: t.Shape[1],
			dtype: t.Dtype, rowBytes: width * t.Shape[1]}, nil
	}
	return nil, fmt.Errorf("no safetensors in %s holds %s", dir, name)
}

// lookup is rows ids as [1, L, dim] of dt (the table's own type: each value
// is exact in float32 on the way).
func (r *rowTable) lookup(ids []int32, dt mlx.Dtype) (mlx.Array, error) {
	out := make([]float32, len(ids)*r.dim)
	buf := make([]byte, r.rowBytes)
	for i, id := range ids {
		if id < 0 || int(id) >= r.rows {
			return 0, fmt.Errorf("token %d is outside the embedding's %d rows", id, r.rows)
		}
		if _, err := r.f.ReadAt(buf, r.off+int64(id)*int64(r.rowBytes)); err != nil {
			return 0, err
		}
		row := out[i*r.dim : (i+1)*r.dim]
		for j := range row {
			switch r.dtype {
			case "BF16":
				row[j] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(buf[2*j:])) << 16)
			case "F16":
				row[j] = halfToFloat(binary.LittleEndian.Uint16(buf[2*j:]))
			case "F32":
				row[j] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*j:]))
			}
		}
	}
	a := mlx.FromFloat32(out, 1, len(ids), r.dim)
	if dt == mlx.Float32 {
		return a, nil
	}
	return mlx.Astype(a, dt), nil
}

func (r *rowTable) close() {
	if r != nil && r.f != nil {
		r.f.Close()
		r.f = nil
	}
}

// halfToFloat is IEEE half precision's value.
func halfToFloat(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := int32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal: normalize it
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3ff
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | frac<<13)
	}
	return math.Float32frombits(sign | uint32(exp+112)<<23 | frac<<13)
}
