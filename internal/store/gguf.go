package store

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

// Just enough GGUF to size a cache.
//
// llama.cpp knows all of this, and asking it means loading the model — which on
// the machine this was written for is 34 seconds and 73 GiB, to answer a
// question whose whole point is to be asked beforehand. The header carries what
// is needed and sits at the front of the file.

// Shape is what a model's KV cache is built from.
type Shape struct {
	Arch string

	Layers     int // block_count
	HeadsKV    int // attention.head_count_kv
	EmbedDim   int // embedding_length
	Heads      int // attention.head_count
	TrainedCtx int // context_length

	// Experts is nonzero for a mixture of experts. It does not change the cache,
	// but it changes what batching buys, so fit says so.
	Experts int
}

// HeadDim is the size of one attention head, which GGUF does not always state.
func (s Shape) HeadDim() int {
	if s.Heads > 0 && s.EmbedDim > 0 {
		return s.EmbedDim / s.Heads
	}
	return 0
}

// CacheBytes estimates the KV cache for a given total token count, at f16.
//
// Two bytes per element, two tensors (K and V), one entry per layer per head.
// That is the standard attention cache and it is what most models have.
//
// It is an estimate, and the direction it errs in is known: a hybrid
// architecture — Qwen3-Next and its relatives — replaces some layers with
// recurrent state whose size is per sequence rather than per token, and those
// cost more than this predicts at high sequence counts and less at long
// contexts. Measured on qwen3.8-flash-next, 32768 tokens cost 1.7 GiB spread
// over 2 sequences and 4.3 GiB spread over 32.
func (s Shape) CacheBytes(tokens int) int64 {
	d := s.HeadDim()
	if s.Layers == 0 || s.HeadsKV == 0 || d == 0 {
		return 0
	}
	return int64(tokens) * int64(s.Layers) * int64(s.HeadsKV) * int64(d) * 2 * 2
}

// ReadShape reads a GGUF file's header.
func ReadShape(path string) (Shape, error) {
	f, err := os.Open(path)
	if err != nil {
		return Shape{}, err
	}
	defer f.Close()

	r := &ggufReader{r: f}
	magic, err := r.bytes(4)
	if err != nil || string(magic) != "GGUF" {
		return Shape{}, fmt.Errorf("%s is not a GGUF file", path)
	}
	if _, err := r.u32(); err != nil { // version
		return Shape{}, err
	}
	if _, err := r.u64(); err != nil { // tensor count
		return Shape{}, err
	}
	nKV, err := r.u64()
	if err != nil {
		return Shape{}, err
	}

	var sh Shape
	for i := uint64(0); i < nKV; i++ {
		key, err := r.str()
		if err != nil {
			return Shape{}, err
		}
		typ, err := r.u32()
		if err != nil {
			return Shape{}, err
		}
		v, err := r.value(typ)
		if err != nil {
			return sh, nil // a type this reader does not know ends the walk, not the answer
		}
		n, _ := v.(uint64)
		switch {
		case key == "general.architecture":
			sh.Arch, _ = v.(string)
		case strings.HasSuffix(key, ".block_count"):
			sh.Layers = int(n)
		case strings.HasSuffix(key, ".attention.head_count_kv"):
			sh.HeadsKV = int(n)
		case strings.HasSuffix(key, ".attention.head_count"):
			sh.Heads = int(n)
		case strings.HasSuffix(key, ".embedding_length"):
			sh.EmbedDim = int(n)
		case strings.HasSuffix(key, ".context_length"):
			sh.TrainedCtx = int(n)
		case strings.HasSuffix(key, ".expert_count"):
			sh.Experts = int(n)
		}
	}
	return sh, nil
}

type ggufReader struct{ r io.Reader }

func (g *ggufReader) bytes(n uint64) ([]byte, error) {
	if n > 1<<24 {
		return nil, fmt.Errorf("gguf: implausible length %d", n)
	}
	b := make([]byte, n)
	_, err := io.ReadFull(g.r, b)
	return b, err
}

func (g *ggufReader) u32() (uint32, error) {
	b, err := g.bytes(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (g *ggufReader) u64() (uint64, error) {
	b, err := g.bytes(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

func (g *ggufReader) str() (string, error) {
	n, err := g.u64()
	if err != nil {
		return "", err
	}
	b, err := g.bytes(n)
	return string(b), err
}

// value reads one metadata value, returning integers widened to uint64 so the
// caller has one case to handle.
//
// Arrays must be consumed entirely even though nothing here wants one: the
// tokenizer's is 150,000 strings, and skipping past it by reading four would
// leave every subsequent key misaligned. That bug produced a file that appeared
// to have no architecture at all.
func (g *ggufReader) value(typ uint32) (any, error) {
	switch typ {
	case 0, 1: // uint8, int8
		b, err := g.bytes(1)
		if err != nil {
			return nil, err
		}
		return uint64(b[0]), nil
	case 2, 3: // uint16, int16
		b, err := g.bytes(2)
		if err != nil {
			return nil, err
		}
		return uint64(binary.LittleEndian.Uint16(b)), nil
	case 4, 5: // uint32, int32
		v, err := g.u32()
		return uint64(v), err
	case 6: // float32
		_, err := g.bytes(4)
		return nil, err
	case 7: // bool
		_, err := g.bytes(1)
		return nil, err
	case 8: // string
		return g.str()
	case 9: // array
		et, err := g.u32()
		if err != nil {
			return nil, err
		}
		n, err := g.u64()
		if err != nil {
			return nil, err
		}
		for i := uint64(0); i < n; i++ {
			if _, err := g.value(et); err != nil {
				return nil, err
			}
		}
		return nil, nil
	case 10, 11: // uint64, int64
		v, err := g.u64()
		return v, err
	case 12: // float64
		_, err := g.bytes(8)
		return nil, err
	}
	return nil, fmt.Errorf("gguf: unknown value type %d", typ)
}
