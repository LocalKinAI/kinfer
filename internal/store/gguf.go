package store

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strconv"
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
	KeyLen     int // attention.key_length, when a head is not EmbedDim/Heads
	ValLen     int // attention.value_length
	TrainedCtx int // context_length

	// Hybrid architectures — Qwen3.5, Qwen3-Next and their relatives — put
	// attention on every AttnInterval-th layer and a recurrent block on the
	// rest. The recurrent blocks keep a state per sequence that does not grow
	// with the context: sized from the ssm.* keys below.
	AttnInterval int // full_attention_interval; 0 means every layer attends
	SSMConv      int // ssm.conv_kernel
	SSMState     int // ssm.state_size
	SSMGroups    int // ssm.group_count
	SSMInner     int // ssm.inner_size

	// Experts is nonzero for a mixture of experts. It does not change the cache,
	// but it changes what batching buys, so fit says so.
	Experts int

	// LazyBytes is the part of the files llama.cpp leaves on disk and reads a
	// row at a time, summed over every shard: a model's size less this is
	// what it takes in memory. See ReadsLazily.
	LazyBytes int64

	// Vocab is the tokenizer's vocabulary: a pass of n tokens may produce n x
	// Vocab logits, the largest part of its compute buffer.
	Vocab int

	// WeightBytes is what the model's tensors hold in memory, every shard
	// counted and LazyBytes left out. LayerBytes is the part of it in each
	// layer — what a device holding that layer holds — and ExpertBytes the
	// part of that which is the layer's routed experts: what a GPU short of
	// memory can leave in host memory (see engine.Place). All are zero when a
	// shard could not be read, since a model counted in part would look
	// smaller than it is.
	WeightBytes int64
	LayerBytes  []int64
	ExpertBytes []int64
}

// ReadsLazily reports whether the embedded llama.cpp reads part of an
// architecture's weights from disk on demand, and LazyBytes is worth asking.
//
// Three architectures mark one tensor for it — per_layer_token_embd, the
// n-gram table of DeepSeek V4.1 (its "Engram", 111 GB) and Qwen3.8-Flash-Next,
// and Gemma 4's per-layer embeddings — with TENSOR_READ_LAZY in
// src/models/{deepseek41,qwen4exp,gemma4}.cpp at b11998. llama.cpp's default
// lazy mode, which kinfer uses, honours the mark for a tensor over 4 GiB and
// loads a smaller one whole.
//
// Counting such a table as weights put V4.1's 2-bit set at 249 GB of memory
// where it needs 138, and Flash-Next's IQ3_S at 84 GB, which a 77.8 GiB budget
// cannot hold, where it needs 55, which it can.
func ReadsLazily(arch string) bool { return lazyArchs[arch] }

var lazyArchs = map[string]bool{"deepseek41": true, "qwen4exp": true, "gemma4": true}

const (
	lazyTensor  = "per_layer_token_embd.weight"
	lazyMinimum = 4 << 30 // ggml_nbytes at or below this is loaded whole
)

// Header is what one GGUF file's header says: the model's shape (complete only
// in a split model's first file), how many files the model is split into, and
// where this file's tensors sit.
type Header struct {
	Shape      Shape
	SplitCount int
	SplitNo    int // 0 for a model's first file, which is the one llama.cpp opens

	tensors   []tensorEntry
	dataStart int64
	fileSize  int64
}

type tensorEntry struct {
	name   string
	offset uint64 // from the start of the data section
}

// Lazy is how much of this file llama.cpp reads on demand for a model of arch.
// A split model's later files carry no architecture of their own, so it is
// passed in.
func (h Header) Lazy(arch string) int64 {
	var lazy int64
	h.eachTensor(func(name string, size int64) {
		if isLazy(arch, name, size) {
			lazy += size
		}
	})
	return lazy
}

func isLazy(arch, name string, size int64) bool {
	return lazyArchs[arch] && name == lazyTensor && size > lazyMinimum
}

// eachTensor calls fn with each tensor of this file and its size: the distance
// to the next one, the data section's end for the last — alignment padding
// included, at most a few dozen bytes.
func (h Header) eachTensor(fn func(name string, size int64)) {
	ts := append([]tensorEntry(nil), h.tensors...)
	sort.Slice(ts, func(i, j int) bool { return ts[i].offset < ts[j].offset })
	for i, t := range ts {
		end := h.fileSize - h.dataStart
		if i+1 < len(ts) {
			end = int64(ts[i+1].offset)
		}
		fn(t.name, end-int64(t.offset))
	}
}

// layerTensor names a tensor of a layer and captures the layer;
// expertTensor names a layer's routed experts, as llama.cpp's --cpu-moe
// matches them (LLM_FFN_EXPS_REGEX in common/common.h).
var (
	layerTensor  = regexp.MustCompile(`^blk\.(\d+)\.`)
	expertTensor = regexp.MustCompile(`^blk\.\d+\.ffn_(up|down|gate|gate_up)_(ch|)exps\.`)
)

// account adds this file's tensors to sh, a model of arch: what is read
// lazily, what is held in memory, and of that what each layer holds and how
// much of it is routed experts.
func (h Header) account(sh *Shape, arch string) {
	h.eachTensor(func(name string, size int64) {
		if isLazy(arch, name, size) {
			sh.LazyBytes += size
			return
		}
		sh.WeightBytes += size
		m := layerTensor.FindStringSubmatch(name)
		if m == nil {
			return
		}
		layer, _ := strconv.Atoi(m[1])
		for len(sh.LayerBytes) <= layer {
			sh.LayerBytes = append(sh.LayerBytes, 0)
			sh.ExpertBytes = append(sh.ExpertBytes, 0)
		}
		sh.LayerBytes[layer] += size
		if expertTensor.MatchString(name) {
			sh.ExpertBytes[layer] += size
		}
	})
}

// HeadDim is the size of one attention head. GGUF states it outright when it
// is not EmbedDim/Heads, which on the Qwen3.5 family it is not.
func (s Shape) HeadDim() int {
	if s.KeyLen > 0 {
		return s.KeyLen
	}
	if s.Heads > 0 && s.EmbedDim > 0 {
		return s.EmbedDim / s.Heads
	}
	return 0
}

// AttnLayers is how many layers hold a KV cache.
func (s Shape) AttnLayers() int {
	if s.AttnInterval > 1 {
		return s.Layers / s.AttnInterval
	}
	return s.Layers
}

// CacheBytes estimates the attention cache for a total token count, at f16:
// two bytes per element, K and V, one entry per attending layer per KV head.
//
// This is the part that grows with the context. A hybrid model also holds a
// per-sequence state that does not — see CacheBytesFor, which is what sizing
// uses. Before the header's full_attention_interval and ssm.* keys were read,
// this counted every layer as attending and put the 35B's cache at twice its
// measured size, which is how a server was sized to 32768 tokens per
// conversation on a machine with room for 65536.
func (s Shape) CacheBytes(tokens int) int64 {
	d := s.HeadDim()
	if s.AttnLayers() == 0 || s.HeadsKV == 0 || d == 0 {
		return 0
	}
	v := s.ValLen
	if v == 0 {
		v = d
	}
	return int64(tokens) * int64(s.AttnLayers()) * int64(s.HeadsKV) * int64(d+v) * 2
}

// RecurrentBytesPerSeq is the state one sequence holds in the recurrent
// layers of a hybrid model, in f32: the convolution window and the SSM state.
//
// Checked against llama.cpp's own report for the 35B (16 sequences, 30
// recurrent layers): R 45.00 MiB and S 960.00 MiB, which is 96 KiB and 2 MiB
// per sequence per layer — exactly what these terms give.
func (s Shape) RecurrentBytesPerSeq() int64 {
	if s.AttnInterval <= 1 || s.SSMInner == 0 || s.SSMState == 0 {
		return 0
	}
	recurrent := s.Layers - s.AttnLayers()
	conv := int64(max(s.SSMConv-1, 0)) * int64(s.SSMInner+2*s.SSMGroups*s.SSMState)
	state := int64(s.SSMInner) * int64(s.SSMState)
	return int64(recurrent) * (conv + state) * 4
}

// CacheBytesFor is the whole cache for tokens spread over seqs sequences: the
// attention part, which scales with tokens, plus the recurrent part, which
// scales with sequences.
func (s Shape) CacheBytesFor(tokens, seqs int) int64 {
	attn := s.CacheBytes(tokens)
	if attn == 0 {
		return 0
	}
	return attn + int64(seqs)*s.RecurrentBytesPerSeq()
}

// ReadShape reads a GGUF model's header — from its first file, for a split
// model — and the headers of its other files, to total what its tensors hold.
func ReadShape(path string) (Shape, error) {
	h, err := readHeaderFile(path)
	if err != nil {
		return Shape{}, err
	}
	sh := h.Shape
	h.account(&sh, sh.Arch)
	whole := true
	for _, p := range otherShards(path, h.SplitCount) {
		// A missing shard is llama.cpp's to report at load; it cannot be
		// counted here.
		oh, err := readHeaderFile(p)
		if err != nil {
			whole = false
			continue
		}
		oh.account(&sh, sh.Arch)
	}
	if !whole {
		sh.WeightBytes, sh.LayerBytes, sh.ExpertBytes = 0, nil, nil
	}
	return sh, nil
}

func readHeaderFile(path string) (Header, error) {
	f, err := os.Open(path)
	if err != nil {
		return Header{}, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return Header{}, err
	}
	h, err := ReadHeader(f, info.Size())
	if err != nil {
		return Header{}, fmt.Errorf("%s: %w", path, err)
	}
	return h, nil
}

// otherShards names the rest of a split model from its first file's name,
// "…-00001-of-00004.gguf". A file not named that way has none.
func otherShards(first string, count int) []string {
	m := shardSuffix.FindStringSubmatch(first)
	if m == nil || count < 2 {
		return nil
	}
	base := first[:len(first)-len(m[0])]
	var out []string
	for i := 2; i <= count; i++ {
		out = append(out, fmt.Sprintf("%s-%05d-of-%05d.gguf", base, i, count))
	}
	return out
}

// ReadHeader reads a GGUF header from r, a file of fileSize bytes: the shape
// keys, the split count and where each tensor sits. It reads only the header,
// which is why hub can call it over a ranged HTTP request.
func ReadHeader(r io.Reader, fileSize int64) (Header, error) {
	cr := &countingReader{r: bufio.NewReaderSize(r, 1<<16)}
	g := &ggufReader{r: cr}
	magic, err := g.bytes(4)
	if err != nil || string(magic) != "GGUF" {
		return Header{}, fmt.Errorf("not a GGUF file")
	}
	if _, err := g.u32(); err != nil { // version
		return Header{}, err
	}
	nTensors, err := g.u64()
	if err != nil {
		return Header{}, err
	}
	nKV, err := g.u64()
	if err != nil {
		return Header{}, err
	}

	h := Header{SplitCount: 1, fileSize: fileSize}
	sh := &h.Shape
	alignment := uint64(32) // GGUF's default when general.alignment is absent
	for i := uint64(0); i < nKV; i++ {
		key, err := g.str()
		if err != nil {
			return Header{}, err
		}
		typ, err := g.u32()
		if err != nil {
			return Header{}, err
		}
		v, err := g.value(typ)
		if err != nil {
			return h, nil // a type this reader does not know ends the walk, not the answer
		}
		n, _ := v.(uint64)
		switch {
		case key == "split.count":
			h.SplitCount = int(n)
		case key == "split.no":
			h.SplitNo = int(n)
		case key == "general.alignment":
			if n > 0 {
				alignment = n
			}
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
		case strings.HasSuffix(key, ".attention.key_length"):
			sh.KeyLen = int(n)
		case strings.HasSuffix(key, ".attention.value_length"):
			sh.ValLen = int(n)
		case strings.HasSuffix(key, ".full_attention_interval"):
			sh.AttnInterval = int(n)
		case strings.HasSuffix(key, ".ssm.conv_kernel"):
			sh.SSMConv = int(n)
		case strings.HasSuffix(key, ".ssm.state_size"):
			sh.SSMState = int(n)
		case strings.HasSuffix(key, ".ssm.group_count"):
			sh.SSMGroups = int(n)
		case strings.HasSuffix(key, ".ssm.inner_size"):
			sh.SSMInner = int(n)
		case key == "tokenizer.ggml.tokens":
			if a, ok := v.(arrayLen); ok {
				sh.Vocab = int(a)
			}
		}
	}

	for i := uint64(0); i < nTensors; i++ {
		name, err := g.str()
		if err != nil {
			return Header{}, err
		}
		nDims, err := g.u32()
		if err != nil {
			return Header{}, err
		}
		if _, err := g.bytes(uint64(nDims) * 8); err != nil {
			return Header{}, err
		}
		if _, err := g.u32(); err != nil { // type
			return Header{}, err
		}
		off, err := g.u64()
		if err != nil {
			return Header{}, err
		}
		h.tensors = append(h.tensors, tensorEntry{name: name, offset: off})
	}
	h.dataStart = int64((cr.n + alignment - 1) / alignment * alignment)
	return h, nil
}

type countingReader struct {
	r io.Reader
	n uint64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += uint64(n)
	return n, err
}

type ggufReader struct{ r io.Reader }

// arrayLen is what value returns for an array: how many elements it had, which
// is all a shape needs of one (the vocabulary's size).
type arrayLen uint64

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
		return arrayLen(n), nil
	case 10, 11: // uint64, int64
		v, err := g.u64()
		return v, err
	case 12: // float64
		_, err := g.bytes(8)
		return nil, err
	}
	return nil, fmt.Errorf("gguf: unknown value type %d", typ)
}
