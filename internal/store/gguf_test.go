package store

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// The hybrid arithmetic, checked against what llama.cpp printed while loading
// the 35B with 16 sequences: "R (f32): 45.00 MiB, S (f32): 960.00 MiB" for
// the recurrent state, and a KV cache of 20 KiB per token. Get either wrong
// and serve sizes itself to a context that refuses real prompts.
func TestShapeSizesAHybridLikeLlamaCppDoes(t *testing.T) {
	sh := Shape{Layers: 40, HeadsKV: 2, Heads: 16, EmbedDim: 4096, KeyLen: 256, ValLen: 256,
		AttnInterval: 4, SSMConv: 4, SSMState: 128, SSMGroups: 16, SSMInner: 4096}
	if got := sh.AttnLayers(); got != 10 {
		t.Errorf("AttnLayers = %d, want 10 of 40", got)
	}
	if got, want := sh.CacheBytes(1), int64(20<<10); got != want {
		t.Errorf("CacheBytes(1) = %d, want %d (10 layers x 2 heads x 512 x 2 bytes)", got, want)
	}
	per := sh.RecurrentBytesPerSeq()
	if got, want := 16*per, int64((45+960)<<20); got != want {
		t.Errorf("16 sequences of recurrent state = %d, want %d (45 + 960 MiB)", got, want)
	}
	// A plain transformer has no recurrent state and every layer attends.
	dense := Shape{Layers: 24, HeadsKV: 2, Heads: 14, EmbedDim: 896}
	if dense.RecurrentBytesPerSeq() != 0 || dense.AttnLayers() != 24 {
		t.Errorf("dense model: recurrent %d, attending %d", dense.RecurrentBytesPerSeq(), dense.AttnLayers())
	}
	if got, want := dense.CacheBytes(1), int64(24*2*64*2*2); got != want {
		t.Errorf("dense CacheBytes(1) = %d, want %d", got, want)
	}
}

// ggufFile writes a GGUF header: string and uint16 keys, and one-dimensional
// tensors at the given offsets. No tensor data — offsets are all Lazy reads.
func ggufFile(kvs map[string]any, tensors []tensorEntry) []byte {
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(len(tensors)))
	w(uint64(len(kvs)))
	keys := make([]string, 0, len(kvs))
	for k := range kvs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		str(k)
		switch v := kvs[k].(type) {
		case string:
			w(uint32(8))
			str(v)
		case uint16:
			w(uint32(2))
			w(v)
		case []string:
			w(uint32(9))
			w(uint32(8))
			w(uint64(len(v)))
			for _, s := range v {
				str(s)
			}
		}
	}
	for _, t := range tensors {
		str(t.name)
		w(uint32(1))
		w(uint64(1))
		w(uint32(0))
		w(t.offset)
	}
	return b.Bytes()
}

const gib = int64(1) << 30

func TestLazyCountsOnlyWhatLlamaCppLeavesOnDisk(t *testing.T) {
	big := []tensorEntry{{"blk.0.ffn_up.weight", 0}, {lazyTensor, 1 << 20}, {"output.weight", 1<<20 + 5<<30}}
	small := []tensorEntry{{lazyTensor, 0}, {"output.weight", 4 << 30}} // exactly 4 GiB: loaded whole

	for _, c := range []struct {
		arch    string
		tensors []tensorEntry
		want    int64
	}{
		{"qwen4exp", big, 5 * gib},
		{"deepseek41", big, 5 * gib},
		{"gemma4", big, 5 * gib},
		{"gemma3n", big, 0}, // has the tensor, but its loader does not mark it
		{"qwen4exp", small, 0},
	} {
		raw := ggufFile(map[string]any{"general.architecture": c.arch}, c.tensors)
		h, err := ReadHeader(bytes.NewReader(raw), int64(len(raw)))
		if err != nil {
			t.Fatal(err)
		}
		if h.Shape.Arch != c.arch {
			t.Errorf("arch %q, want %q", h.Shape.Arch, c.arch)
		}
		if got := h.Lazy(c.arch); got != c.want {
			t.Errorf("%s: Lazy = %d, want %d", c.arch, got, c.want)
		}
	}

	// The table last in its file runs to the end of the data section.
	raw := ggufFile(map[string]any{"general.architecture": "qwen4exp"},
		[]tensorEntry{{"blk.0.ffn_up.weight", 0}, {lazyTensor, 64}})
	h, err := ReadHeader(bytes.NewReader(raw), 0)
	if err != nil {
		t.Fatal(err)
	}
	h.fileSize = h.dataStart + 64 + 6*gib
	if got := h.Lazy("qwen4exp"); got != 6*gib {
		t.Errorf("last tensor: Lazy = %d, want %d", got, 6*gib)
	}
}

// A split model's table sits in a later file, which names no architecture of
// its own — and in ISTA's layout that file is a link to a shared one.
func TestReadShapeTotalsLazyBytesAcrossShards(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, raw []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, raw, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	first := write("m-00001-of-00003.gguf", ggufFile(map[string]any{
		"general.architecture": "deepseek41", "split.count": uint16(3),
	}, nil))
	write("m-00002-of-00003.gguf", ggufFile(map[string]any{"split.count": uint16(3)},
		[]tensorEntry{{"blk.0.ffn_up_exps.weight", 0}, {"blk.1.ffn_up_exps.weight", 30 << 30}}))
	shared := write("engram.gguf", ggufFile(map[string]any{"split.count": uint16(3)},
		[]tensorEntry{{lazyTensor, 0}, {"blk.1.engram_kv.weight", 103 << 30}}))
	if err := os.Symlink(shared, filepath.Join(dir, "m-00003-of-00003.gguf")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	sh, err := ReadShape(first)
	if err != nil {
		t.Fatal(err)
	}
	if sh.Arch != "deepseek41" || sh.LazyBytes != 103*gib {
		t.Errorf("arch %q, LazyBytes %d; want deepseek41, %d", sh.Arch, sh.LazyBytes, 103*gib)
	}

	// An architecture that reads nothing lazily has nothing lazy in any file.
	plain := write("p-00001-of-00002.gguf", ggufFile(map[string]any{
		"general.architecture": "llama", "split.count": uint16(2)}, nil))
	if sh, err := ReadShape(plain); err != nil || sh.LazyBytes != 0 {
		t.Errorf("plain split model: %d, %v", sh.LazyBytes, err)
	}
}

// Placement needs what a model holds and, by layer, how much of it is routed
// experts — every shard counted, and nothing at all when one is missing: a
// model counted in part would look smaller than it is.
func TestReadShapeCountsWeightsAndExpertsByLayer(t *testing.T) {
	const mib = 1 << 20
	dir := t.TempDir()
	// write pads a header with dataLen bytes of tensor data, so the last
	// tensor's size is the data section's end less its offset.
	write := func(name string, raw []byte, dataLen int64) string {
		data := make([]byte, (len(raw)+31)/32*32-len(raw)+int(dataLen))
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, append(raw, data...), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	first := write("x-00001-of-00002.gguf", ggufFile(map[string]any{
		"general.architecture": "qwen3moe", "split.count": uint16(2),
		"tokenizer.ggml.tokens": []string{"a", "b", "c"},
	}, []tensorEntry{
		{"token_embd.weight", 0},
		{"blk.0.ffn_gate_exps.weight", 1 * mib},
		{"blk.0.attn_q.weight", 3 * mib},
	}), 4*mib)
	second := write("x-00002-of-00002.gguf", ggufFile(map[string]any{"split.count": uint16(2)},
		[]tensorEntry{
			{"blk.1.ffn_up_exps.weight", 0},
			{"blk.1.ffn_down_exps.weight", 4 * mib},
			{"blk.1.ffn_gate_inp.weight", 6 * mib}, // the router: every token uses it
			{"output.weight", 7 * mib},
		}), 8*mib)

	sh, err := ReadShape(first)
	if err != nil {
		t.Fatal(err)
	}
	if sh.Vocab != 3 {
		t.Errorf("Vocab = %d, want 3", sh.Vocab)
	}
	if sh.WeightBytes != 12*mib {
		t.Errorf("WeightBytes = %d MiB, want 12 (4 in the first file, 8 in the second)", sh.WeightBytes/mib)
	}
	if want := []int64{2 * mib, 6 * mib}; len(sh.ExpertBytes) != 2 || sh.ExpertBytes[0] != want[0] || sh.ExpertBytes[1] != want[1] {
		t.Errorf("ExpertBytes = %v, want %v", sh.ExpertBytes, want)
	}

	if err := os.Remove(second); err != nil {
		t.Fatal(err)
	}
	if sh, err := ReadShape(first); err != nil || sh.WeightBytes != 0 || sh.ExpertBytes != nil {
		t.Errorf("a shard missing: WeightBytes %d, ExpertBytes %v, %v; want nothing counted", sh.WeightBytes, sh.ExpertBytes, err)
	}
}
