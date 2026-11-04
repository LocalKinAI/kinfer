package engine

import (
	"testing"

	"github.com/LocalKinAI/kinfer/internal/store"
)

func TestCacheQ8(t *testing.T) {
	qwen := store.Shape{Arch: "qwen3", Layers: 28, HeadsKV: 8, Heads: 16, EmbedDim: 2048, KeyLen: 128, ValLen: 128}
	next := flashNext
	next.Arch = "qwen4exp"
	cases := []struct {
		name      string
		sh        store.Shape
		cacheType string
		want      bool
	}{
		{"a model it suits", qwen, "q8_0", true},
		{"Flash-Next, whose heads are 256", next, "q8_0", true},
		{"asked for f16", qwen, "f16", false},
		{"not asked", qwen, "", false},
		{"no header read", store.Shape{}, "q8_0", false},
		// 2560 / 32 heads: 80, which q8_0's blocks of 32 do not divide.
		{"heads of 80", store.Shape{Arch: "phi2", Layers: 32, HeadsKV: 32, Heads: 32, EmbedDim: 2560}, "q8_0", false},
		{"DeepSeek V4.1", store.Shape{Arch: "deepseek41", Layers: 43, HeadsKV: 1, Heads: 64, EmbedDim: 4096, KeyLen: 512, ValLen: 512}, "q8_0", false},
		{"Kimi, on DeepSeek's design", store.Shape{Arch: "deepseek2", Layers: 61, HeadsKV: 1, Heads: 64, EmbedDim: 7168, KeyLen: 576, ValLen: 512}, "q8_0", false},
	}
	for _, c := range cases {
		if got := CacheQ8(c.sh, c.cacheType); got != c.want {
			t.Errorf("%s: CacheQ8 = %v, want %v", c.name, got, c.want)
		}
	}
}

// Sized at q8_0, the cache that held a conversation of 65536 tokens on the
// evening AutoSize was written for holds one twice as long.
func TestAutoSizeGivesAnEightBitCacheTwiceTheContext(t *testing.T) {
	f16, err := AutoSize(moe, 457*GiB/10, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	q8 := moe
	q8.CacheQ8 = true
	got, err := AutoSize(q8, 457*GiB/10, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.Slots != f16.Slots || got.PerSeq != 2*f16.PerSeq {
		t.Errorf("at q8_0, %d slots x %d tokens; at f16 it was %d x %d", got.Slots, got.PerSeq, f16.Slots, f16.PerSeq)
	}
}
