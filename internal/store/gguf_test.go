package store

import "testing"

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
