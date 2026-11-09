package llama

import (
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
)

// TestSamplerFloatsCrossTheCall sends float arguments into llama.cpp and reads
// what it did with them from the logits: Windows x86-64 passes floats and
// integers by position, every other platform in separate register files.
func TestSamplerFloatsCrossTheCall(t *testing.T) {
	dir, err := nativelib.Prepare()
	if err != nil {
		t.Skipf("no native libraries in this build: %v", err)
	}
	if err := Bind(dir); err != nil {
		t.Fatal(err)
	}
	lib, err := dl.Open(filepath.Join(dir, libName()))
	if err != nil {
		t.Fatal(err)
	}
	var apply func(s Sampler, cur *TokenDataArray)
	if err := register(&apply, lib, "llama_sampler_apply"); err != nil {
		t.Fatal(err)
	}

	near := func(got, want float32) bool { return got > want-1e-3 && got < want+1e-3 }

	penalties := SamplerPenalties(3, 64, 1.25, 0, 0)
	SamplerAccept(penalties, 1)
	data := []TokenData{{ID: 0, Logit: 1}, {ID: 1, Logit: 2}, {ID: 2, Logit: -3}}
	apply(penalties, &TokenDataArray{Data: &data[0], Size: 3, Selected: -1})
	SamplerFree(penalties)
	if !near(data[1].Logit, 1.6) || !near(data[0].Logit, 1) || !near(data[2].Logit, -3) {
		t.Errorf("after a penalty of 1.25 on token 1: %+v, want its logit 1.6 and the others unchanged", data)
	}

	temp := SamplerTemp(0.5)
	data = []TokenData{{ID: 0, Logit: 1}, {ID: 1, Logit: 2}}
	apply(temp, &TokenDataArray{Data: &data[0], Size: 2, Selected: -1})
	SamplerFree(temp)
	if !near(data[0].Logit, 2) || !near(data[1].Logit, 4) {
		t.Errorf("after a temperature of 0.5: %+v, want logits 2 and 4", data)
	}
}

// TestStructsCrossTheCall passes and returns llama.cpp's structs by value, which
// purego does differently on each system. It loads no model: the defaults come
// back with the values llama.cpp's source gives them, and a load of a file that
// is not there carries an 80-byte struct into C and returns nothing.
func TestStructsCrossTheCall(t *testing.T) {
	dir, err := nativelib.Prepare()
	if err != nil {
		t.Skipf("no native libraries in this build: %v", err)
	}
	if err := Bind(dir); err != nil {
		t.Fatal(err)
	}

	cp := DefaultContextParams()
	if cp.NCtx != 512 || cp.NBatch != 2048 || cp.NUbatch != 512 || cp.NSeqMax != 1 {
		t.Errorf("context defaults = n_ctx %d, n_batch %d, n_ubatch %d, n_seq_max %d; want 512, 2048, 512, 1",
			cp.NCtx, cp.NBatch, cp.NUbatch, cp.NSeqMax)
	}

	mp := DefaultModelParams()
	if mp.NGpuLayers != -1 {
		t.Errorf("n_gpu_layers default = %d, want -1", mp.NGpuLayers)
	}

	tokens := []Token{1, 2, 3}
	b := BatchGetOne(tokens)
	if b.NTokens != 3 || b.Token != uintptr(unsafe.Pointer(&tokens[0])) {
		t.Errorf("batch = %d tokens at %#x, want 3 at %p", b.NTokens, b.Token, &tokens[0])
	}

	chain := samplerChainInit(samplerChainDefault())
	if chain == 0 {
		t.Error("the sampler chain did not initialise")
	} else {
		SamplerFree(chain)
	}

	if m := LoadModel(filepath.Join(t.TempDir(), "absent.gguf"), mp); m != 0 {
		t.Errorf("loading an absent file returned model %#x", uintptr(m))
	}
}
