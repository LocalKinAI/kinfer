package llama

import (
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/LocalKinAI/kinfer/internal/nativelib"
)

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
