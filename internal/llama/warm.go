package llama

import (
	"os"

	"github.com/ebitengine/purego"
)

// ─── keeping the GPU's clock up ──────────────────────────────────────────────

// macOS lowers an idle GPU's clock within milliseconds, and a GPU that waits
// between graphs pays for it on the next one. A model split across machines
// waits on every token — each GPU while the other computes its layers — and on
// an M3 Ultra and an M5 Max the same 8 layers went from 6 ms to 11–14 ms a token
// over a reply's first 20 tokens. kinfer's llama.cpp patch (patches/llama.cpp)
// can keep a tiny kernel running on the GPU for a while after each graph, so
// the next one finds it at speed: split decode went from 25 to 38 tok/s.

// WarmWindowMs is how long kinfer keeps a GPU warm after a graph: the gaps
// between a reply's tokens are milliseconds, and a second after the reply the
// GPU is let go.
const WarmWindowMs = 1000

var metalSetWarm func(windowMs int32)

// bindWarm resolves the keep-warm switch in libggml-metal. It is kinfer's own
// addition, so a build from a llama.cpp release lacks it and runs without.
func bindWarm(dir string) {
	lib, err := purego.Dlopen(siblingLib(dir, "ggml-metal"), purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	if register(&metalSetWarm, lib, "ggml_backend_metal_set_warm") != nil {
		metalSetWarm = nil
	}
}

// KeepGPUWarm keeps the GPU's clock up for windowMs after each graph it
// computes; 0 turns that off. It reports whether this build can. Someone who
// set GGML_METAL_WARM_MS chose a window themselves, and it is left alone.
func KeepGPUWarm(windowMs int) bool {
	if metalSetWarm == nil {
		return false
	}
	if os.Getenv("GGML_METAL_WARM_MS") == "" {
		metalSetWarm(int32(windowMs))
	}
	return true
}
