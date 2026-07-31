package engine

import gollama "github.com/dianlight/gollama.cpp"

// setContextSize works around a struct-layout mismatch in gollama.cpp v0.2.2.
//
// llama.cpp removed `seed` from llama_context_params (sampling owns the seed
// now), but gollama's Go mirror still declares Seed as the first field. Every
// field after it therefore lands one slot early when the struct crosses into C:
// a value assigned to NCtx arrives as n_batch, NBatch arrives as n_ubatch, and
// so on. Nothing errors — the parameters simply do not take effect.
//
// Demonstrated directly:
//
//	cp.Seed = 8192; cp.NCtx = 111  →  llama_context: n_ctx = 8192
//
// The consequence was not academic. Serving a LocalKin soul failed with
// "prompt is 8267 tokens but the context holds 4096" no matter what -ctx said,
// because the real context stayed at llama.cpp's 512-token default.
//
// So the size goes into the field that actually reaches n_ctx. This is a
// workaround, not a design: it should be replaced by binding
// llama_init_from_model with a correct struct, and it MUST be revisited when
// gollama is upgraded — a fixed upstream would make this write the seed
// instead, silently restoring the original bug. Open() verifies the resulting
// context size for exactly that reason.
func setContextSize(cp *gollama.LlamaContextParams, n uint32) {
	cp.Seed = n
	cp.NCtx = n // harmless when the layout is right, and correct if it is fixed
}
