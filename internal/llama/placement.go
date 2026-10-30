package llama

import "github.com/ebitengine/purego"

// ─── where weights live ─────────────────────────────────────────────────────

// TensorBuftOverride mirrors struct llama_model_tensor_buft_override: tensors
// whose names match the regex pattern go to that buffer type. An array of them
// ends with a zero entry.
type TensorBuftOverride struct {
	Pattern *byte
	Buft    uintptr
}

var cpuBufferType func() uintptr

// bindCPU resolves ggml's CPU buffer type, in libggml-cpu: the "host memory"
// a weight can be told to stay in instead of the GPU's. Best-effort, like the
// rest of the device bindings.
func bindCPU(dir string) {
	lib, err := purego.Dlopen(siblingLib(dir, "ggml-cpu"), purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	if err := register(&cpuBufferType, lib, "ggml_backend_cpu_buffer_type"); err != nil {
		cpuBufferType = nil
	}
}

// CPUBufferType is ggml's host-memory buffer type, or 0 when this build has
// none. A weight placed there and loaded with mmap stays in the file's pages:
// read from the SSD as it is touched, kept in the page cache while it is used.
func CPUBufferType() uintptr {
	if cpuBufferType == nil {
		return 0
	}
	return cpuBufferType()
}
