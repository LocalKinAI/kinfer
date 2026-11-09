package engine

import (
	"runtime"

	"github.com/LocalKinAI/kinfer/internal/hardware"
	"github.com/LocalKinAI/kinfer/internal/llama"
)

// useTheCores gives a context a thread per core where the CPU does the work.
// llama.cpp's default is 4, which costs a Mac nothing because Metal does the
// arithmetic, and leaves most of any other machine idle.
func useTheCores(cp *llama.ContextParams) {
	if runtime.GOOS == "darwin" {
		return
	}
	n := int32(hardware.ComputeThreads())
	cp.NThreads, cp.NThreadsBatch = n, n
}
