package engine

import (
	"sync"

	"github.com/LocalKinAI/kinfer/internal/llama"
)

// The GPU is kept warm between tokens (llama/warm.go) while a model split
// across machines is loaded and no model here keeps experts in host memory.
//
// The MoE cache waits on the GPU several times a layer, and a GPU kept busy by
// anything else makes each wait long: on the MacBook Pro (M5 Max), Flash-Next
// with 27 layers' experts on the host decoded at 1 tok/s instead of 30 with
// the keep-warm kernel running, and the same with it in another process. On
// the box (M3 Ultra) host experts lost nothing to it. A layer split without
// host experts waits on the GPU two or three times a token, and gains a half.

var warm struct {
	sync.Mutex
	split, cache int // models loaded: split across machines; with host experts
}

// warmAdd counts a model in or out (+1 or -1 each) and sets the window.
func warmAdd(split, cache int) {
	warm.Lock()
	defer warm.Unlock()
	warm.split += split
	warm.cache += cache
	llama.KeepGPUWarm(warmWindow(warm.split, warm.cache))
}

// warmWindow is the keep-warm window for the models loaded.
func warmWindow(split, cache int) int {
	if split > 0 && cache == 0 {
		return llama.WarmWindowMs
	}
	return 0
}
