package hardware

import "runtime"

// ComputeThreads is how many threads inference on the CPU should use: one per
// physical core, since hyper-threads share a core's arithmetic units and would
// only queue for them, or fewer if this process may use fewer (GOMAXPROCS
// follows a container's CPU quota).
func ComputeThreads() int {
	n := physicalCores()
	if g := runtime.GOMAXPROCS(0); g < n {
		n = g
	}
	if n < 1 {
		n = 1
	}
	return n
}
