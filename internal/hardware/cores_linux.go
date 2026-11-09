//go:build linux

package hardware

import (
	"os"
	"path/filepath"
	"runtime"
)

// physicalCores counts the distinct sets of sibling hardware threads, one set
// to a core.
func physicalCores() int {
	paths, _ := filepath.Glob("/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings")
	cores := make(map[string]bool, len(paths))
	for _, p := range paths {
		if b, err := os.ReadFile(p); err == nil {
			cores[string(b)] = true
		}
	}
	if len(cores) == 0 {
		return runtime.NumCPU()
	}
	return len(cores)
}
