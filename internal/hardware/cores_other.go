//go:build !linux && !windows

package hardware

import "runtime"

func physicalCores() int { return runtime.NumCPU() }
