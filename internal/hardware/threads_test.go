package hardware

import (
	"runtime"
	"testing"
)

func TestComputeThreadsIsWithinTheMachine(t *testing.T) {
	n := ComputeThreads()
	if n < 1 || n > runtime.GOMAXPROCS(0) {
		t.Errorf("ComputeThreads() = %d, want 1 to %d", n, runtime.GOMAXPROCS(0))
	}
	if c := physicalCores(); c < 1 || c > runtime.NumCPU() {
		t.Errorf("physicalCores() = %d, want 1 to %d", c, runtime.NumCPU())
	}
}
