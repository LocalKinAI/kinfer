//go:build linux

package llama

import (
	"sync"
	"syscall"
	"testing"
)

func TestComputeRunsOnOneThread(t *testing.T) {
	var (
		mu   sync.Mutex
		seen = map[int]bool{}
		wg   sync.WaitGroup
	)
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 16; j++ {
				onComputeThread(func() {
					mu.Lock()
					seen[syscall.Gettid()] = true
					mu.Unlock()
				})
			}
		}()
	}
	wg.Wait()
	if len(seen) != 1 {
		t.Errorf("work ran on %d operating-system threads, want 1", len(seen))
	}
}
