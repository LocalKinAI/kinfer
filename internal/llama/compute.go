package llama

import (
	"runtime"
	"sync"
)

// OpenMP keeps a team of worker threads for each operating-system thread that
// starts a parallel region, and a goroutine moves between those threads as it
// pleases. Called from any of them, every new one added a team. Once the teams
// together outnumbered the processors, libgomp stopped spinning between passes:
// on Linux with 8 threads, decoding fell from 160 tok/s to 45 within a few
// replies and stayed there.
var computeThread struct {
	once sync.Once
	work chan func()
}

// onComputeThread runs f on the one operating-system thread that starts all of
// llama.cpp's work on the CPU, and waits for it. On a Mac, Metal does that work
// and f simply runs here.
func onComputeThread(f func()) {
	if runtime.GOOS == "darwin" {
		f()
		return
	}
	computeThread.once.Do(func() {
		computeThread.work = make(chan func())
		go func() {
			runtime.LockOSThread()
			for g := range computeThread.work {
				g()
			}
		}()
	})
	done := make(chan struct{})
	computeThread.work <- func() {
		f()
		close(done)
	}
	<-done
}
