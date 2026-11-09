//go:build !windows

package main

import (
	"os"
	"syscall"
)

// terminate asks a kinfer process to stop, which it does once its requests are
// done.
func terminate(p *os.Process) error { return p.Signal(syscall.SIGTERM) }

// execAgent becomes the agent: its terminal, its signals and its exit status
// are the agent's own, with no kinfer process left in between.
func execAgent(bin string, argv, env []string) error {
	return syscall.Exec(bin, argv, env)
}
