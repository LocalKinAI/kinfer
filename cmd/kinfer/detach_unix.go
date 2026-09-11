//go:build !windows

package main

import (
	"os/exec"
	"syscall"
)

// detach puts the daemon in its own session so it survives the terminal that
// started it.
func detach(c *exec.Cmd) {
	c.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}
