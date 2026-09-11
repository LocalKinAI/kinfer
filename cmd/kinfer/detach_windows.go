//go:build windows

package main

import "os/exec"

// detach is a no-op on Windows: a child started without a console window
// already outlives its parent shell.
func detach(c *exec.Cmd) {}
