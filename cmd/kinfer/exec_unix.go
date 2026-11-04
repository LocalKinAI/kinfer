//go:build !windows

package main

import "syscall"

// execAgent becomes the agent: its terminal, its signals and its exit status
// are the agent's own, with no kinfer process left in between.
func execAgent(bin string, argv, env []string) error {
	return syscall.Exec(bin, argv, env)
}
