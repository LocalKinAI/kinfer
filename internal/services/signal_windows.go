//go:build windows

package services

import (
	"os"
	"syscall"
)

// Windows has no process groups to signal here: a service's own process is
// ended, and its children are left to it.
func groupAttr() *syscall.SysProcAttr   { return nil }
func sessionAttr() *syscall.SysProcAttr { return nil }

func signalGroup(pid int, now bool) { killNow(pid) }
func terminate(pid int)             { killNow(pid) }

func killNow(pid int) {
	if p, err := os.FindProcess(pid); err == nil {
		p.Kill()
	}
}

func alive(pid int) bool {
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	p.Release()
	return true
}

func pgidOf(pid int) int { return pid }
