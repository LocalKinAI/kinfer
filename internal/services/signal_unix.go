//go:build !windows

package services

import "syscall"

// A service runs in a process group of its own, so its children go with it;
// one released runs in a session of its own, so it outlives the server.
func groupAttr() *syscall.SysProcAttr   { return &syscall.SysProcAttr{Setpgid: true} }
func sessionAttr() *syscall.SysProcAttr { return &syscall.SysProcAttr{Setsid: true} }

// signalGroup asks the group led by pid to end, or makes it.
func signalGroup(pid int, now bool) {
	sig := syscall.SIGTERM
	if now {
		sig = syscall.SIGKILL
	}
	syscall.Kill(-pid, sig)
}

func terminate(pid int)  { syscall.Kill(pid, syscall.SIGTERM) }
func killNow(pid int)    { syscall.Kill(pid, syscall.SIGKILL) }
func alive(pid int) bool { return syscall.Kill(pid, 0) == nil }
func pgidOf(pid int) int { g, _ := syscall.Getpgid(pid); return g }
