package main

import "syscall"

func getTermios(fd int) (*syscall.Termios, error) {
	var t syscall.Termios
	return &t, ioctlTermios(fd, syscall.TIOCGETA, &t)
}

func setTermios(fd int, t *syscall.Termios) error {
	return ioctlTermios(fd, syscall.TIOCSETA, t)
}
