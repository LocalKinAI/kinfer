//go:build darwin || linux

package main

import (
	"syscall"
	"unsafe"
)

// isTerminal says whether fd is a terminal.
func isTerminal(fd int) bool {
	_, err := getTermios(fd)
	return err == nil
}

// makeRaw hands the terminal on fd to the line editor: keys arrive as typed,
// unechoed, Ctrl-C and Ctrl-Z among them, and output is processed as before.
// It returns how to put the terminal back.
func makeRaw(fd int) (func(), error) {
	old, err := getTermios(fd)
	if err != nil {
		return nil, err
	}
	raw := *old
	raw.Iflag &^= syscall.ICRNL | syscall.INLCR | syscall.IGNCR | syscall.IXON | syscall.ISTRIP
	raw.Lflag &^= syscall.ICANON | syscall.ECHO | syscall.IEXTEN | syscall.ISIG
	raw.Cc[syscall.VMIN] = 1
	raw.Cc[syscall.VTIME] = 0
	if err := setTermios(fd, &raw); err != nil {
		return nil, err
	}
	return func() { _ = setTermios(fd, old) }, nil
}

// termWidth is the terminal's width in columns, 80 when it cannot say.
func termWidth(fd int) int {
	var ws struct{ Row, Col, X, Y uint16 }
	_, _, e := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), syscall.TIOCGWINSZ, uintptr(unsafe.Pointer(&ws)))
	if e != 0 || ws.Col == 0 {
		return 80
	}
	return int(ws.Col)
}

func ioctlTermios(fd int, req uintptr, t *syscall.Termios) error {
	_, _, e := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), req, uintptr(unsafe.Pointer(t)))
	if e != 0 {
		return e
	}
	return nil
}
