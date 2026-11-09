//go:build windows

package hardware

import (
	"syscall"
	"unsafe"
)

// memoryStatusEx mirrors MEMORYSTATUSEX.
type memoryStatusEx struct {
	length, memoryLoad uint32
	totalPhys          uint64
	availPhys          uint64
	totalPageFile      uint64
	availPageFile      uint64
	totalVirtual       uint64
	availVirtual       uint64
	availExtended      uint64
}

var globalMemoryStatusEx = syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

// windowsRAM is the installed memory in bytes, or 0 if it cannot be read.
func windowsRAM() int64 {
	s := memoryStatusEx{length: uint32(unsafe.Sizeof(memoryStatusEx{}))}
	if ok, _, _ := globalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&s))); ok == 0 {
		return 0
	}
	return int64(s.totalPhys)
}
