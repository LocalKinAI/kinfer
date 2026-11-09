//go:build windows

package hardware

import (
	"encoding/binary"
	"runtime"
	"syscall"
	"unsafe"
)

var getLogicalProcessorInformationEx = syscall.NewLazyDLL("kernel32.dll").NewProc("GetLogicalProcessorInformationEx")

const relationProcessorCore = 0

// physicalCores counts the records Windows returns for processor cores. Each
// record starts with its relationship and its own size, both 32 bits.
func physicalCores() int {
	var size uint32
	getLogicalProcessorInformationEx.Call(relationProcessorCore, 0, uintptr(unsafe.Pointer(&size)))
	if size == 0 {
		return runtime.NumCPU()
	}
	buf := make([]byte, size)
	ok, _, _ := getLogicalProcessorInformationEx.Call(relationProcessorCore,
		uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size)))
	if ok == 0 {
		return runtime.NumCPU()
	}
	n := 0
	for off := uint32(0); off+8 <= size; n++ {
		step := binary.LittleEndian.Uint32(buf[off+4:])
		if step == 0 {
			break
		}
		off += step
	}
	if n == 0 {
		return runtime.NumCPU()
	}
	return n
}
