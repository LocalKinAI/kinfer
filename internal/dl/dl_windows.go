//go:build windows

package dl

import (
	"fmt"
	"syscall"
	"unsafe"
)

// loadWithAlteredSearchPath makes Windows look for a library's own imports in
// that library's directory first, which is where kinfer unpacked its siblings.
// It needs an absolute path.
const loadWithAlteredSearchPath = 0x00000008

var loadLibraryEx = syscall.NewLazyDLL("kernel32.dll").NewProc("LoadLibraryExW")

// Open loads the library at path and returns its handle.
func Open(path string) (uintptr, error) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, fmt.Errorf("library path %q: %w", path, err)
	}
	h, _, callErr := loadLibraryEx.Call(uintptr(unsafe.Pointer(p)), 0, loadWithAlteredSearchPath)
	if h == 0 {
		return 0, fmt.Errorf("LoadLibraryEx %s: %w", path, callErr)
	}
	return h, nil
}

// Sym returns the address of name in the library, or an error when it has none.
func Sym(lib uintptr, name string) (uintptr, error) {
	addr, err := syscall.GetProcAddress(syscall.Handle(lib), name)
	if err != nil {
		return 0, fmt.Errorf("GetProcAddress %s: %w", name, err)
	}
	return addr, nil
}
