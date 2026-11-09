//go:build windows

package dl

import (
	"fmt"
	"runtime"
	"syscall"
	"unsafe"
)

// loadWithAlteredSearchPath makes Windows look for a library's own imports in
// that library's directory first, which is where kinfer unpacked its siblings.
// It needs an absolute path.
const loadWithAlteredSearchPath = 0x00000008

// errModNotFound is ERROR_MOD_NOT_FOUND.
const errModNotFound = syscall.Errno(126)

var loadLibraryEx = syscall.NewLazyDLL("kernel32.dll").NewProc("LoadLibraryExW")

// Open loads the library at path and returns its handle.
func Open(path string) (uintptr, error) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, fmt.Errorf("library path %q: %w", path, err)
	}
	h, _, callErr := loadLibraryEx.Call(uintptr(unsafe.Pointer(p)), 0, loadWithAlteredSearchPath)
	if h == 0 {
		if callErr == errModNotFound {
			// The library is there, since kinfer just unpacked it, so what is
			// missing is something it imports: llama.cpp needs this runtime.
			arch := map[string]string{"amd64": "x64", "arm64": "arm64"}[runtime.GOARCH]
			return 0, fmt.Errorf("LoadLibraryEx %s: %w: install the Microsoft Visual C++ runtime: https://aka.ms/vs/17/release/vc_redist.%s.exe", path, callErr, arch)
		}
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
