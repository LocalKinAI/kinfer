//go:build windows

package hub

import (
	"os"
	"syscall"
)

const fileShareDelete = 0x4

// openPart opens name.part for reading and writing, creating it if need be,
// and lets it be renamed and removed while open. A pull gives the file its name
// while still holding it, which os.OpenFile's sharing mode refuses on Windows.
func openPart(path string) (*os.File, error) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
	}
	h, err := syscall.CreateFile(p, syscall.GENERIC_READ|syscall.GENERIC_WRITE,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|fileShareDelete,
		nil, syscall.OPEN_ALWAYS, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
	}
	return os.NewFile(uintptr(h), path), nil
}
