//go:build darwin || linux

package hub

import (
	"os"
	"syscall"
)

// lock holds f for this process until it closes it, so a second pull of the
// same file finds it taken instead of writing into it too.
func lock(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
}

// flush hands what has been written to f to the disk. os.File.Sync on macOS
// also empties the drive's own cache (F_FULLFSYNC), a cost to every program
// writing to that drive, every two seconds; what it adds is safety from a
// power cut, which the sha256 check covers anyway.
func flush(f *os.File) error {
	return syscall.Fsync(int(f.Fd()))
}

// free is how many bytes the disk dir is on has for this user.
func free(dir string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * int64(st.Bsize), nil
}
