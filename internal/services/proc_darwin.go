//go:build darwin

package services

import (
	"bytes"
	"encoding/binary"
	"errors"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// What macOS says of a process, asked of libSystem directly: its exact
// arguments and environment (KERN_PROCARGS2: ps prints them joined by
// spaces, which a path with a space in it breaks), and its physical
// footprint, which on Apple silicon counts what it holds on the GPU too.

var (
	sysOnce      sync.Once
	sysErr       error
	sysctl       func(mib *int32, n uint32, old unsafe.Pointer, oldLen *uintptr, newp unsafe.Pointer, newLen uintptr) int32
	procPidUsage func(pid int32, flavor int32, buf unsafe.Pointer) int32
)

func bindSys() error {
	sysOnce.Do(func() {
		lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			sysErr = err
			return
		}
		purego.RegisterLibFunc(&sysctl, lib, "sysctl")
		purego.RegisterLibFunc(&procPidUsage, lib, "proc_pid_rusage")
	})
	return sysErr
}

// procArgs is a process's arguments and environment.
func procArgs(pid int) (argv, env []string, err error) {
	if err := bindSys(); err != nil {
		return nil, nil, err
	}
	argmax := [2]int32{1, 8} // CTL_KERN, KERN_ARGMAX
	var max int32
	n := unsafe.Sizeof(max)
	if sysctl(&argmax[0], 2, unsafe.Pointer(&max), &n, nil, 0) != 0 || max <= 0 {
		max = 1 << 20
	}
	buf := make([]byte, max)
	size := uintptr(len(buf))
	mib := [3]int32{1, 49, int32(pid)} // CTL_KERN, KERN_PROCARGS2
	if sysctl(&mib[0], 3, unsafe.Pointer(&buf[0]), &size, nil, 0) != 0 {
		return nil, nil, errors.New("its arguments cannot be read")
	}
	buf = buf[:size]
	if len(buf) < 4 {
		return nil, nil, errors.New("its arguments are empty")
	}
	argc := int(binary.LittleEndian.Uint32(buf))
	rest := buf[4:]
	// The executable's path, then NULs up to the arguments.
	i := bytes.IndexByte(rest, 0)
	if i < 0 {
		return nil, nil, errors.New("its arguments are cut short")
	}
	rest = rest[i:]
	for len(rest) > 0 && rest[0] == 0 {
		rest = rest[1:]
	}
	next := func() (string, bool) {
		j := bytes.IndexByte(rest, 0)
		if j < 0 {
			return "", false
		}
		s := string(rest[:j])
		rest = rest[j+1:]
		return s, true
	}
	for len(argv) < argc {
		s, ok := next()
		if !ok {
			break
		}
		argv = append(argv, s)
	}
	for {
		s, ok := next()
		if !ok || s == "" {
			break
		}
		env = append(env, s)
	}
	return argv, env, nil
}

// footprint is a process's physical footprint in bytes, 0 if unknown.
func footprint(pid int) uint64 {
	if bindSys() != nil {
		return 0
	}
	var buf [64]uint64 // rusage_info_v4 and room to spare
	if procPidUsage(int32(pid), 4, unsafe.Pointer(&buf[0])) != 0 {
		return 0
	}
	// ri_uuid is 16 bytes, then user and system time, idle and interrupt
	// wakeups, pageins, wired and resident size: the footprint is the next.
	return buf[9]
}
