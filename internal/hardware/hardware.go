// Package hardware inspects the machine and recommends what will actually run
// on it.
//
// Picking a local model is guesswork for most people: quantisation names carry
// no size information, "7B" says nothing about RAM, and the failure mode is
// either an out-of-memory kill or swapping so heavy the machine becomes unusable.
// The runtime knows the answer, so it should say it.
package hardware

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// Info describes the machine.
type Info struct {
	OS      string
	Arch    string
	Cores   int
	RAM     int64  // bytes; 0 when it could not be determined
	GPU     string // human-readable accelerator description
	Metal   bool   // Apple GPU with unified memory
	CUDA    bool
	Unified bool // GPU shares system RAM (Apple Silicon)
}

// Detect inspects the current machine.
func Detect() Info {
	i := Info{
		OS:    runtime.GOOS,
		Arch:  runtime.GOARCH,
		Cores: runtime.NumCPU(),
		RAM:   totalRAM(),
	}

	switch {
	case runtime.GOOS == "darwin" && runtime.GOARCH == "arm64":
		// Every Apple Silicon Mac has a Metal GPU sharing system memory, so the
		// whole of RAM is potentially VRAM — the reason a laptop can run models
		// that would need a dedicated card elsewhere.
		i.Metal, i.Unified = true, true
		i.GPU = "Apple GPU (Metal, unified memory)"
	case runtime.GOOS == "darwin":
		i.Metal = true
		i.GPU = "Metal (Intel Mac)"
	case hasNvidia():
		i.CUDA = true
		i.GPU = "NVIDIA GPU (CUDA)"
	default:
		i.GPU = "none detected — CPU only"
	}

	return i
}

// totalRAM returns installed memory in bytes, or 0 if it cannot be read.
func totalRAM() int64 {
	switch runtime.GOOS {
	case "darwin":
		// Shelling out to sysctl rather than binding it: the standard library
		// offers only SysctlUint32, which overflows on any machine with more
		// than 4 GB, and golang.org/x/sys would be the first dependency added
		// purely for one number. `sysctl` ships with every macOS install, and
		// this runs once per `kinfer fit`.
		out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
		if err != nil {
			return 0
		}
		n, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
		if err != nil {
			return 0
		}
		return n
	case "linux":
		f, err := os.Open("/proc/meminfo")
		if err != nil {
			return 0
		}
		defer f.Close()

		sc := bufio.NewScanner(f)
		for sc.Scan() {
			line := sc.Text()
			if !strings.HasPrefix(line, "MemTotal:") {
				continue
			}
			fields := strings.Fields(line)
			if len(fields) < 2 {
				return 0
			}
			kb, err := strconv.ParseInt(fields[1], 10, 64)
			if err != nil {
				return 0
			}
			return kb * 1024
		}
	}
	return 0
}

func hasNvidia() bool {
	// Presence of the device node is a cheaper and more reliable signal than
	// shelling out to nvidia-smi, which may not be on PATH.
	_, err := os.Stat("/dev/nvidiactl")
	return err == nil
}

// String renders the machine for display.
func (i Info) String() string {
	ram := "unknown RAM"
	if i.RAM > 0 {
		ram = fmt.Sprintf("%.0f GB RAM", float64(i.RAM)/(1<<30))
	}
	return fmt.Sprintf("%s/%s · %d cores · %s · %s", i.OS, i.Arch, i.Cores, ram, i.GPU)
}
