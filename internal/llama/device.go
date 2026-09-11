package llama

import (
	"fmt"
	"path/filepath"

	"github.com/ebitengine/purego"
)

// Device memory, from ggml's backend registry.
//
// This is the only way to find out how much room a machine actually has. On
// Apple Silicon the GPU has no memory of its own — it shares the system's —
// but Metal still publishes a working-set budget (recommendedMaxWorkingSetSize,
// about 75% of RAM), and a command buffer that pushes past it fails with
// kIOGPUCommandBufferCallbackErrorOutOfMemory rather than falling back to host
// memory. A 96 GB Mac Studio therefore has roughly 81.5 GiB to spend, not 96,
// and a 73.5 GiB model plus its KV cache sits close enough to that line that
// whether it survives depends on what else is running.
//
// kinfer asks before it promises: see engine.Open.
type Device struct {
	Name  string
	Type  DeviceType
	Free  uint64 // bytes the backend says are still available
	Total uint64 // the budget, not the machine's RAM
}

// DeviceType mirrors enum ggml_backend_dev_type.
type DeviceType int32

const (
	DeviceCPU DeviceType = iota
	DeviceGPU
	DeviceAccel
)

func (t DeviceType) String() string {
	switch t {
	case DeviceCPU:
		return "CPU"
	case DeviceGPU:
		return "GPU"
	case DeviceAccel:
		return "accel"
	default:
		return fmt.Sprintf("type(%d)", int32(t))
	}
}

var (
	devCount  func() uint64
	devGet    func(uint64) uintptr
	devMemory func(uintptr, *uint64, *uint64)
	devName   func(uintptr) *byte
	devType   func(uintptr) int32
)

// bindGGML resolves the device-registry entry points, which live in libggml
// rather than libllama.
//
// Failure is not fatal: everything here is reporting, and a build that lacks
// these symbols should still serve. Devices() returns nothing instead.
func bindGGML(dir string) {
	lib, err := purego.Dlopen(ggmlName(dir), purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&devCount, "ggml_backend_dev_count"},
		{&devGet, "ggml_backend_dev_get"},
		{&devMemory, "ggml_backend_dev_memory"},
		{&devName, "ggml_backend_dev_name"},
		{&devType, "ggml_backend_dev_type"},
	} {
		if err := register(b.fn, lib, b.name); err != nil {
			devCount = nil // all or nothing; a half-bound set is a crash waiting
			return
		}
	}
}

// Devices lists the backends ggml registered, with each one's current memory.
//
// Call it after Bind. The numbers move as models load, which is the point:
// taking one reading before a load and one after is how kinfer learns what a
// model actually cost rather than estimating it from the file size.
func Devices() []Device {
	if devCount == nil {
		return nil
	}
	n := devCount()
	out := make([]Device, 0, n)
	for i := uint64(0); i < n; i++ {
		d := devGet(i)
		if d == 0 {
			continue
		}
		var free, total uint64
		devMemory(d, &free, &total)
		out = append(out, Device{
			Name:  goString(devName(d)),
			Type:  DeviceType(devType(d)),
			Free:  free,
			Total: total,
		})
	}
	return out
}

// Accelerator returns the device kinfer will actually compute on — the first
// GPU, or the first accelerator when there is no GPU. ok is false on a machine
// with neither, where there is no budget to police.
func Accelerator() (Device, bool) {
	devs := Devices()
	for _, want := range []DeviceType{DeviceGPU, DeviceAccel} {
		for _, d := range devs {
			if d.Type == want && d.Total > 0 {
				return d, true
			}
		}
	}
	return Device{}, false
}

func ggmlName(dir string) string {
	switch libName() {
	case "llama.dll":
		return filepath.Join(dir, "ggml.dll")
	case "libllama.so.0":
		return filepath.Join(dir, "libggml.so.0")
	default:
		return filepath.Join(dir, "libggml.0.dylib")
	}
}
