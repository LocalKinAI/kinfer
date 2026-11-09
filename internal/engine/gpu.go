package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/LocalKinAI/kinfer/internal/dl"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
)

// Off a Mac the GPU is reached through Vulkan, as an experiment that must never
// cost anyone the CPU path that works: the backend is fetched only where there
// is a driver to talk to, tried first in a process of its own, because a driver
// that crashes takes the process with it, and any model that does not fit or
// does not load runs on the CPU instead. KINFER_GPU=off leaves all of it out.

// GPUProbeArgs is how the running program tries a backend in a process of its
// own: its arguments, before the backend's path. Nil, as in a test binary or a
// program that has no such command, means no GPU backend is attached.
var GPUProbeArgs []string

// ProbeDevice is one device a backend offered.
type ProbeDevice struct {
	Name  string `json:"name"`
	Type  string `json:"type"`
	Total uint64 `json:"total"`
	Free  uint64 `json:"free"`
}

// GPUProbe is what trying a backend found.
type GPUProbe struct {
	OK      bool          `json:"ok"`
	Error   string        `json:"error,omitempty"`
	Devices []ProbeDevice `json:"devices,omitempty"`
}

// GPUs are the probed devices that are graphics processors.
func (p GPUProbe) GPUs() []ProbeDevice {
	var out []ProbeDevice
	for _, d := range p.Devices {
		if d.Type == "GPU" || d.Type == "iGPU" {
			out = append(out, d)
		}
	}
	return out
}

// GPUReport is the state of the GPU path, for kinfer selftest.
type GPUReport struct {
	Loader  bool     // the system has a Vulkan loader
	Backend string   // the backend's path, once fetched
	Problem string   // why it could not be fetched
	Probe   GPUProbe // what trying it found
}

// CheckGPU fetches the backend if need be and tries it afresh, whatever an
// earlier try found: a new driver may have changed the answer.
func CheckGPU() GPUReport {
	var r GPUReport
	if _, err := dl.Open(nativelib.VulkanLoader()); err != nil {
		return r
	}
	r.Loader = true
	path, err := nativelib.VulkanBackend()
	if err != nil {
		r.Problem = err.Error()
		return r
	}
	r.Backend = path
	r.Probe = probeGPU(path, true)
	return r
}

var (
	gpuOnce     sync.Once
	gpuAttached bool
)

// attachGPU adds the Vulkan backend to llama.cpp's devices, once, if this
// machine has a GPU it works on. It says why not only when something failed: no
// driver at all is the ordinary case on a server.
func attachGPU(gpuLayers int) {
	if runtime.GOOS == "darwin" || gpuLayers == 0 || GPUProbeArgs == nil || strings.EqualFold(os.Getenv("KINFER_GPU"), "off") {
		return
	}
	gpuOnce.Do(func() {
		if _, err := dl.Open(nativelib.VulkanLoader()); err != nil {
			return
		}
		path, err := nativelib.VulkanBackend()
		if err != nil {
			log.Printf("gpu: no Vulkan backend (%v); running on the CPU", err)
			return
		}
		p := probeGPU(path, false)
		if !p.OK {
			log.Printf("gpu: the Vulkan backend did not start here (%s); running on the CPU. kinfer selftest tries it again", p.Error)
			return
		}
		gpus := p.GPUs()
		if len(gpus) == 0 {
			return
		}
		if err := llama.LoadBackend(path); err != nil {
			log.Printf("gpu: %v; running on the CPU", err)
			return
		}
		gpuAttached = true
		log.Printf("gpu: Vulkan on %s, experimental; KINFER_GPU=off turns it off", gpus[0].Name)
	})
}

// probeGPU tries the backend in a child process and remembers the answer beside
// it, so that a crash is paid for once. fresh tries again regardless.
func probeGPU(path string, fresh bool) GPUProbe {
	record := filepath.Join(filepath.Dir(path), "probe.json")
	if !fresh {
		var p GPUProbe
		if b, err := os.ReadFile(record); err == nil && json.Unmarshal(b, &p) == nil {
			return p
		}
	}

	p := runProbe(path)
	if b, err := json.Marshal(p); err == nil {
		os.WriteFile(record, b, 0o644)
	}
	return p
}

func runProbe(path string) GPUProbe {
	exe, err := os.Executable()
	if err != nil {
		return GPUProbe{Error: err.Error()}
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	var out bytes.Buffer
	cmd := exec.CommandContext(ctx, exe, append(append([]string{}, GPUProbeArgs...), path)...)
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return GPUProbe{Error: err.Error()}
	}
	var p GPUProbe
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &p); err != nil {
		return GPUProbe{Error: "the probe's answer was not readable: " + err.Error()}
	}
	return p
}

// gpuFits says whether a model's weights, and room for a context beside them,
// fit in what the GPU has free. Off a Mac there is one rule for now: a model
// that fits runs on the GPU, one that does not runs on the CPU. Splitting it
// between the two waits for someone with such a machine to measure it.
func gpuFits(weights, free uint64) bool {
	return weights+thinHeadroom <= free
}

// runsOnGPU decides the layers a model gets off a Mac, and what its memory is
// budgeted against; on a Mac it changes nothing.
func runsOnGPU(name string, weights uint64, layers int, bud *budget) (int, *budget) {
	if runtime.GOOS == "darwin" {
		return layers, bud
	}
	if layers == 0 {
		return 0, &budget{} // whatever GPU there is, this model is not on it
	}
	if bud.known && !gpuFits(weights, bud.atStart) {
		log.Printf("gpu: %s needs %s and the GPU has %s free, with room for its context; it runs on the CPU",
			name, gib(weights), gib(bud.atStart))
		return 0, &budget{}
	}
	return layers, bud
}
