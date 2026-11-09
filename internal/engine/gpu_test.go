package engine

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGPUFitsLeavesRoomForTheContext(t *testing.T) {
	const gib = 1 << 30
	for _, c := range []struct {
		weights, free uint64
		want          bool
	}{
		{4 * gib, 8 * gib, true},
		{6 * gib, 8 * gib, true},
		{7 * gib, 8 * gib, false},
		{20 * gib, 8 * gib, false},
	} {
		if got := gpuFits(c.weights, c.free); got != c.want {
			t.Errorf("gpuFits(%d GiB, %d GiB) = %v, want %v", c.weights/gib, c.free/gib, got, c.want)
		}
	}
}

func TestProbeRememberedUnlessAskedAfresh(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "backend")
	os.WriteFile(filepath.Join(dir, "probe.json"), []byte(`{"ok":true,"devices":[{"name":"Vulkan0","type":"GPU"},{"name":"CPU","type":"CPU"}]}`), 0o644)

	p := probeGPU(path, false)
	if !p.OK || len(p.GPUs()) != 1 || p.GPUs()[0].Name != "Vulkan0" {
		t.Errorf("remembered probe = %+v", p)
	}
}
