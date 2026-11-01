package engine

import (
	"fmt"
	"log"
	"net"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// peerSettle is how long deviceList waits after its own check before llama.cpp
// connects to a worker; see there.
const peerSettle = 500 * time.Millisecond

// deviceList is the device list a model loads onto when it is split across
// machines: every peer's devices, then this machine's GPU, then the zero
// llama.cpp reads as the end — and the split as llama.cpp takes it, one share
// per device in that order. With no peers both are nil and llama.cpp chooses
// on its own, as it always has.
//
// The peers come first because llama.cpp orders them so: the first layers go
// to them and the last, with the output, stay here, where the reply is
// sampled.
func deviceList(peers []string, split []float32) ([]uintptr, []float32, error) {
	if len(peers) == 0 {
		if len(split) > 0 {
			return nil, nil, fmt.Errorf("a split needs peers to split across")
		}
		return nil, nil, nil
	}
	var devs []uintptr
	for _, p := range peers {
		// ggml aborts the whole process when it cannot connect, rather than
		// returning an error, so the connection is tried here first: a worker
		// on a laptop that went to sleep is a failed load, not a dead server.
		c, err := net.DialTimeout("tcp", p, 3*time.Second)
		if err != nil {
			return nil, nil, fmt.Errorf("worker %s is not reachable: %w", p, err)
		}
		c.Close()
		// The worker serves one connection at a time and keeps a queue of
		// one, and connecting again at once was refused or reset over the
		// cable from the MacBook Pro to the box, most loads in a row; half a
		// second later every time.
		time.Sleep(peerSettle)
		d, err := llama.RPCDevices(p)
		if err != nil {
			return nil, nil, err
		}
		devs = append(devs, d...)
	}
	local := llama.LocalGPUs()
	if len(local) == 0 {
		return nil, nil, fmt.Errorf("no GPU here to keep the rest of the model on")
	}
	devs = append(devs, local[0])
	// This GPU waits on every token while the peers compute their layers.
	llama.KeepGPUWarm(llama.WarmWindowMs)

	var shares []float32
	if len(split) > 0 {
		if len(split) != len(devs) {
			return nil, nil, fmt.Errorf("the split names %d shares for %d devices (each peer's, then this machine's)", len(split), len(devs))
		}
		shares = append(make([]float32, 0, len(split)), split...)
	}

	parts := make([]string, len(devs))
	for i, d := range devs {
		free, total := llama.DeviceMemory(d)
		parts[i] = fmt.Sprintf("%s %s free of %s", llama.DeviceName(d), store.HumanSize(int64(free)), store.HumanSize(int64(total)))
		if shares != nil {
			parts[i] += fmt.Sprintf(" (share %g)", shares[i])
		}
	}
	log.Printf("devices: %s — the first layers go to the peers, the rest and the output stay here", strings.Join(parts, ", "))
	return append(devs, 0), shares, nil
}
