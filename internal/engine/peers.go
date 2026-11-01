package engine

import (
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/link"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// peerSettle is how long deviceList waits after its own check before llama.cpp
// connects to a worker; see there.
const peerSettle = 500 * time.Millisecond

// helloWait is how long a worker has to answer that check.
const helloWait = 2 * time.Second

// findWait is how long kinfer listens for workers answering over the
// Thunderbolt cable; they answered in 1-20 ms there.
const findWait = 300 * time.Millisecond

// findPeers is the workers a model borrows when kinfer was asked to find them
// (-peers auto): none for a model that fits this GPU — a layer behind the
// cable costs every token a round trip, and a worker takes one machine at a
// time, so it is not held for a model that does not need it — and otherwise
// whichever answer over the Thunderbolt cable. With none, the model stays
// here, its experts in host memory as placement decides.
func findPeers(sh store.Shape, bud *budget, opts Options, name string) []string {
	if !bud.known || sh.WeightBytes == 0 {
		return nil
	}
	room := placeRoom(sh, opts.Slots, opts.ContextSize, opts.Batch)
	if uint64(sh.WeightBytes)+room.Reserve <= bud.atStart {
		return nil
	}
	b, err := link.FindBridge()
	if err != nil {
		log.Printf("peers: %s does not fit this GPU, and there is %v; it stays here", name, err)
		return nil
	}
	found, err := link.Find(b, findWait)
	if err != nil || len(found) == 0 {
		log.Printf("peers: %s does not fit this GPU, and no kinfer worker answered over the Thunderbolt cable (%s); it stays here",
			name, b.Name)
		return nil
	}
	log.Printf("peers: %s does not fit this GPU; over the Thunderbolt cable %s answered", name, strings.Join(found, ", "))
	return found
}

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
		// returning an error, and waits forever on a worker busy with another
		// machine, so the worker is asked first: a worker on a laptop that went
		// to sleep, or lent to someone else, is a failed load, not a dead server.
		if err := llama.RPCHello(p, helloWait); err != nil {
			return nil, nil, err
		}
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
