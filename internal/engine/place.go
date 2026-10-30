package engine

import (
	"fmt"
	"log"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/hardware"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// Placing a model the GPU cannot hold.
//
// What does not fit the GPU can wait in host memory, and what does not fit
// there in the file on the SSD behind it — at a usable speed only for a
// mixture of experts. A token uses a few of each layer's experts (10 of
// Flash-Next's 512), so the rest can stay in host memory and a GPU cache keep
// the ones in use, read from the SSD as they are missed (experts.go). A dense
// model reads every weight for every token.
//
// Place decides how many layers' experts leave the GPU and how big the cache
// is, from the model's header and the memory the GPU has free. The operator
// used to, with -host-experts and -moe-cache, found by loading the model again
// and again. On the MacBook Pro (48 GB, 40 GiB of it the GPU's), Flash-Next
// IQ3_S has 51.1 GiB of weights; the best found by hand was 24 of 48 layers'
// experts on the host, 22.7 GiB, and an 8 GiB cache — 36% of them, which
// generation hit 94% of the time, at 31 tok/s (with kinfer's build of
// llama.cpp, scripts/build-libs.sh). The rule below is that ratio: as few
// layers' experts on the host as free what the GPU lacks plus a cache of a
// third of them. There it comes out at 28 layers and a 9.0 GiB cache, with
// room for 32768 tokens of context, a prompt pool and passes of 2048: measured
// back to back, a 7,753-token prompt at 706 and 726 tok/s and decode at 27-29,
// against 712 and 29 with the hand-tuned flags, which kept half the context,
// no pool and 0.7 GiB free instead of 2.3.

// placeCacheShare: the GPU cache holds at least 1/placeCacheShare of the
// experts left in host memory.
const placeCacheShare = 3

// Placement is where a model's weights go.
type Placement struct {
	// HostExperts is how many of the first layers keep their routed experts in
	// host memory; 0 is none, everything on the GPU.
	HostExperts int
	// MoeCache is MiB of GPU memory caching the host experts in use.
	MoeCache int
	// Host is the bytes of experts left in host memory, and Short what the GPU
	// lacked for the whole model and the reserve.
	Host, Short uint64
}

// Place decides where a model of shape sh goes on a GPU with free bytes free,
// keeping reserve of them for the context, a pass's compute and the headroom
// (see placeReserve). A model that fits, or whose size is unknown, stays whole
// on the GPU. One that does not fit and cannot be split — not a mixture of
// experts, or short even with every expert in host memory — is an error saying
// so; whether to load it anyway is the caller's.
func Place(sh store.Shape, free, reserve uint64) (Placement, error) {
	need := uint64(sh.WeightBytes) + reserve
	if sh.WeightBytes == 0 || free == 0 || need <= free {
		return Placement{}, nil
	}
	p := Placement{Short: need - free}
	for n, b := range sh.ExpertBytes {
		p.HostExperts, p.Host = n+1, p.Host+uint64(b)
		if p.Host-p.Host/placeCacheShare >= p.Short {
			break
		}
	}
	switch {
	case p.Host == 0:
		return Placement{Short: p.Short}, fmt.Errorf("%s of weights and %s kept free need %s more than the GPU's %s, "+
			"and it is not a mixture of experts, whose experts could wait in host memory",
			gib(uint64(sh.WeightBytes)), gib(reserve), gib(p.Short), gib(free))
	case p.Host < p.Short:
		return Placement{Short: p.Short}, fmt.Errorf("%s of weights and %s kept free need %s more than the GPU's %s, "+
			"and every layer's experts are only %s: what every token uses does not fit",
			gib(uint64(sh.WeightBytes)), gib(reserve), gib(p.Short), gib(free), gib(p.Host))
	}
	p.MoeCache = int((p.Host - p.Short) >> 20)
	return p, nil
}

// placeFor is the placement Open loads a model with when the operator named
// none (PlaceUnasked), said once with the reasons.
func placeFor(sh store.Shape, free uint64, opts Options, name string) Placed {
	p, err := PlaceUnasked(sh, free, opts.Slots, opts.ContextSize, opts.Batch)
	if err != nil {
		log.Printf("placement: %s: %v — loading it whole anyway", name, err)
		return Placed{}
	}
	if p.HostExperts == 0 {
		return Placed{}
	}
	reserve, ubatch := p.Reserve, p.Ubatch
	// Whether the host experts stay in memory or come from the SSD is the
	// difference between a slower model and a much slower one: on unified
	// memory, what the GPU holds is not there for them.
	where := "the ones the cache misses are read from the SSD"
	onGPU := uint64(sh.WeightBytes) - p.Host + uint64(p.MoeCache)<<20 + reserve
	if ram := uint64(hardware.Detect().RAM); ram > 0 {
		beside := sub(ram, onGPU)
		where = fmt.Sprintf("host memory beside the GPU's part is about %s, so %s", gib(beside), where)
		if p.Host+thinHeadroom <= beside {
			where = fmt.Sprintf("host memory beside the GPU's part is about %s, enough to keep them", gib(beside))
		}
	}
	log.Printf("placement: %s has %s of weights; with %s kept free for its context, a pass of %d and the headroom, "+
		"it needs %s more than the GPU's %s. The routed experts of its first %d of %d layers, %s, stay in host memory, "+
		"and a %s GPU cache keeps the ones in use; %s. -host-experts and -moe-cache choose instead, -place=false loads it whole.",
		name, gib(uint64(sh.WeightBytes)), gib(reserve), ubatch, gib(p.Short), gib(free),
		p.HostExperts, len(sh.ExpertBytes), gib(p.Host), gib(uint64(p.MoeCache)<<20), where)
	return p
}

// Placed is a placement with what it kept room for: Reserve bytes beyond the
// weights, for Slots conversations of Tokens each, a prompt pool entry as long,
// and a pass of Ubatch tokens. The context is sized to that, since that is all
// there is (see Open). Reserve is zero when nothing was placed.
type Placed struct {
	Placement
	Reserve               uint64
	Slots, Tokens, Ubatch int
}

// PlaceUnasked is the placement serve loads a model of shape sh with, on a GPU
// with free bytes free, when -slots, -ctx and -batch are as given (0 for
// unset) and neither -host-experts nor -moe-cache is. The room it keeps is for
// the conversations asked for, or one, of the context asked for, or of what
// sizing aims at (ctxFloor); and for a pass of -batch, or the 2048 host
// experts get.
func PlaceUnasked(sh store.Shape, free uint64, slots, ctx, batch int) (Placed, error) {
	p := placeRoom(sh, slots, ctx, batch)
	var err error
	p.Placement, err = Place(sh, free, p.Reserve)
	if p.HostExperts == 0 {
		p.Reserve = 0 // nothing placed, so nothing kept
	}
	return p, err
}

// placeRoom is the room placement keeps, before it places anything.
func placeRoom(sh store.Shape, slots, ctx, batch int) Placed {
	p := Placed{Slots: max(slots, 1), Tokens: ctx, Ubatch: batch}
	if p.Tokens <= 0 {
		p.Tokens = ctxFloor
		if sh.TrainedCtx > 0 {
			p.Tokens = min(p.Tokens, sh.TrainedCtx)
		}
	}
	if p.Ubatch <= 0 {
		p.Ubatch = ubatchLarge
	}
	p.Reserve = placeReserve(sh, p.Tokens, p.Slots, p.Ubatch)
	return p
}

// placeReserve is what placement keeps free beyond the weights: the cache of
// slots conversations of tokens each and a prompt pool entry as long, the
// compute buffer of a pass of ubatch tokens, and the headroom serve warns
// below (thinHeadroom).
func placeReserve(sh store.Shape, tokens, slots, ubatch int) uint64 {
	return uint64(sh.CacheBytesFor(tokens*(slots+1), slots+1)) + computeBytes(sh, ubatch) + thinHeadroom
}

// computeBytes estimates the compute buffer of a pass of ubatch tokens: its
// logits, ubatch x Vocab floats, and about 32 x EmbedDim floats of activations
// a token. Measured: qwen2.5-0.5b in passes of 1024, 703.7 MiB, where this
// says 705.5; Flash-Next in passes of 2048 with 16384 tokens of context,
// 2212.8 MiB, where this says 2580. A long context adds attention scores.
func computeBytes(sh store.Shape, ubatch int) uint64 {
	vocab := sh.Vocab
	if vocab == 0 {
		vocab = 152064 // a Qwen-sized vocabulary
	}
	return uint64(ubatch) * 4 * uint64(vocab+32*sh.EmbedDim)
}

// Placing a model across machines.
//
// With -peers, other machines' GPUs can hold some of a model's layers: the
// first ones, as llama.cpp orders its devices (peers first), with the rest and
// the output here. Every token's activations cross the network to them and
// back, so a layer there costs a round trip a token where one here costs
// nothing: over the USB 2.0 cable between the box and the MacBook Pro, the
// same model decoded at 26.3 tok/s with 2 layers on the laptop and 23.7 with 8
// — the trips, not the laptop's GPU. So the peers take only what this machine
// cannot hold, and none of a model that fits here. llama.cpp left alone splits
// by free memory instead, which put a third of the model behind the cable.
//
// What the peers cannot hold either stays in this machine's host memory, as
// on one machine: the routed experts of its own last layers, with a GPU cache.
// Not the first layers', which live on the peers: their experts would be
// computed here for layers computed there.

// DevicePlacement is where a model goes across machines: how many of its
// layers each device holds, in llama.cpp's order — the peers', then this
// machine's, which also holds the output — and which of this machine's layers
// keep their routed experts in host memory: HostExperts of them from
// FirstHost.
type DevicePlacement struct {
	Layers    []int
	FirstHost int
	Placement
}

// PlaceDevices decides it for a model of shape sh on devices with free bytes
// free — the peers', then this machine's last — keeping reserve of this
// machine's for its context, the compute of a pass and the headroom, and
// peerReserve of each peer's. A model whose size is unknown gets no
// placement, and llama.cpp's own split.
func PlaceDevices(sh store.Shape, free []uint64, reserve, peerReserve uint64) (DevicePlacement, error) {
	n := len(sh.LayerBytes)
	if n == 0 || sh.WeightBytes == 0 || len(free) < 2 {
		return DevicePlacement{}, nil
	}
	var layers uint64
	for _, b := range sh.LayerBytes {
		layers += uint64(b)
	}
	// The output and the norms stay here with the reserve; the input
	// embedding llama.cpp keeps on the CPU, counted here all the same.
	here := sub(free[len(free)-1], reserve+sub(uint64(sh.WeightBytes), layers))
	d := DevicePlacement{Layers: make([]int, len(free))}

	// Each peer in turn takes the next layers, while what is left would not
	// fit here and the layer fits there.
	next, rest := 0, layers
	for p := range free[:len(free)-1] {
		room := sub(free[p], peerReserve)
		for next < n && rest > here && uint64(sh.LayerBytes[next]) <= room {
			room -= uint64(sh.LayerBytes[next])
			rest -= uint64(sh.LayerBytes[next])
			d.Layers[p]++
			next++
		}
	}
	d.Layers[len(free)-1] = n - next
	d.FirstHost = n
	if rest <= here {
		return d, nil
	}

	// Still short: this machine's last layers' experts go to host memory, as
	// few as free what is missing plus a cache of a third of them.
	d.Short = rest - here
	for i := n - 1; i >= next && d.Host-d.Host/placeCacheShare < d.Short; i-- {
		d.Host += uint64(sh.ExpertBytes[i])
		d.HostExperts++
		d.FirstHost = i
	}
	if d.Host < d.Short {
		return DevicePlacement{}, fmt.Errorf("%s of layers need %s more than these machines' GPUs can hold, "+
			"and the experts of this machine's %d layers are only %s",
			gib(layers), gib(d.Short), n-next, gib(d.Host))
	}
	d.MoeCache = int((d.Host - d.Short) >> 20)
	return d, nil
}

// Split is the tensor_split that gives each device its Layers of a model of
// n layers. llama.cpp puts layer il on the first device whose cumulative
// share exceeds il/(n+1), the output counting as layer n; a boundary half a
// layer before the next device's first layer cannot land on either side by
// rounding.
func (d DevicePlacement) Split(n int) []float32 {
	split := make([]float32, len(d.Layers))
	var prev float64
	took := 0
	for i, c := range d.Layers[:len(d.Layers)-1] {
		took += c
		edge := prev
		if c > 0 {
			edge = (float64(took) - 0.5) / float64(n+1)
		}
		split[i] = float32(edge - prev)
		prev = edge
	}
	split[len(split)-1] = float32(1 - prev)
	return split
}

// peerReserve is what placement keeps free on a peer: the activations of a
// pass of ubatch tokens (the logits are computed here) and the headroom —
// a peer is someone's machine.
func peerReserve(sh store.Shape, ubatch int) uint64 {
	return uint64(ubatch)*4*uint64(32*sh.EmbedDim) + thinHeadroom
}

// placeAcross is the placement Open loads a model with across devs — the
// peers', then this machine's, which has free bytes free — when the operator
// named no split and no host experts (PlaceDevices), said once with the
// reasons. The Placed is empty when the model fits here and the peers get
// nothing; a model it cannot place is left to llama.cpp's split.
func placeAcross(sh store.Shape, devs []uintptr, free uint64, opts Options, name string) (DevicePlacement, Placed) {
	room := placeRoom(sh, opts.Slots, opts.ContextSize, opts.Batch)
	var frees []uint64
	var peers []string
	for _, d := range devs {
		if d == 0 {
			break // the list llama.cpp takes ends in a null
		}
		f, _ := llama.DeviceMemory(d)
		frees = append(frees, f)
		peers = append(peers, fmt.Sprintf("%s has %s free", llama.DeviceName(d), gib(f)))
	}
	if len(frees) < 2 {
		return DevicePlacement{}, Placed{}
	}
	frees[len(frees)-1] = free
	peers = peers[:len(peers)-1]
	d, err := PlaceDevices(sh, frees, room.Reserve, peerReserve(sh, room.Ubatch))
	if err != nil {
		log.Printf("placement: %s: %v — llama.cpp splits it by free memory instead", name, err)
		return DevicePlacement{}, Placed{}
	}
	if d.Layers == nil {
		return d, Placed{}
	}
	head := fmt.Sprintf("placement: %s has %s of weights; this GPU has %s free and keeps %s of it for the context, "+
		"a pass of %d and the headroom; %s.", name, gib(uint64(sh.WeightBytes)), gib(free), gib(room.Reserve),
		room.Ubatch, strings.Join(peers, ", "))
	here := len(d.Layers) - 1
	if d.Layers[here] == len(sh.LayerBytes) {
		log.Printf("%s It fits here, so the peers take none of its layers: a layer behind the network costs a round "+
			"trip a token. -split gives them some.", head)
		return d, Placed{}
	}
	var parts []string
	first := 0
	for i, c := range d.Layers[:here] {
		if c > 0 {
			parts = append(parts, fmt.Sprintf("layers %d-%d go to %s", first, first+c-1, llama.DeviceName(devs[i])))
		}
		first += c
	}
	parts = append(parts, fmt.Sprintf("the other %d and the output stay here", d.Layers[here]))
	if d.HostExperts > 0 {
		parts = append(parts, fmt.Sprintf("and the routed experts of layers %d-%d, %s, stay in host memory with a %s GPU "+
			"cache for the ones in use", d.FirstHost, d.FirstHost+d.HostExperts-1, gib(d.Host), gib(uint64(d.MoeCache)<<20)))
	}
	log.Printf("%s Only what does not fit here goes to the peers: %s. -split chooses instead, "+
		"-place=false leaves the split to llama.cpp.", head, strings.Join(parts, ", "))
	room.Placement = d.Placement
	return d, room
}
