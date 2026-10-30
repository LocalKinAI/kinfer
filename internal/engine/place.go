package engine

import (
	"fmt"
	"log"

	"github.com/LocalKinAI/kinfer/internal/hardware"
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
// there is (see Open).
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
	var err error
	p.Placement, err = Place(sh, free, p.Reserve)
	return p, err
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
