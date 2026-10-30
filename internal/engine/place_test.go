package engine

import (
	"strings"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/store"
)

// experts is a mixture of experts with n layers of per bytes of routed experts
// each and dense bytes of everything else.
func experts(n int, per, dense int64) store.Shape {
	sh := store.Shape{Experts: 512, ExpertBytes: make([]int64, n)}
	for i := range sh.ExpertBytes {
		sh.ExpertBytes[i] = per
	}
	sh.WeightBytes = dense + int64(n)*per
	return sh
}

const gibi = uint64(1) << 30

// checkPlacement asserts what Place promises: the GPU's part fits with the
// reserve, the cache holds at least a third of the host experts unless every
// layer's are there, and no fewer layers would have done.
func checkPlacement(t *testing.T, sh store.Shape, free, reserve uint64, p Placement) {
	t.Helper()
	cache := uint64(p.MoeCache) << 20
	onGPU := uint64(sh.WeightBytes) - p.Host + cache
	if onGPU+reserve > free {
		t.Errorf("the GPU's part is %s + %s kept free, more than its %s", gib(onGPU), gib(reserve), gib(free))
	}
	if p.HostExperts < len(sh.ExpertBytes) && cache < p.Host/placeCacheShare-1<<20 {
		t.Errorf("cache %s for %s of host experts: less than a third", gib(cache), gib(p.Host))
	}
	if p.HostExperts > 1 {
		var fewer uint64
		for _, b := range sh.ExpertBytes[:p.HostExperts-1] {
			fewer += uint64(b)
		}
		if fewer-fewer/placeCacheShare >= p.Short {
			t.Errorf("%d layers would have done: %s of experts covers %s short with a third cached",
				p.HostExperts-1, gib(fewer), gib(p.Short))
		}
	}
}

// The MacBook Pro: Flash-Next IQ3_S, 51.1 GiB of weights of which 45.4 are
// 48 layers' experts, on a GPU with 40 GiB. Tuned by hand it ran best with 24
// layers' experts on the host and an 8 GiB cache; the rule asks for a few more
// layers, since it keeps more free, and a cache of the same size.
func TestPlaceSplitsAModelTheGPUCannotHold(t *testing.T) {
	g := float64(gibi)
	per := int64(22.7 / 24 * g)
	sh := experts(48, per, int64(51.1*g)-48*per)
	free, reserve := 40*gibi, 5*gibi+600<<20
	p, err := Place(sh, free, reserve)
	if err != nil {
		t.Fatal(err)
	}
	checkPlacement(t, sh, free, reserve, p)
	if p.HostExperts < 24 || p.HostExperts > 28 {
		t.Errorf("%d layers' experts on the host; the hand-tuned 24 ran best, so expect 24-28", p.HostExperts)
	}
	if c := p.MoeCache >> 10; c < 7 || c > 10 {
		t.Errorf("a %d MiB cache; 8 GiB ran best, so expect 7-10 GiB", p.MoeCache)
	}
}

func TestPlaceLeavesAModelThatFitsOnTheGPU(t *testing.T) {
	per := int64(gibi)
	sh := experts(48, per, 6*int64(gibi)) // 54 GiB
	for _, c := range []struct {
		name          string
		sh            store.Shape
		free, reserve uint64
	}{
		{"fits", sh, 77 * gibi, 8 * gibi},
		{"fits exactly", sh, 62 * gibi, 8 * gibi},
		{"size unknown", store.Shape{Experts: 512}, 10 * gibi, 8 * gibi},
		{"no GPU", sh, 0, 8 * gibi},
	} {
		p, err := Place(c.sh, c.free, c.reserve)
		if err != nil || p != (Placement{}) {
			t.Errorf("%s: %+v, %v; want everything on the GPU", c.name, p, err)
		}
	}
}

func TestPlaceSaysWhatCannotBeSplit(t *testing.T) {
	dense := store.Shape{WeightBytes: 70 << 30}
	if p, err := Place(dense, 40*gibi, 4*gibi); err == nil || p.HostExperts != 0 || !strings.Contains(err.Error(), "not a mixture of experts") {
		t.Errorf("a dense model 34 GiB short: %+v, %v", p, err)
	}
	// Experts are 24 of the 70 GiB: every layer's on the host still leaves 46
	// GiB for a GPU of 30.
	moe := experts(24, int64(gibi), 46<<30)
	if p, err := Place(moe, 30*gibi, 4*gibi); err == nil || p.HostExperts != 0 || !strings.Contains(err.Error(), "what every token uses") {
		t.Errorf("a mixture short even without experts: %+v, %v", p, err)
	}
}

// Short by more than two thirds of every expert, but not by all of them:
// every layer's experts go to the host and the cache gets what is left.
func TestPlaceTakesEveryLayerBeforeGivingUp(t *testing.T) {
	sh := experts(10, int64(gibi), 4<<30) // 14 GiB, 10 of it experts
	free, reserve := 7*gibi, 2*gibi       // 9 GiB short
	p, err := Place(sh, free, reserve)
	if err != nil {
		t.Fatal(err)
	}
	if p.HostExperts != 10 || p.MoeCache != 1024 {
		t.Errorf("%+v; want every layer's experts on the host and the 1 GiB left as cache", p)
	}
	checkPlacement(t, sh, free, reserve, p)
}

// DeepSeek-style models start with dense layers: they free nothing, and are
// counted past.
func TestPlaceCountsPastDenseLayers(t *testing.T) {
	sh := experts(8, int64(gibi), 2<<30)
	sh.ExpertBytes[0], sh.ExpertBytes[1] = 0, 0
	sh.WeightBytes -= 2 << 30
	p, err := Place(sh, 6*gibi, gibi) // 8 GiB of weights + 1 kept: 3 short
	if err != nil {
		t.Fatal(err)
	}
	checkPlacement(t, sh, 6*gibi, gibi, p)
	if p.HostExperts != 7 || p.Host != 5*gibi {
		t.Errorf("%+v; want 7 layers, the first two dense, so 5 GiB of experts", p)
	}
}

// The compute buffer is mostly logits; checked against what llama.cpp
// allocated for qwen2.5-0.5b in passes of 1024 (703.7 MiB).
func TestComputeBytesMatchesWhatLlamaCppAllocated(t *testing.T) {
	sh := store.Shape{Vocab: 151936, EmbedDim: 896}
	if got := computeBytes(sh, 1024) >> 20; got < 690 || got > 720 {
		t.Errorf("qwen2.5-0.5b at 1024: %d MiB, llama.cpp allocated 703.7", got)
	}
}
