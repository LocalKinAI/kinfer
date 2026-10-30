package engine

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/llama"
)

// A mixture of experts does not need all of its experts on the GPU. A token
// uses a few per layer (10 of Flash-Next's 512, 6 of DeepSeek-V4.1's 384), and
// the same few keep coming back. llama.cpp can leave a layer's routed experts
// in host memory — mapped from the file, so read from the SSD as they are
// touched and kept in the page cache while they are used — and keep the ones
// in use in a GPU cache (llama-moe-cache.cpp, b11998): only a miss is copied
// over. It caches only layers whose experts are all in host memory, and only
// batches of up to 32 tokens; a prompt's larger batches read what the cache
// holds and leave the rest to the CPU.
//
// It is what Strata does on a Mac with its own engine. kinfer's build of
// llama.cpp (patches/llama.cpp) adds Strata's other three parts: misses read
// by a pool of threads instead of one page fault at a time, the next layer's
// experts guessed from this layer's input and read ahead, and a warm start
// from the experts generation used last time (expertProfile below).

// expertsRegex names a layer's routed experts, as llama.cpp's own --cpu-moe
// does (LLM_FFN_EXPS_REGEX in common/common.h).
const expertsRegex = `\.ffn_(up|down|gate|gate_up)_(ch|)exps`

// hostExpertOverrides places the routed experts of the first n layers in host
// memory, or of every layer when n is negative, as --n-cpu-moe and --cpu-moe
// do. keep holds the C strings the overrides point at; both must outlive the
// model load.
func hostExpertOverrides(n int) (ov []llama.TensorBuftOverride, keep [][]byte, err error) {
	if n == 0 {
		return nil, nil, nil
	}
	cpu := llama.CPUBufferType()
	if cpu == 0 {
		return nil, nil, fmt.Errorf("this build has no host buffer type to keep experts in")
	}
	patterns := []string{expertsRegex}
	if n > 0 {
		patterns = patterns[:0]
		for i := 0; i < n; i++ {
			// blk\.1\. cannot match blk.10.: the dot after the number anchors it.
			patterns = append(patterns, fmt.Sprintf(`blk\.%d%s`, i, expertsRegex))
		}
	}
	keep = make([][]byte, len(patterns))
	ov = make([]llama.TensorBuftOverride, 0, len(patterns)+1)
	for i, p := range patterns {
		keep[i] = append([]byte(p), 0)
		ov = append(ov, llama.TensorBuftOverride{Pattern: &keep[i][0], Buft: cpu})
	}
	return append(ov, llama.TensorBuftOverride{}), keep, nil
}

// expertProfile is the file that keeps how often generation selected each
// expert of the layers the MoE cache holds, ~/.kinfer/profiles/<model>.moeprof.
// A context loads it to start with the experts it will use instead of filling
// its cache one miss at a time, and saves its counts when it is freed, the
// file's older counts halved. Empty when there is no MoE cache or this build
// of llama.cpp keeps no profile.
type expertProfile string

func newExpertProfile(model string, moeCache int) expertProfile {
	if moeCache <= 0 || !llama.MoeProfiles() {
		return ""
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	dir := filepath.Join(home, ".kinfer", "profiles")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		log.Printf("experts: no profile — %v", err)
		return ""
	}
	return expertProfile(filepath.Join(dir, strings.TrimSuffix(filepath.Base(model), ".gguf")+".moeprof"))
}

// load fills lctx's fresh cache from the profile, if there is one yet.
func (p expertProfile) load(lctx llama.Context) {
	if p == "" {
		return
	}
	if _, err := os.Stat(string(p)); err != nil {
		log.Printf("experts: no profile yet — %s is written when this context is freed", p)
		return
	}
	if !llama.LoadMoeProfile(lctx, string(p)) {
		log.Printf("experts: %s does not match this model and cache, so the cache starts empty", p)
	}
}

// save adds what lctx generated to the profile. lctx must not be decoding.
func (p expertProfile) save(lctx llama.Context) {
	if p == "" || lctx == 0 {
		return
	}
	if !llama.SaveMoeProfile(lctx, string(p)) {
		log.Printf("experts: could not write %s", p)
	}
}
