package engine

import (
	"strings"

	"github.com/LocalKinAI/kinfer/internal/store"
)

// CacheQ8 says whether a model's attention cache is kept at q8_0 rather than
// f16, which cacheType "q8_0" asks for: about half the memory, for replies
// Ollama's documentation calls usually no different, so the same memory holds
// twice the context. It is not the default, because the context is all it
// buys: on the box (M3 Ultra) with 20,000-token prompts, decode went from
// 28.8 to 28.4 tok/s on qwen3.8-27b, 97.4 to 92.3 on ornith-1.5-35b and 259
// to 222 on Qwen2.5-0.5B, while sizing gave the memory saved to longer
// conversations and a larger micro-batch.
//
// Asked for, it stays f16 where llama.cpp would refuse the context or
// kinfer's sizing does not describe the cache:
//   - a head that 32, q8_0's block, does not divide, which llama.cpp refuses;
//   - the DeepSeek family and the models built on it (Kimi, GLM's DSA), whose
//     latent and compressed caches are not the per-head K and V CacheBytes
//     counts — and whose cache is small already;
//   - Grok, for which llama.cpp turns off the flash attention a q8_0 V needs.
func CacheQ8(sh store.Shape, cacheType string) bool {
	if cacheType != "q8_0" {
		return false
	}
	k, v := sh.HeadDim(), sh.ValLen
	if v == 0 {
		v = k
	}
	if k == 0 || k%32 != 0 || v%32 != 0 {
		return false
	}
	switch arch := strings.ToLower(sh.Arch); {
	case strings.HasPrefix(arch, "deepseek"), strings.HasPrefix(arch, "kimi"), arch == "glm-dsa", arch == "grok":
		return false
	}
	return true
}
