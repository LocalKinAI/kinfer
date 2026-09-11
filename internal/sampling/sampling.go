// Package sampling picks the next token from a model's logits.
//
// It used to implement the whole pipeline in Go, because gollama.cpp bound only
// a greedy sampler and greedy decoding degenerates — observed on Qwen2.5-0.5B:
// "财是人的命根，是人的命脉，是人的命关。" repeated until the token budget ran
// out. That worked, but cost most of the throughput: picking the best 40 of
// 151,936 candidates meant sorting all of them, once per token, and copying
// every logit into Go first. Measured 44.7 tok/s against 134 for greedy on the
// same model.
//
// Now that llama.cpp is bound directly (internal/llama), this is a thin builder
// over llama_sampler_chain_*. llama.cpp does a partial selection rather than a
// full sort, in C, over its own logit buffer — nothing crosses into Go until a
// token has been chosen.
//
// The chain order is llama.cpp's own:
//
//	repetition penalty → top-k → top-p → temperature → draw
package sampling

import "github.com/LocalKinAI/kinfer/internal/llama"

// Params configures a Sampler. The zero value is not useful; start from
// DefaultParams.
type Params struct {
	// Temperature flattens (>1) or sharpens (<1) the distribution.
	// Zero means greedy: always take the most likely token.
	Temperature float32

	// TopK keeps only the K most likely tokens. Zero disables.
	TopK int

	// TopP (nucleus) keeps the smallest set of tokens whose probabilities sum
	// to P. Values >= 1 disable it.
	TopP float32

	// RepeatPenalty divides the logit of recently seen tokens. 1.0 disables.
	// Around 1.1 is the usual choice — high enough to break loops, low enough
	// that natural repetition ("the … the") still survives.
	RepeatPenalty float32

	// RepeatLastN is how many recent tokens the penalty considers.
	RepeatLastN int

	// Seed makes a run reproducible. Zero picks a random seed.
	Seed int64
}

// DefaultParams are conservative settings that produce coherent prose without
// the loops greedy decoding falls into.
func DefaultParams() Params {
	return Params{
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.95,
		RepeatPenalty: 1.1,
		RepeatLastN:   64,
	}
}

// Sampler draws tokens and remembers what it has drawn, so the repetition
// penalty has history to work with.
//
// Not safe for concurrent use; give each generation its own. Close it when the
// generation ends — the chain is C memory, which the Go collector will not
// reclaim.
type Sampler struct {
	chain llama.Sampler
}

// New builds a sampler chain for p.
func New(p Params) *Sampler {
	chain := llama.NewSamplerChain()

	// Penalties first, so later stages see the adjusted logits.
	if p.RepeatPenalty != 1 && p.RepeatPenalty > 0 && p.RepeatLastN != 0 {
		llama.SamplerChainAdd(chain, llama.SamplerPenalties(
			int32(p.RepeatLastN), p.RepeatPenalty,
			0, // frequency penalty: off, as it is in Ollama's defaults
			0, // presence penalty: likewise
		))
	}

	// Temperature 0 means greedy, and a greedy chain wants no narrowing stages
	// in front of it: taking the most likely token is the same decision whether
	// or not the field was trimmed first.
	if p.Temperature <= 0 {
		llama.SamplerChainAdd(chain, llama.SamplerGreedy())
		return &Sampler{chain: chain}
	}

	if p.TopK > 0 {
		llama.SamplerChainAdd(chain, llama.SamplerTopK(int32(p.TopK)))
	}
	if p.TopP > 0 && p.TopP < 1 {
		// min_keep 1: never narrow the field to nothing.
		llama.SamplerChainAdd(chain, llama.SamplerTopP(p.TopP, 1))
	}
	llama.SamplerChainAdd(chain, llama.SamplerTemp(p.Temperature))

	// A chain must end in something that selects. Dist draws from whatever
	// distribution the stages above left behind.
	llama.SamplerChainAdd(chain, llama.SamplerDist(seedOf(p.Seed)))

	return &Sampler{chain: chain}
}

// Sample returns the next token for ctx, reading the logits at index idx of the
// most recent decode. It also records the token for the repetition penalty.
//
// idx matters once a batch carries several sequences: each one's logits live at
// the position its last token occupied, and -1 (the last row) would hand every
// slot the same token.
func (s *Sampler) Sample(ctx llama.Context, idx int32) llama.Token {
	return llama.SamplerSample(s.chain, ctx, idx)
}

// Reset forgets the penalty history, as if the sampler were new.
func (s *Sampler) Reset() {
	if s.chain != 0 {
		llama.SamplerReset(s.chain)
	}
}

// Close frees the chain. Safe to call twice.
func (s *Sampler) Close() {
	if s.chain != 0 {
		llama.SamplerFree(s.chain)
		s.chain = 0
	}
}

// seedOf maps kinfer's "0 means random" onto llama.cpp's sentinel.
func seedOf(seed int64) uint32 {
	if seed == 0 {
		return llama.DefaultSeed
	}
	return uint32(seed)
}
