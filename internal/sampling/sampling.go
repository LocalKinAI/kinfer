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

import (
	"errors"
	"math"
	"unsafe"

	"github.com/LocalKinAI/kinfer/internal/llama"
)

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

	// RepeatLastN is how many recent tokens the penalties consider; -1 is
	// every token of the reply.
	RepeatLastN int

	// PresencePenalty is subtracted from the logit of every token seen in
	// those, however often: vLLM's and OpenAI's presence_penalty, which keeps a
	// small model from going round in the same few lines. Zero disables.
	PresencePenalty float32

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

// PenaltiesOn reports whether p asks for a repeat or presence penalty.
func PenaltiesOn(p Params) bool {
	return p.RepeatLastN != 0 && (p.RepeatPenalty != 1 && p.RepeatPenalty > 0 || p.PresencePenalty != 0)
}

// Sampler draws tokens and remembers what it has drawn, so the repetition
// penalty has history to work with.
//
// Not safe for concurrent use; give each generation its own. Close it when the
// generation ends — the chain is C memory, which the Go collector will not
// reclaim.
type Sampler struct {
	chain llama.Sampler

	// grammar holds the reply to a grammar from the moment trigger is drawn
	// outside a think block; see Constrain. held is that moment passed.
	grammar                  llama.Sampler
	trigger, open, close     llama.Token
	thinking, startsThinking bool
	held                     bool
	redrawn                  int
	nVocab                   int32
	cur                      []llama.TokenData // allocated once held
}

// NoToken is no token: a Grammar's think block, where the model has none.
const NoToken llama.Token = -1

// Grammar holds what a Sampler draws to a GBNF grammar from the moment it draws
// Trigger, the token the grammar's text opens with, to the end of the reply.
type Grammar struct {
	Vocab llama.Vocab

	// Rules is the grammar, its first rule named root.
	Rules string

	// Trigger is drawn freely; it and everything after it are held to Rules.
	Trigger llama.Token

	// Open and Close are a think block's tokens, or NoToken, and Thinking
	// says the reply starts inside one the prompt opened. Inside it a model
	// may write anything, a call it is only thinking about included, so
	// Trigger counts only outside one.
	Open, Close llama.Token
	Thinking    bool
}

// New builds a sampler chain for p.
func New(p Params) *Sampler {
	chain := llama.NewSamplerChain()

	// Penalties first, so later stages see the adjusted logits.
	if PenaltiesOn(p) {
		// llama.cpp takes a negative window for none; -1 means the whole reply.
		lastN := int32(p.RepeatLastN)
		if lastN < 0 {
			lastN = 1 << 16
		}
		repeat := p.RepeatPenalty
		if repeat <= 0 {
			repeat = 1
		}
		llama.SamplerChainAdd(chain, llama.SamplerPenalties(
			lastN, repeat,
			0, // frequency penalty: off, as it is in Ollama's defaults
			p.PresencePenalty,
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
	if s.held {
		return s.drawHeld(llama.Logits(ctx, idx, s.nVocab))
	}
	tok := llama.SamplerSample(s.chain, ctx, idx)
	s.saw(tok)
	return tok
}

// saw follows a token drawn freely: a think block's open or close, or the
// grammar's trigger, which it takes as the first of its text.
func (s *Sampler) saw(tok llama.Token) {
	if s.grammar == 0 {
		return
	}
	switch {
	case tok == s.open:
		s.thinking = true
	case tok == s.close:
		s.thinking = false
	case tok == s.trigger && !s.thinking:
		llama.SamplerAccept(s.grammar, tok)
		s.held = true
	}
}

// Constrain holds the rest of the reply to g. An error means its rules did
// not parse, and the sampler draws as it did.
func (s *Sampler) Constrain(g Grammar) error {
	gs := llama.SamplerGrammar(g.Vocab, g.Rules, "root")
	if gs == 0 {
		return errors.New("llama.cpp did not take the grammar")
	}
	if s.grammar != 0 {
		llama.SamplerFree(s.grammar)
	}
	s.grammar, s.trigger, s.open, s.close = gs, g.Trigger, g.Open, g.Close
	s.thinking, s.startsThinking, s.held = g.Thinking, g.Thinking, false
	s.nVocab = llama.NVocab(g.Vocab)
	return nil
}

// drawHeld draws from logits under the grammar the way llama.cpp's own server
// does: from the chain first, and only when the grammar rules out what that
// drew, again with the grammar applied first. A model writing a call it was taught nearly
// always draws an allowed token, and applying the grammar first every time
// would test all of the vocabulary's 151,936 candidates against it for each
// token instead of one.
//
// Only a token drawn through the grammar is ever accepted by it: a grammar
// told to accept one it rules out throws, which aborts the process.
func (s *Sampler) drawHeld(logits []float32) llama.Token {
	tok := s.pick(logits, false)
	one := llama.TokenData{ID: tok, Logit: 1}
	llama.SamplerApply(s.grammar, &llama.TokenDataArray{Data: &one, Size: 1, Selected: -1})
	if math.IsInf(float64(one.Logit), -1) {
		tok = s.pick(logits, true)
		s.redrawn++
	}
	llama.SamplerAccept(s.chain, tok)
	llama.SamplerAccept(s.grammar, tok)
	return tok
}

// pick runs the chain over every candidate, after the grammar if grammarFirst,
// and returns the token it selected without accepting it.
func (s *Sampler) pick(logits []float32, grammarFirst bool) llama.Token {
	if len(s.cur) != len(logits) {
		s.cur = make([]llama.TokenData, len(logits))
	}
	for i, l := range logits {
		s.cur[i] = llama.TokenData{ID: llama.Token(i), Logit: l}
	}
	arr := llama.TokenDataArray{Data: &s.cur[0], Size: uint64(len(s.cur)), Selected: -1}
	if grammarFirst {
		llama.SamplerApply(s.grammar, &arr)
	}
	llama.SamplerApply(s.chain, &arr)
	return unsafe.Slice(arr.Data, arr.Size)[arr.Selected].ID
}

// Held reports whether a grammar took hold of the reply, and how many of the
// tokens drawn since it did were drawn again because it ruled the first out.
func (s *Sampler) Held() (held bool, redrawn int) { return s.held, s.redrawn }

// Reset forgets the penalty history, as if the sampler were new. A grammar
// stays, waiting for its trigger again.
func (s *Sampler) Reset() {
	if s.chain != 0 {
		llama.SamplerReset(s.chain)
	}
	if s.grammar != 0 {
		llama.SamplerReset(s.grammar)
		s.thinking, s.held, s.redrawn = s.startsThinking, false, 0
	}
}

// Close frees the chain and any grammar. Safe to call twice.
func (s *Sampler) Close() {
	if s.chain != 0 {
		llama.SamplerFree(s.chain)
		s.chain = 0
	}
	if s.grammar != 0 {
		llama.SamplerFree(s.grammar)
		s.grammar = 0
	}
}

// seedOf maps kinfer's "0 means random" onto llama.cpp's sentinel.
func seedOf(seed int64) uint32 {
	if seed == 0 {
		return llama.DefaultSeed
	}
	return uint32(seed)
}
