package engine

import (
	"github.com/LocalKinAI/kinfer/internal/llama"
)

// prefixPool keeps recently seen prompt prefixes resident in the KV cache, each
// under its own sequence id, so a request that begins the same way skips
// prefilling that part.
//
// This is the fleet's case exactly. An agent's system prompt is identical byte
// for byte on every turn it ever takes, and a soul runs to thousands of tokens;
// without a pool each turn prefills all of it again. Two agents sharing a soul
// pay for it twice, a hundred agents a hundred times.
//
// Nothing is copied. llama_memory_seq_cp tags cells that already exist with a
// second sequence id, so adopting a 4000-token prefix costs a pointer walk
// rather than 4000 tokens of compute. It does require llama.cpp's unified KV
// buffer (kv_unified), which is what lets cells belong to more than one
// sequence — see Open.
//
// Only the scheduler goroutine touches a pool, so nothing here locks.
type prefixPool struct {
	entries []*prefixEntry

	// The two llama.cpp calls are fields so the matching and eviction rules —
	// which is where this can go quietly wrong — can be tested without a model
	// and a GPU.
	share  func(src, dst int32, n int32)
	forget func(seq int32)

	// minMatch is the shortest prefix worth adopting. Below it the bookkeeping
	// outweighs a prefill that llama.cpp would have done in microseconds.
	minMatch int

	clock int64
}

// prefixEntry is one pooled prefix: a sequence id whose KV cells hold exactly
// these tokens at positions 0..len(tokens)-1.
type prefixEntry struct {
	seq     int32
	tokens  []llama.Token
	lastUse int64
}

// newPrefixPool claims n sequence ids starting at firstSeq.
func newPrefixPool(lctx llama.Context, firstSeq int32, n, minMatch int) *prefixPool {
	if n <= 0 {
		return nil
	}
	p := &prefixPool{
		minMatch: minMatch,
		share: func(src, dst int32, n int32) {
			llama.CopySequence(lctx, src, dst, 0, n)
		},
		forget: func(seq int32) {
			llama.ForgetSequence(lctx, seq, -1, -1)
		},
	}
	for i := 0; i < n; i++ {
		p.entries = append(p.entries, &prefixEntry{seq: firstSeq + int32(i)})
	}
	return p
}

// match returns the pooled prefix sharing the most tokens with the given
// prompt, and how many tokens that is.
//
// The count never reaches len(tokens): a prompt has to contribute at least one
// token to the batch, because logits are only produced for tokens that were
// decoded, and with none there is nothing to sample the first reply token from.
func (p *prefixPool) match(tokens []llama.Token) (*prefixEntry, int) {
	if p == nil || len(tokens) < 2 {
		return nil, 0
	}
	limit := len(tokens) - 1

	var best *prefixEntry
	bestLen := 0
	for _, e := range p.entries {
		n := commonPrefix(e.tokens, tokens)
		if n > limit {
			n = limit
		}
		if n > bestLen {
			best, bestLen = e, n
		}
	}
	if bestLen < p.minMatch {
		return nil, 0
	}
	return best, bestLen
}

// adopt gives a slot's sequence the pooled prefix's first n cells.
func (p *prefixPool) adopt(e *prefixEntry, slotSeq int32, n int) {
	p.share(e.seq, slotSeq, int32(n))
	p.clock++
	e.lastUse = p.clock
}

// publish records a freshly prefilled prompt so later requests can share it.
//
// The cells are already in the cache under the slot's sequence, so this only
// tags them a second time — the pool never prefills anything itself.
func (p *prefixPool) publish(slotSeq int32, tokens []llama.Token) {
	if p == nil || len(tokens) < p.minMatch {
		return
	}

	e := p.victim(tokens)
	if e == nil {
		return // already covered by an entry at least this long
	}

	// Drop whatever the entry held. Its cells may still be shared with a slot
	// that adopted them; removing this sequence id leaves the slot's copy.
	p.forget(e.seq)
	p.share(slotSeq, e.seq, int32(len(tokens)))

	e.tokens = append(e.tokens[:0], tokens...)
	p.clock++
	e.lastUse = p.clock
}

// victim picks the entry to overwrite: an empty one, the one this prompt
// extends, or the least recently used. It returns nil when an entry already
// covers the prompt.
func (p *prefixPool) victim(tokens []llama.Token) *prefixEntry {
	var lru *prefixEntry
	for _, e := range p.entries {
		if len(e.tokens) == 0 {
			return e
		}
		n := commonPrefix(e.tokens, tokens)
		if n == len(e.tokens) {
			if n == len(tokens) {
				p.clock++
				e.lastUse = p.clock
				return nil // identical prompt already pooled
			}
			// This prompt continues where the entry stops: replace it with the
			// longer one rather than keeping a prefix of a prefix.
			return e
		}
		if lru == nil || e.lastUse < lru.lastUse {
			lru = e
		}
	}
	return lru
}

// commonPrefix counts the tokens two sequences begin with in common.
func commonPrefix(a, b []llama.Token) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
