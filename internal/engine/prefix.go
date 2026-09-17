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
//
// # What reuse costs
//
// Adopted cells are correct — a prompt reused in full still lets the model
// quote the last sentence of its own system prompt, which a position error
// would destroy — but reuse is not bit-exact.
//
// How much of a prompt is adopted varies with what the pool happens to hold.
// The first request prefills its question in one batch of a dozen tokens; once
// that prompt is pooled, a repeat decodes a single token instead. Different
// batch shapes mean different floating-point reduction orders, so a logit that
// was a near-tie can land the other way.
//
// Measured on an M3 Ultra with a 4,000-token system prompt: with the pool off
// the same question produced one output in ten runs; with it on, two. Across
// twenty varied prompts, nineteen were byte-identical to the pooled-off run and
// one differed — an arithmetic question where the model was genuinely torn
// between two phrasings, and where the pooled answer happened to be the better
// one. Replies stay on topic and coherent either way; what is lost is the
// guarantee that a fixed seed reproduces a previous run exactly. A caller that
// needs that guarantee should run with -prefix -1.
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

	// exact limits reuse to whole entries: a prompt adopts a pooled prefix
	// only when it continues past the end of it. It is set for models that
	// keep recurrent state. Attention cells can be shared up to any position,
	// but a recurrent layer's state is one value for the whole sequence, so
	// adopting the first n tokens of a longer entry hands the new prompt a
	// state that has already read the rest of the old one. llama.cpp says so
	// ("non-consecutive token position") and the output shows it: on
	// ornith-1.5-35b the same prompt sent twice came back the second time with
	// its tool call written as <invoke> prose instead of a <tool_call>.
	//
	// An agent's next turn continues its previous prompt, so the reuse that
	// matters most is the kind that stays. What goes is reuse of a repeat, an
	// edited history, or another conversation that shares only a system prompt.
	exact bool

	clock int64
}

// prefixEntry is one pooled prefix: a sequence id whose KV cells hold exactly
// these tokens at positions 0..len(tokens)-1.
type prefixEntry struct {
	seq     int32
	tokens  []llama.Token
	lastUse int64

	// shared marks a system prompt: a prefix every conversation of the agent
	// that sent it opens with, as opposed to one conversation so far. It only
	// matters to an exact pool; see victim.
	shared bool
}

// newPrefixPool claims n sequence ids starting at firstSeq.
func newPrefixPool(lctx llama.Context, firstSeq int32, n, minMatch int, exact bool) *prefixPool {
	if n <= 0 {
		return nil
	}
	p := &prefixPool{
		minMatch: minMatch,
		exact:    exact,
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

// wholeOnly reports a pool that adopts only whole entries; see exact. A nil
// pool adopts nothing, so it is not one.
func (p *prefixPool) wholeOnly() bool { return p != nil && p.exact }

// match returns the pooled prefix sharing the most tokens with the given
// prompt, and how many tokens that is.
//
// The count never reaches len(tokens): a prompt has to contribute at least one
// token to the batch, because logits are only produced for tokens that were
// decoded, and with none there is nothing to sample the first reply token from.
//
// An exact pool only considers entries the prompt contains whole; see exact.
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
		if p.exact && n != len(e.tokens) {
			continue
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
	p.put(slotSeq, tokens, false)
}

// publishShared records a system prompt: the prefix other conversations of
// the same agent open with. An exact pool keeps it under the conversations
// that extend it.
func (p *prefixPool) publishShared(slotSeq int32, tokens []llama.Token) {
	p.put(slotSeq, tokens, true)
}

func (p *prefixPool) put(slotSeq int32, tokens []llama.Token, shared bool) {
	if p == nil || len(tokens) < p.minMatch {
		return
	}

	e := p.victim(tokens, shared)
	if e == nil {
		return // already covered by an entry at least this long
	}

	// Drop whatever the entry held. Its cells may still be shared with a slot
	// that adopted them; removing this sequence id leaves the slot's copy.
	p.forget(e.seq)
	p.share(slotSeq, e.seq, int32(len(tokens)))

	e.tokens = append(e.tokens[:0], tokens...)
	e.shared = shared
	p.clock++
	e.lastUse = p.clock
}

// clear empties every entry and reports whether any held a prompt. A pool that
// shares the slots' cells gives them back this way when a batch needs them.
// Slots that adopted an entry keep their copy: forgetting the pool's sequence
// id leaves theirs.
func (p *prefixPool) clear() bool {
	if p == nil {
		return false
	}
	freed := false
	for _, e := range p.entries {
		if len(e.tokens) == 0 {
			continue
		}
		p.forget(e.seq)
		e.tokens = e.tokens[:0]
		e.shared = false
		freed = true
	}
	return freed
}

// victim picks the entry to overwrite: an empty one, the one this prompt
// extends, or the least recently used. It returns nil when an entry already
// covers the prompt.
//
// An exact pool keeps a system prompt under the conversations that extend it —
// replacing it would take the one prefix every other conversation can still
// adopt whole — and when full it evicts a conversation before a system prompt.
func (p *prefixPool) victim(tokens []llama.Token, shared bool) *prefixEntry {
	var lru, lruConversation *prefixEntry
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
			if !p.exact || e.shared == shared {
				return e
			}
		}
		if lru == nil || e.lastUse < lru.lastUse {
			lru = e
		}
		if !e.shared && (lruConversation == nil || e.lastUse < lruConversation.lastUse) {
			lruConversation = e
		}
	}
	if p.exact && lruConversation != nil {
		return lruConversation
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
