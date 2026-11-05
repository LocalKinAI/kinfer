package mlxengine

// prefixPool keeps the caches of recently read prompts, so a prompt that
// begins with one skips reading that part: an agent's next turn begins with
// its conversation so far, and its next conversation with its system prompt.
// The rules are the GGUF engine's for a model with recurrent state (see
// internal/engine/prefix.go): Qwen3.5's DeltaNet layers keep one state for
// the whole sequence, so an entry is adopted only whole, and only by a prompt
// that goes on past it.
//
// An entry costs no copy when it is made. It holds second handles to the
// arrays of the cache it was taken from, and MLX's arrays are values: the
// prompt that goes on reading writes new ones, copying a KV buffer the entry
// still holds before writing into it.
//
// Only the engine's run goroutine touches a pool.
type prefixPool struct {
	entries []*prefixEntry
	clock   int64
}

// held is what an entry keeps of a cache; a test keeps something else.
type held interface{ Free() }

type prefixEntry struct {
	tokens  []int32
	cache   held
	shared  bool // a system prompt, which conversations open with
	lastUse int64
}

// minPrefixMatch is the shortest prefix worth pooling; the GGUF engine's.
const minPrefixMatch = 128

func newPrefixPool(n int) *prefixPool {
	if n <= 0 {
		return nil
	}
	p := &prefixPool{}
	for i := 0; i < n; i++ {
		p.entries = append(p.entries, &prefixEntry{})
	}
	return p
}

// match returns the longest entry tokens begins with and goes on past, or
// nil.
func (p *prefixPool) match(tokens []int32) *prefixEntry {
	if p == nil {
		return nil
	}
	var best *prefixEntry
	for _, e := range p.entries {
		n := len(e.tokens)
		if n == 0 || n >= len(tokens) || commonPrefix(e.tokens, tokens) != n {
			continue
		}
		if best == nil || n > len(best.tokens) {
			best = e
		}
	}
	if best != nil {
		p.clock++
		best.lastUse = p.clock
	}
	return best
}

// put pools tokens, taking what to keep of their cache from take when there
// is an entry to keep it in.
func (p *prefixPool) put(tokens []int32, shared bool, take func() held) {
	if p == nil || len(tokens) < minPrefixMatch {
		return
	}
	e := p.victim(tokens, shared)
	if e == nil {
		return
	}
	if e.cache != nil {
		e.cache.Free()
	}
	e.tokens = append(e.tokens[:0], tokens...)
	e.cache, e.shared = take(), shared
	p.clock++
	e.lastUse = p.clock
}

// victim picks the entry to overwrite: an empty one, the one tokens extends,
// or the least recently used conversation, before any system prompt. It
// returns nil when an entry holds tokens already. A system prompt is not
// replaced by a conversation that extends it: it is the one entry every
// other conversation of that agent can still adopt.
func (p *prefixPool) victim(tokens []int32, shared bool) *prefixEntry {
	var lru, lruConversation *prefixEntry
	for _, e := range p.entries {
		if len(e.tokens) == 0 {
			return e
		}
		if n := commonPrefix(e.tokens, tokens); n == len(e.tokens) {
			if n == len(tokens) {
				p.clock++
				e.lastUse = p.clock
				return nil
			}
			if e.shared == shared {
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
	if lruConversation != nil {
		return lruConversation
	}
	return lru
}

// free releases every entry's cache.
func (p *prefixPool) free() {
	if p == nil {
		return
	}
	for _, e := range p.entries {
		if e.cache != nil {
			e.cache.Free()
		}
		*e = prefixEntry{}
	}
}

// checkpoint is a position to pool a prompt at as it is read.
type checkpoint struct {
	pos    int
	shared bool
}

func commonPrefix(a, b []int32) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
