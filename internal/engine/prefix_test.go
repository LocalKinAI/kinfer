package engine

import (
	"testing"

	"github.com/LocalKinAI/kinfer/internal/llama"
)

// newTestPool builds a pool whose llama.cpp calls are recorded instead of made.
func newTestPool(n, minMatch int) (*prefixPool, *[]string) {
	var calls []string
	p := &prefixPool{minMatch: minMatch}
	for i := 0; i < n; i++ {
		p.entries = append(p.entries, &prefixEntry{seq: int32(100 + i)})
	}
	p.share = func(src, dst int32, n int32) {
		calls = append(calls, "share")
		_ = src
		_ = dst
		_ = n
	}
	p.forget = func(seq int32) { calls = append(calls, "forget") }
	return p, &calls
}

func toks(vals ...int) []llama.Token {
	out := make([]llama.Token, len(vals))
	for i, v := range vals {
		out[i] = llama.Token(v)
	}
	return out
}

func TestCommonPrefix(t *testing.T) {
	cases := []struct {
		a, b []llama.Token
		want int
	}{
		{toks(1, 2, 3), toks(1, 2, 3), 3},
		{toks(1, 2, 3), toks(1, 2, 9), 2},
		{toks(1, 2), toks(1, 2, 3, 4), 2},
		{toks(9), toks(1, 2), 0},
		{nil, toks(1), 0},
	}
	for _, c := range cases {
		if got := commonPrefix(c.a, c.b); got != c.want {
			t.Errorf("commonPrefix(%v, %v) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}

// TestMatchLeavesATokenToDecode guards a subtlety that would hang a request:
// logits exist only for tokens that were decoded, so a prompt adopted in full
// would leave nothing to sample the first reply token from.
func TestMatchLeavesATokenToDecode(t *testing.T) {
	p, _ := newTestPool(2, 2)
	prompt := toks(1, 2, 3, 4, 5)
	p.entries[0].tokens = append([]llama.Token(nil), prompt...)

	e, n := p.match(prompt)
	if e == nil {
		t.Fatal("an identical prompt did not match")
	}
	if n != len(prompt)-1 {
		t.Errorf("matched %d of %d tokens; the last one must still be decoded", n, len(prompt))
	}
}

func TestMatchPicksTheLongestAndRespectsMinimum(t *testing.T) {
	p, _ := newTestPool(3, 3)
	p.entries[0].tokens = toks(1, 2)       // shorter
	p.entries[1].tokens = toks(1, 2, 3, 4) // longest match
	p.entries[2].tokens = toks(9, 9)       // no overlap

	e, n := p.match(toks(1, 2, 3, 4, 5, 6))
	if e != p.entries[1] || n != 4 {
		t.Errorf("matched entry %v at %d tokens, want entries[1] at 4", e, n)
	}

	// Two shared tokens is below minMatch, so it is not worth adopting.
	if e, n := p.match(toks(1, 2, 7, 7, 7)); e != nil || n != 0 {
		t.Errorf("a %d-token match was adopted despite minMatch=%d", n, p.minMatch)
	}
}

// TestPublishExtendsRatherThanDuplicates: a longer prompt that continues where
// an entry stops should replace it, not occupy a second slot with a prefix of
// itself.
func TestPublishExtendsRatherThanDuplicates(t *testing.T) {
	p, _ := newTestPool(3, 2)

	p.publish(0, toks(1, 2, 3))
	p.publish(0, toks(1, 2, 3, 4, 5))

	used := 0
	for _, e := range p.entries {
		if len(e.tokens) > 0 {
			used++
		}
	}
	if used != 1 {
		t.Errorf("two nested prompts occupy %d entries, want 1", used)
	}
	if got := len(p.entries[0].tokens); got != 5 {
		t.Errorf("entry holds %d tokens, want the longer 5", got)
	}
}

func TestPublishSkipsAnIdenticalPrompt(t *testing.T) {
	p, calls := newTestPool(2, 2)
	p.publish(0, toks(1, 2, 3))
	before := len(*calls)
	p.publish(0, toks(1, 2, 3))
	if len(*calls) != before {
		t.Error("re-publishing an identical prompt touched the KV cache again")
	}
}

func TestPublishEvictsLeastRecentlyUsed(t *testing.T) {
	p, _ := newTestPool(2, 2)
	p.publish(0, toks(1, 1, 1))
	p.publish(0, toks(2, 2, 2))

	// Touch the first so the second becomes the least recently used.
	if e, _ := p.match(toks(1, 1, 1, 9)); e != nil {
		p.adopt(e, 0, 3)
	}

	p.publish(0, toks(3, 3, 3))

	var have [][]llama.Token
	for _, e := range p.entries {
		have = append(have, e.tokens)
	}
	if commonPrefix(have[0], toks(1, 1, 1)) != 3 && commonPrefix(have[1], toks(1, 1, 1)) != 3 {
		t.Errorf("the recently used prefix was evicted; pool holds %v", have)
	}
	for _, h := range have {
		if commonPrefix(h, toks(2, 2, 2)) == 3 {
			t.Errorf("the least recently used prefix survived; pool holds %v", have)
		}
	}
}

func TestPublishIgnoresShortPrompts(t *testing.T) {
	p, calls := newTestPool(2, 8)
	p.publish(0, toks(1, 2, 3))
	if len(*calls) != 0 {
		t.Error("a prompt below minMatch was pooled; it would evict something useful for no gain")
	}
}

func TestNilPoolIsInert(t *testing.T) {
	var p *prefixPool
	if e, n := p.match(toks(1, 2, 3)); e != nil || n != 0 {
		t.Error("a disabled pool reported a match")
	}
	p.publish(0, toks(1, 2, 3)) // must not panic
}
