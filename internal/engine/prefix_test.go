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

// A model with recurrent layers keeps one state per sequence, not a cache of
// positions, so a pooled prompt can be adopted only whole — by a prompt that
// continues past its end. Anything shorter would hand the new prompt a state
// that has already read the rest of the old one.
func TestExactPoolAdoptsOnlyWholeEntries(t *testing.T) {
	p, _ := newTestPool(1, 2)
	p.exact = true
	p.entries[0].tokens = toks(1, 2, 3, 4, 5, 6)

	cases := []struct {
		name   string
		prompt []llama.Token
		want   int
	}{
		{"continues past the entry", toks(1, 2, 3, 4, 5, 6, 7, 8), 6},
		{"diverges inside the entry", toks(1, 2, 3, 9, 9, 9, 9), 0},
		// A repeat can only reuse len-1 tokens, one short of the entry.
		{"identical to the entry", toks(1, 2, 3, 4, 5, 6), 0},
		{"shorter than the entry", toks(1, 2, 3, 4), 0},
	}
	for _, c := range cases {
		if _, n := p.match(c.prompt); n != c.want {
			t.Errorf("%s: matched %d tokens, want %d", c.name, n, c.want)
		}
	}
}

func TestExactPoolPicksTheLongestWholeEntry(t *testing.T) {
	p, _ := newTestPool(3, 2)
	p.exact = true
	p.entries[0].tokens = toks(1, 2, 3)
	p.entries[1].tokens = toks(1, 2, 3, 4, 5)
	p.entries[2].tokens = toks(1, 2, 3, 4, 5, 6, 7, 8, 9) // shares 7, not whole

	e, n := p.match(toks(1, 2, 3, 4, 5, 6, 7, 0))
	if e != p.entries[1] || n != 5 {
		t.Errorf("matched entry %v for %d tokens; want the 5-token entry, whole", e, n)
	}
}

// Without recurrent state nothing changes: attention cells can be shared up
// to any position, so a partial prefix is still worth adopting.
func TestAttentionPoolStillSharesPartialPrefixes(t *testing.T) {
	p, _ := newTestPool(1, 2)
	p.entries[0].tokens = toks(1, 2, 3, 4, 5, 6)
	if _, n := p.match(toks(1, 2, 3, 9)); n != 3 {
		t.Errorf("matched %d tokens, want 3", n)
	}
}

// A conversation's entry must not replace the system prompt it extends: every
// other conversation opening with that prompt can still adopt it whole.
func TestExactPoolKeepsTheSystemPromptUnderAConversation(t *testing.T) {
	p, _ := newTestPool(4, 2)
	p.exact = true
	p.publishShared(1, toks(1, 2, 3, 4))
	p.publish(1, toks(1, 2, 3, 4, 5, 6, 7))

	if _, n := p.match(toks(1, 2, 3, 4, 9, 9)); n != 4 {
		t.Errorf("a new conversation adopted %d tokens, want the 4-token system prompt", n)
	}
	if _, n := p.match(toks(1, 2, 3, 4, 5, 6, 7, 8)); n != 7 {
		t.Errorf("the next turn adopted %d tokens, want the 7-token conversation", n)
	}
}

// A conversation's later turn replaces its earlier one, which it covers for
// every prompt that could have used it.
func TestExactPoolReplacesAConversationsEarlierTurn(t *testing.T) {
	p, _ := newTestPool(4, 2)
	p.exact = true
	p.publish(1, toks(1, 2, 3, 4, 5))
	p.publish(1, toks(1, 2, 3, 4, 5, 6, 7, 8))
	used := 0
	for _, e := range p.entries {
		if len(e.tokens) > 0 {
			used++
		}
	}
	if used != 1 {
		t.Errorf("%d entries in use, want 1: the later turn replaces the earlier", used)
	}
}

func TestExactPoolEvictsAConversationBeforeASystemPrompt(t *testing.T) {
	p, _ := newTestPool(2, 2)
	p.exact = true
	p.publishShared(1, toks(1, 2, 3)) // the least recently used
	p.publish(1, toks(7, 8, 9, 10))
	p.publish(1, toks(20, 21, 22)) // needs an entry

	if _, n := p.match(toks(1, 2, 3, 4)); n != 3 {
		t.Errorf("the system prompt was evicted (matched %d)", n)
	}
	if _, n := p.match(toks(7, 8, 9, 10, 11)); n != 0 {
		t.Errorf("the older conversation survived (matched %d); it should have gone first", n)
	}
}
