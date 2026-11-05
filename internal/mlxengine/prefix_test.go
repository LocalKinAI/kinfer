package mlxengine

import "testing"

// fake is a pooled cache that records being freed.
type fake struct {
	name  string
	freed *[]string
}

func (f fake) Free() { *f.freed = append(*f.freed, f.name) }

func seq(base int32, n int) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = base + int32(i)
	}
	return out
}

func cat(parts ...[]int32) []int32 {
	var out []int32
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

func TestPoolAdoptsOnlyWholeEntriesAPromptGoesOnFrom(t *testing.T) {
	var freed []string
	p := newPrefixPool(4)
	sys, turn := seq(1000, 200), seq(5000, 300)
	p.put(sys, true, func() held { return fake{"sys", &freed} })
	p.put(cat(sys, turn), false, func() held { return fake{"conv", &freed} })

	if e := p.match(cat(sys, turn, seq(9000, 50))); e == nil || len(e.tokens) != 500 {
		t.Errorf("the next turn should adopt the conversation, got %v", e)
	}
	if e := p.match(cat(sys, seq(7000, 50))); e == nil || len(e.tokens) != 200 {
		t.Errorf("another conversation should adopt the system prompt, got %v", e)
	}
	// The conversation shares 250 tokens with this prompt, but its state has
	// read all 500: only the system prompt can be adopted.
	if e := p.match(cat(sys, turn[:50], seq(8000, 50))); e == nil || len(e.tokens) != 200 {
		t.Errorf("an edited history should fall back to the system prompt, got %v", e)
	}
	// A prompt that is exactly an entry leaves nothing to read for its first
	// token.
	if e := p.match(cat(sys)); e != nil {
		t.Errorf("a prompt that is the entry adopted it: %v", e)
	}
	if e := p.match(seq(1, 400)); e != nil {
		t.Errorf("an unrelated prompt adopted %v", e)
	}
}

func TestPoolKeepsSystemPromptAndReplacesEarlierTurns(t *testing.T) {
	var freed []string
	p := newPrefixPool(2)
	sys := seq(1000, 200)
	p.put(sys, true, func() held { return fake{"sys", &freed} })
	t1 := cat(sys, seq(5000, 100))
	p.put(t1, false, func() held { return fake{"turn1", &freed} })
	// Turn 2 extends turn 1: it takes turn 1's entry, not the system prompt's.
	t2 := cat(t1, seq(6000, 100))
	p.put(t2, false, func() held { return fake{"turn2", &freed} })
	if len(freed) != 1 || freed[0] != "turn1" {
		t.Fatalf("freed %v, want turn1 only", freed)
	}
	// A second conversation evicts the first, never the system prompt.
	p.put(cat(sys, seq(7000, 100)), false, func() held { return fake{"other", &freed} })
	if len(freed) != 2 || freed[1] != "turn2" {
		t.Fatalf("freed %v, want turn1 then turn2", freed)
	}
	if e := p.match(cat(sys, seq(8000, 10))); e == nil || !e.shared {
		t.Errorf("the system prompt is gone: %v", e)
	}
	// The same prompt again is pooled already; nothing is taken.
	p.put(sys, true, func() held { t.Error("took a cache for a prompt already pooled"); return fake{"x", &freed} })
	p.free()
	if len(freed) != 4 {
		t.Errorf("free released %v, want every entry", freed)
	}
}

func TestPoolIgnoresShortPromptsAndNilIsInert(t *testing.T) {
	var freed []string
	p := newPrefixPool(2)
	p.put(seq(1, minPrefixMatch-1), false, func() held { t.Error("pooled a short prompt"); return fake{"x", &freed} })
	var none *prefixPool
	none.put(seq(1, 500), false, func() held { t.Error("a nil pool pooled"); return fake{"x", &freed} })
	if none.match(seq(1, 500)) != nil {
		t.Error("a nil pool matched")
	}
	none.free()
	if newPrefixPool(0) != nil || newPrefixPool(-1) != nil {
		t.Error("a pool of no entries is not nil")
	}
}
