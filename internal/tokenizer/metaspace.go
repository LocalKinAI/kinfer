package tokenizer

import (
	"container/heap"
	"encoding/json"
	"fmt"
)

// spacesAsMeta reports a normalizer that writes every space as ▁ — what
// Gemma's tokenizer.json has, alone or in a Sequence — and nothing else this
// package would have to do.
func spacesAsMeta(raw json.RawMessage) bool {
	var n struct {
		Type        string            `json:"type"`
		Normalizers []json.RawMessage `json:"normalizers"`
		Pattern     struct {
			String string `json:"String"`
		} `json:"pattern"`
		Content string `json:"content"`
	}
	if len(raw) == 0 || json.Unmarshal(raw, &n) != nil {
		return false
	}
	switch n.Type {
	case "Replace":
		return n.Pattern.String == " " && n.Content == "▁"
	case "Sequence":
		return len(n.Normalizers) == 1 && spacesAsMeta(n.Normalizers[0])
	}
	return false
}

// metaPretokenizer is what a SentencePiece-style file splits its normalized
// text by: nothing, or a Split on " " — which, the normalizer having written
// every space as ▁, finds nothing to split at either.
func metaPretokenizer(raw json.RawMessage) (func(string) []string, error) {
	whole := func(s string) []string { return []string{s} }
	if len(raw) == 0 || string(raw) == "null" {
		return whole, nil
	}
	var p struct {
		Type    string `json:"type"`
		Pattern struct {
			String string `json:"String"`
		} `json:"pattern"`
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, err
	}
	if p.Type == "Split" && p.Pattern.String == " " {
		return whole, nil
	}
	return nil, fmt.Errorf("pre-tokenizer %q after a ▁ normalizer is not supported", p.Type)
}

// byteToken reads a byte-fallback piece, <0x41>.
func byteToken(p string) (byte, bool) {
	if len(p) != 6 || p[:3] != "<0x" || p[5] != '>' {
		return 0, false
	}
	var b byte
	for _, c := range p[3:5] {
		switch {
		case c >= '0' && c <= '9':
			b = b<<4 | byte(c-'0')
		case c >= 'A' && c <= 'F':
			b = b<<4 | byte(c-'A'+10)
		default:
			return 0, false
		}
	}
	return b, true
}

// mergeChars is BPE over the characters of word, as the tokenizers library's
// BPE does it: the lowest-ranked pair anywhere merges first, the leftmost of
// equals, and a character outside the vocabulary is spelled in byte tokens
// before anything merges. A SentencePiece-style file merges a whole text as
// one word, so this keeps its pairs in a heap rather than scanning for the
// best one after every merge.
func (t *Tokenizer) mergeChars(word string) []int32 {
	if id, ok := t.vocab[word]; ok {
		return []int32{id}
	}
	short := len(word) <= 64
	if short {
		t.mu.Lock()
		ids, ok := t.cache[word]
		t.mu.Unlock()
		if ok {
			return ids
		}
	}

	var syms []symbol
	for _, r := range word {
		c := string(r)
		if _, ok := t.vocab[c]; !ok && t.byteFallback {
			for _, b := range []byte(c) {
				syms = append(syms, symbol{s: fmt.Sprintf("<0x%02X>", b)})
			}
			continue
		}
		syms = append(syms, symbol{s: c})
	}
	for i := range syms {
		syms[i].prev, syms[i].next = i-1, i+1
	}
	if len(syms) > 0 {
		syms[len(syms)-1].next = -1
	}

	q := &pairs{}
	push := func(l int) {
		if l < 0 || syms[l].next < 0 {
			return
		}
		r := syms[l].next
		if rank, ok := t.ranks[[2]string{syms[l].s, syms[r].s}]; ok {
			heap.Push(q, pair{rank: rank, pos: l, merged: syms[l].s + syms[r].s})
		}
	}
	for i := range syms {
		push(i)
	}
	for q.Len() > 0 {
		p := heap.Pop(q).(pair)
		l := p.pos
		r := syms[l].next
		// A pair from before a neighbour merged: it is no longer there.
		if syms[l].gone || r < 0 || syms[l].s+syms[r].s != p.merged {
			continue
		}
		syms[l].s = p.merged
		syms[r].gone = true
		syms[l].next = syms[r].next
		if n := syms[l].next; n >= 0 {
			syms[n].prev = l
		}
		if pv := syms[l].prev; pv >= 0 {
			push(pv)
		}
		push(l)
	}

	ids := make([]int32, 0, len(syms))
	for i := 0; i >= 0 && i < len(syms); i = syms[i].next {
		if id, ok := t.vocab[syms[i].s]; ok {
			ids = append(ids, id)
		} else if t.unk >= 0 {
			ids = append(ids, t.unk)
		}
	}
	if short {
		t.mu.Lock()
		if len(t.cache) < 1<<16 {
			t.cache[word] = ids
		}
		t.mu.Unlock()
	}
	return ids
}

type symbol struct {
	s          string
	prev, next int
	gone       bool
}

// pair is a merge waiting in the heap: the symbol at pos with the one after
// it, ranked; merged is what the two make, which is also how a stale entry
// is told from a live one.
type pair struct {
	rank, pos int
	merged    string
}

type pairs []pair

func (h pairs) Len() int { return len(h) }
func (h pairs) Less(i, j int) bool {
	if h[i].rank != h[j].rank {
		return h[i].rank < h[j].rank
	}
	return h[i].pos < h[j].pos
}
func (h pairs) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *pairs) Push(x any)   { *h = append(*h, x.(pair)) }
func (h *pairs) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}
