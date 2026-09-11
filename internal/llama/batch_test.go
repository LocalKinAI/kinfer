package llama

import (
	"testing"
	"unsafe"
)

// The batch is where continuous batching is easiest to get silently wrong: C
// reads seq_id as llama_seq_id**, an array of pointers to per-token arrays, and
// a builder that pointed every token at the same sequence would put all the
// conversations in one KV stream without erroring. These tests cover the layout
// and the bookkeeping; whether llama.cpp likes the result is covered end to end
// by the isolation checks in the scheduler.

func TestBatchBuilderLayout(t *testing.T) {
	b := NewBatchBuilder(4)

	if got := b.Cap(); got != 4 {
		t.Errorf("Cap() = %d, want 4", got)
	}
	if got := b.Len(); got != 0 {
		t.Errorf("a fresh builder holds %d tokens, want 0", got)
	}

	// Two sequences interleaved, which is what a real step looks like.
	if idx := b.Add(100, 0, 0, false); idx != 0 {
		t.Errorf("first Add returned index %d, want 0", idx)
	}
	if idx := b.Add(200, 7, 3, true); idx != 1 {
		t.Errorf("second Add returned index %d, want 1", idx)
	}

	if b.Len() != 2 {
		t.Fatalf("Len() = %d, want 2", b.Len())
	}
	if b.tokens[0] != 100 || b.tokens[1] != 200 {
		t.Errorf("tokens = %v, want [100 200]", b.tokens[:2])
	}
	if b.pos[0] != 0 || b.pos[1] != 7 {
		t.Errorf("pos = %v, want [0 7]", b.pos[:2])
	}
	if b.seqIDs[0] != 0 || b.seqIDs[1] != 3 {
		t.Errorf("seqIDs = %v, want [0 3]", b.seqIDs[:2])
	}
	if b.logits[0] != 0 || b.logits[1] != 1 {
		t.Errorf("logits = %v, want [0 1] — only the token whose output is read may be marked", b.logits[:2])
	}

	// n_seq_id must be 1 for every slot, always: C reads exactly this many
	// entries from seq_id[i], and a zero would make llama.cpp read none.
	for i, n := range b.nSeqID {
		if n != 1 {
			t.Errorf("nSeqID[%d] = %d, want 1", i, n)
		}
	}

	// Each seq_id pointer must address its own element. Pointing them all at
	// the same one would merge every conversation into one KV stream, silently.
	for i := range b.seqIDs {
		want := uintptr(unsafe.Pointer(&b.seqIDs[i]))
		if b.seqPtrs[i] != want {
			t.Fatalf("seqPtrs[%d] does not point at seqIDs[%d]", i, i)
		}
	}
}

func TestBatchBuilderResetAndOverflow(t *testing.T) {
	b := NewBatchBuilder(2)
	b.Add(1, 0, 0, true)
	b.Add(2, 0, 1, true)

	if idx := b.Add(3, 0, 2, true); idx != -1 {
		t.Errorf("Add past capacity returned %d, want -1 — a silent drop would lose a slot's turn", idx)
	}
	if b.Len() != 2 {
		t.Errorf("a rejected Add changed Len to %d, want 2", b.Len())
	}

	b.Reset()
	if b.Len() != 0 {
		t.Errorf("Len() after Reset = %d, want 0", b.Len())
	}
	if idx := b.Add(9, 4, 1, false); idx != 0 {
		t.Errorf("Add after Reset returned index %d, want 0", idx)
	}
	if b.logits[0] != 0 {
		t.Error("Reset left a stale logits flag, which would compute an output nobody reads")
	}
}
