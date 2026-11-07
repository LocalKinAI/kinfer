package engine

import (
	"fmt"
	"log"

	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// Drafting with the model's own MTP head.
//
// A model trained with multi-token prediction carries a small head beside it —
// one more layer, on Qwen3.8-27B — that reads the hidden state the model left
// at a token, with the token after it, and guesses the one after that. Run a
// few times in a row it drafts the next several tokens for a fraction of a
// pass, and the model then reads them all in one: a pass of five tokens costs
// a dense model about what a pass of one does, because either way it reads
// every weight once. Each draft the model would have written itself is kept;
// the first it would not is replaced by what it did write, so every token is
// one the model chose, and Qwen3.8-27B's replies were token for token what it
// writes without drafts (a mixture of experts' need not be: see moeDraft).
// Measured on the box: Qwen3.8-27B from 31.5 tok/s to 37 on prose and 69 on
// code, accepting exactly the drafts llama-server's own MTP drafting accepts
// on the same prompts.
//
// The head has a cache of its own, in a context of its own over the same
// weights (llama.cpp's LLAMA_CONTEXT_TYPE_MTP), and it has to read every token
// the model reads, paired with the model's hidden state at the token before:
// catchUp does that after every pass. That cache mirrors the model's — the
// same sequences, the same positions — and everything that moves the model's
// cells moves the head's too: a slot taking a request, the prefix pool, a
// rejected draft.
//
// A mixture of experts drafts one token a step unless asked for more (-draft):
// every token of a pass fetches its own experts, so each draft costs nearly a
// token's reading, and the head's guesses past the first are seldom right.
// ornith-1.5-35b, on the same box, accepted two drafts in three or more at one
// a step and went from 108 tok/s to 118-131; at four it accepted one in five
// and fell to 78-85. See the scheduler's comment.

// defaultDraft is how many tokens a step drafts when nobody said: measured on
// Qwen3.8-27B, 49 tok/s at three, 53 at four and 54.7 at six, with acceptance
// falling from 67-77% to 59-72% between three and four. Ollama drafts four.
const defaultDraft = 4

// moeDraft is the same for a mixture of experts, which gains only from the
// first draft: ornith-1.5-35b wrote 118-131 tok/s at one, 101-120 at two, 108
// without. It is also the most for which a pass reads exactly what reading the
// tokens one at a time would: past two tokens a pass, Metal multiplies some
// weights (Q4_K, at three rows) with a few-row kernel that sums in another
// order, and the expert router turns that last-bit difference into a different
// word where the model was torn — ornith's replies at two drafts or more were
// the same as without in 2 of 5, "insert a key" becoming "insert a key-value
// pair".
const moeDraft = 1

// draftPass is about how many tokens the pass that checks drafts should read,
// the conversations writing shared out: four drafts each for one or two, two
// for three, one for four to eight. A pass of five tokens costs a dense model
// little more than a pass of one — 54 ms against 32 on Qwen3.8-27B — but past
// that each token costs about 7 ms more, draft or not, and every conversation
// in the pass pays: four at once with four drafts each wrote 42.6 tok/s
// together, against 61.8 without drafts, and 89.3 with one draft each.
const draftPass = 10

// draftingMost is how many conversations may write at once for drafts to be
// made at all. A draft the model accepts nine times in ten still paid at eight
// — 110.7 tok/s together against 102.7 — where none at all, the head still
// reading every token, had cost six at once 7%. Past eight it stops reading;
// what is left to pay is a hybrid's snapshots, written every pass: ten at once
// wrote 85.6 tok/s together, against 92.1 with -draft -1.
const draftingMost = 8

// draftsEach is how many tokens to draft for each of n conversations writing,
// before -draft and each one's room bound it.
func draftsEach(n int) int {
	if n > draftingMost {
		return 0
	}
	return max(draftPass/max(n, 1)-1, 1)
}

// maxDraft bounds -draft: each draft is a snapshot of a hybrid model's
// recurrent state kept for every sequence, and past six the gain was 3%.
const maxDraft = 8

// Drafts is how many tokens a step drafts for a model of shape sh, asked for
// want (0 for the default, negative for none), and why not, when it does not
// draft what was asked or a model with a head would otherwise draft.
func Drafts(sh store.Shape, want int) (int, string) {
	return drafts(sh, want, llama.CanDraft())
}

// drafts is Drafts for a build of llama.cpp that can draft or cannot.
func drafts(sh store.Shape, want int, can bool) (int, string) {
	switch {
	case want < 0:
		return 0, ""
	case !sh.MTP || sh.Nextn == 0:
		if want > 0 {
			return 0, "it has no MTP head to draft with"
		}
		return 0, ""
	case sh.Nextn != 1:
		// llama.cpp runs a head of several layers one layer per draft
		// (step35), which kinfer does not do yet.
		return 0, fmt.Sprintf("its MTP head has %d layers, and kinfer drafts only with one", sh.Nextn)
	case !can:
		return 0, "this build of llama.cpp cannot draft with an MTP head"
	case want > 0:
		return min(want, maxDraft), ""
	case sh.Experts > 0:
		return moeDraft, ""
	}
	return defaultDraft, ""
}

// drafter drafts for every generating slot of one context; only the
// scheduler's goroutine uses it.
type drafter struct {
	ctx   llama.Context // the head's, over the same model
	main  llama.Context // the model's, whose hidden states it reads
	batch llama.ExtBatch
	dims  int
	vocab int32
	most  int // tokens drafted per step, at most

	// Per sequence id — the slots', then the pool's: the hidden state the
	// model left at the last token of the sequence it read, and the position
	// of that token, which the next token read is paired with; and whether
	// the head's cache holds the sequence as the model's does. A sequence
	// read some other way — a prompt with pictures, or a pass the head failed
	// — is not drafted for until its slot takes the next request.
	state   [][]float32
	at      []int32
	tracked []bool
	zero    []float32

	// failed is set once a pass of the head has failed, so it is said once.
	failed bool
}

// newDrafter drafts up to most tokens a step for main's sequences, with the
// head's context ctx, which it owns from here on.
func newDrafter(main, ctx llama.Context, model llama.Model, seqs, most int) (*drafter, error) {
	b := llama.NewExtBatch(ctx)
	if b == 0 {
		return nil, fmt.Errorf("no batch for the head's context")
	}
	llama.SetNextn(main, true, false)
	llama.SetNextn(ctx, true, true)
	d := &drafter{
		ctx:     ctx,
		main:    main,
		batch:   b,
		dims:    llama.EmbedDims(model),
		vocab:   llama.NVocab(llama.GetVocab(model)),
		most:    most,
		state:   make([][]float32, seqs),
		at:      make([]int32, seqs),
		tracked: make([]bool, seqs),
	}
	d.zero = make([]float32, d.dims)
	for i := range d.state {
		d.state[i] = make([]float32, d.dims)
		d.at[i] = -1
		d.tracked[i] = true
	}
	return d, nil
}

// free releases the head's context. Safe on nil.
func (d *drafter) free() {
	if d == nil {
		return
	}
	d.batch.Free()
	llama.FreeContext(d.ctx)
	d.ctx = 0
}

// forget empties a sequence, as llama.ForgetSequence does the model's.
func (d *drafter) forget(seq int32) {
	if d == nil {
		return
	}
	llama.ForgetSequence(d.ctx, seq, -1, -1)
	clear(d.state[seq])
	d.at[seq] = -1
	d.tracked[seq] = true
}

// move mirrors the scheduler moving a reply from sequence src to dst, whose
// cells it has already forgotten (compact): the head's cache, the hidden state
// and the position go over, and src is left as forget leaves a sequence.
func (d *drafter) move(src, dst int32) {
	if d == nil {
		return
	}
	llama.CopySequence(d.ctx, src, dst, -1, -1)
	llama.ForgetSequence(d.ctx, src, -1, -1)
	d.state[src], d.state[dst] = d.state[dst], d.state[src]
	d.at[dst], d.tracked[dst] = d.at[src], d.tracked[src]
	clear(d.state[src])
	d.at[src] = -1
	d.tracked[src] = true
}

// untrack stops drafting for a sequence, and the head reading it: one read some
// other way, or not worth drafting for.
func (d *drafter) untrack(seq int32) {
	if d == nil {
		return
	}
	d.tracked[seq] = false
}

// share mirrors the prefix pool tagging dst with src's first n positions —
// over p0, p1, as the model's cells were. The hidden state goes with them when
// it is the one at the last of those positions; otherwise the next token read
// is paired with nothing, as a prompt's first token is.
func (d *drafter) share(src, dst, n, p0, p1 int32) {
	if d == nil {
		return
	}
	llama.CopySequence(d.ctx, src, dst, p0, p1)
	d.tracked[dst] = d.tracked[src]
	if d.at[src] == n-1 {
		copy(d.state[dst], d.state[src])
	} else {
		clear(d.state[dst])
	}
	d.at[dst] = n - 1
}

// catchUp has the head read what the model just read: every token of b,
// paired with the model's hidden state at the token before it.
func (d *drafter) catchUp(b *llama.BatchBuilder) {
	n := b.Len()
	rows := llama.NextnRows(d.main, n, d.dims)
	if rows == nil {
		return
	}
	row := func(k int) []float32 { return rows[k*d.dims : (k+1)*d.dims] }

	d.batch.Clear()
	added := false
	prev := int32(-1) // the sequence of the token before, in b
	for k := range n {
		tok, pos, seq := b.At(k)
		follows := seq == prev
		prev = seq
		if !d.tracked[seq] {
			continue
		}
		// A sequence's tokens are consecutive in a batch, so the one before
		// is in it unless this is the sequence's first.
		state := d.zero
		switch {
		case follows:
			state = row(k - 1)
		case d.at[seq] == pos-1:
			state = d.state[seq]
		}
		if _, err := d.batch.Add(tok, pos, seq, false, state); err != nil {
			d.fail(b, err)
			return
		}
		added = true
	}
	if !added {
		return
	}
	if err := llama.Process(d.ctx, d.batch); err != nil {
		d.fail(b, err)
		return
	}
	for k := range n {
		_, pos, seq := b.At(k)
		if !d.tracked[seq] {
			continue
		}
		if k+1 < n {
			if _, _, next := b.At(k + 1); next == seq {
				continue // not the sequence's last token
			}
		}
		copy(d.state[seq], row(k))
		d.at[seq] = pos
	}
}

// fail stops drafting for every sequence of b, whose cells in the head's
// cache are no longer the model's.
func (d *drafter) fail(b *llama.BatchBuilder, err error) {
	if !d.failed {
		log.Printf("draft: the MTP head could not read a batch (%v) — its conversations go on without drafts", err)
		d.failed = true
	}
	for k := range b.Len() {
		_, _, seq := b.At(k)
		d.tracked[seq] = false
	}
}

// drafting is one slot's part in a round of drafts.
type drafting struct {
	sl  *slot
	idx int32 // its row in the head's batch
	pos int32 // the position of the token there
	max int   // drafts it may take
}

// propose drafts for each slot as many tokens as its allowance says, into
// sl.drafts, with one pass of the head per token drafted.
func (d *drafter) propose(slots []*slot, allowance func(*slot) int) {
	for _, sl := range slots {
		sl.drafts = sl.drafts[:0]
	}
	var ds []drafting
	d.batch.Clear()
	for _, sl := range slots {
		k := min(allowance(sl), d.most)
		if k <= 0 || !d.tracked[sl.seq] || d.at[sl.seq] != sl.nPast-1 {
			continue
		}
		// The token the slot reads next, with the state it follows.
		idx, err := d.batch.Add(sl.next, sl.nPast, sl.seq, true, d.state[sl.seq])
		if err != nil {
			break
		}
		ds = append(ds, drafting{sl: sl, idx: idx, pos: sl.nPast, max: k})
	}
	began := make([]*slot, len(ds))
	for i, g := range ds {
		began[i] = g.sl
	}

	for len(ds) > 0 {
		if err := llama.Process(d.ctx, d.batch); err != nil {
			break // what was drafted so far is still worth checking
		}
		d.batch.Clear()
		going := ds[:0]
		for _, g := range ds {
			tok := argmax(llama.Logits(d.ctx, g.idx, d.vocab))
			g.sl.drafts = append(g.sl.drafts, tok)
			if len(g.sl.drafts) >= g.max {
				continue
			}
			h := llama.NextnRow(d.ctx, g.idx, d.dims)
			idx, err := d.batch.Add(tok, g.pos+1, g.sl.seq, true, h)
			if err != nil {
				continue
			}
			g.idx, g.pos = idx, g.pos+1
			going = append(going, g)
		}
		ds = going
	}

	// The drafts leave the head's cache: it reads them again, with the model's
	// own states, once the model has.
	for _, sl := range began {
		if !llama.TrimSequence(d.ctx, sl.seq, sl.nPast) {
			d.tracked[sl.seq] = false
		}
	}
}

// settle keeps the part of a slot's drafts the model accepted: the token at
// position last is the last kept, read at row of the model's batch, and what
// the head read after it goes.
func (d *drafter) settle(seq, last, row int32) {
	if !d.tracked[seq] {
		return
	}
	if !llama.TrimSequence(d.ctx, seq, last+1) {
		d.tracked[seq] = false
		return
	}
	copy(d.state[seq], llama.NextnRow(d.main, row, d.dims))
	d.at[seq] = last
}

// argmax is the most likely token of a row of logits.
func argmax(logits []float32) llama.Token {
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return llama.Token(best)
}

// checkDrafts reads a step's tokens for one slot: sample(i) is the token the
// model writes after row i of what it read — the slot's next token, then each
// draft — and take adds it to the reply, false when that ends it. Each token
// that is the draft read after it lets the next row count; the first that is
// not ends the step, as does the last row. It returns how many drafts the
// model accepted, and whether the reply goes on.
func checkDrafts(drafts []llama.Token, sample func(i int) llama.Token, take func(tok llama.Token, i int) bool) (accepted int, goesOn bool) {
	for i := 0; ; i++ {
		tok := sample(i)
		if !take(tok, i) {
			return i, false
		}
		if i == len(drafts) || tok != drafts[i] {
			return i, true
		}
	}
}
