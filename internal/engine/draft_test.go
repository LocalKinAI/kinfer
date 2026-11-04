package engine

import (
	"slices"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// Who drafts: a dense model with a head of one layer, four tokens a step unless
// told otherwise; a mixture of experts one, since more slowed ornith down; and
// nobody whose head is missing, longer than kinfer runs, or beyond the build.
func TestDraftsPolicy(t *testing.T) {
	head := store.Shape{Nextn: 1, MTP: true}
	moe := head
	moe.Experts = 256
	for _, c := range []struct {
		name  string
		sh    store.Shape
		want  int
		can   bool
		n     int
		noted bool // whether a reason is given
	}{
		{"a dense model with a head", head, 0, true, defaultDraft, false},
		{"asked for two", head, 2, true, 2, false},
		{"asked for more than pays", head, 50, true, maxDraft, false},
		{"turned off", head, -1, true, 0, false},
		{"a mixture of experts", moe, 0, true, moeDraft, false},
		{"a mixture of experts, asked", moe, 3, true, 3, false},
		{"no head", store.Shape{}, 0, true, 0, false},
		{"no head, asked", store.Shape{}, 4, true, 0, true},
		{"a head counted, its weights left out", store.Shape{Nextn: 1}, 4, true, 0, true},
		{"a head of two layers", store.Shape{Nextn: 2, MTP: true}, 0, true, 0, true},
		{"a build that cannot", head, 0, false, 0, true},
	} {
		n, why := drafts(c.sh, c.want, c.can)
		if n != c.n || (why != "") != c.noted {
			t.Errorf("%s: %d drafts, reason %q; want %d, a reason %v", c.name, n, why, c.n, c.noted)
		}
	}
}

// Reading a step's tokens: every token sampled goes into the reply, a draft
// counts only while each token before it was the draft, and the reply can end
// on any of them.
func TestCheckDrafts(t *testing.T) {
	const eos = 99
	for _, c := range []struct {
		name     string
		drafts   []llama.Token
		model    []llama.Token // what the model writes after each row
		taken    []llama.Token // what reaches the reply
		accepted int
		goesOn   bool
	}{
		{"no drafts", nil, []llama.Token{7}, []llama.Token{7}, 0, true},
		{"all accepted, and the token after", []llama.Token{1, 2, 3}, []llama.Token{1, 2, 3, 4}, []llama.Token{1, 2, 3, 4}, 3, true},
		{"the first rejected", []llama.Token{1, 2, 3}, []llama.Token{9, 2, 3, 4}, []llama.Token{9}, 0, true},
		{"one accepted", []llama.Token{1, 2, 3}, []llama.Token{1, 8, 3, 4}, []llama.Token{1, 8}, 1, true},
		{"the reply ends on an accepted draft", []llama.Token{1, eos, 3}, []llama.Token{1, eos, 3, 4}, []llama.Token{1, eos}, 1, false},
		{"the reply ends in place of a draft", []llama.Token{1, 2}, []llama.Token{eos, 2, 3}, []llama.Token{eos}, 0, false},
	} {
		var taken []llama.Token
		n, goesOn := checkDrafts(c.drafts,
			func(i int) llama.Token { return c.model[i] },
			func(tok llama.Token, i int) bool {
				if len(taken) != i {
					t.Errorf("%s: token %d taken as the %d-th", c.name, tok, i)
				}
				taken = append(taken, tok)
				return tok != eos
			})
		if n != c.accepted || goesOn != c.goesOn || !slices.Equal(taken, c.taken) {
			t.Errorf("%s: accepted %d, goes on %v, reply %v; want %d, %v, %v",
				c.name, n, goesOn, taken, c.accepted, c.goesOn, c.taken)
		}
	}
}

// Drafts are shared out over the conversations writing, so the pass that checks
// them stays about ten tokens long — but each gets one while it pays.
func TestDraftsEach(t *testing.T) {
	for n, want := range map[int]int{1: 9, 2: 4, 3: 2, 4: 1, 5: 1, 6: 1, 8: 1, 9: 0, 32: 0} {
		if got := draftsEach(n); got != want {
			t.Errorf("draftsEach(%d) = %d, want %d", n, got, want)
		}
	}
}

func TestArgmax(t *testing.T) {
	if got := argmax([]float32{-3, 2.5, 2.5001, -1}); got != 2 {
		t.Errorf("argmax = %d, want 2", got)
	}
	if got := argmax([]float32{-7, -9}); got != 0 {
		t.Errorf("argmax of negatives = %d, want 0", got)
	}
}
