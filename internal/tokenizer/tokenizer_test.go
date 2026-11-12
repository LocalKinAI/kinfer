package tokenizer

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"slices"
	"testing"
	"time"
)

// Needs KINFER_TOKENIZER (a tokenizer.json) and KINFER_TOKENIZER_REF
// (ref_tok.py's encodings of test texts with the tokenizers library).
func TestMatchesHuggingFace(t *testing.T) {
	path, refPath := os.Getenv("KINFER_TOKENIZER"), os.Getenv("KINFER_TOKENIZER_REF")
	if path == "" || refPath == "" {
		t.Skip("KINFER_TOKENIZER and KINFER_TOKENIZER_REF not both set")
	}
	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text string  `json:"text"`
		IDs  []int32 `json:"ids"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	bad, tokens := 0, 0
	for _, c := range cases {
		got := tok.Encode(c.Text, true)
		tokens += len(got)
		if !slices.Equal(got, c.IDs) {
			bad++
			if bad <= 8 {
				t.Errorf("%q:\n got  %v\n want %v", c.Text, got, c.IDs)
			}
		}
		if back := tok.Decode(c.IDs); back != c.Text {
			bad++
			if bad <= 8 {
				t.Errorf("decode of %q gave %q", c.Text, back)
			}
		}
	}
	t.Logf("%d texts, %d tokens, %d mismatches", len(cases), tokens, bad)
}

// naiveBPE is the merge bpe did before: the best pair found by scanning the
// whole word after every merge.
func naiveBPE(t *Tokenizer, word string) []int32 {
	if id, ok := t.vocab[word]; ok { // as bpe has it, before and since
		return []int32{id}
	}
	var parts []string
	for _, r := range word {
		parts = append(parts, string(r))
	}
	for len(parts) > 1 {
		best, bestRank := -1, 0
		for i := 0; i+1 < len(parts); i++ {
			if r, ok := t.ranks[[2]string{parts[i], parts[i+1]}]; ok && (best < 0 || r < bestRank) {
				best, bestRank = i, r
			}
		}
		if best < 0 {
			break
		}
		parts[best] += parts[best+1]
		parts = append(parts[:best+1], parts[best+2:]...)
	}
	var ids []int32
	for _, p := range parts {
		if id, ok := t.vocab[p]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

// randomBPE is a vocabulary of the letters of alphabet and n merges of tokens
// already in it, ranked in the order they were made.
func randomBPE(rng *rand.Rand, alphabet string, n int) *Tokenizer {
	t := &Tokenizer{vocab: map[string]int32{}, ranks: map[[2]string]int{}, cache: map[string][]int32{}}
	var toks []string
	for _, r := range alphabet {
		t.vocab[string(r)] = int32(len(t.vocab))
		toks = append(toks, string(r))
	}
	for len(t.ranks) < n {
		a, b := toks[rng.Intn(len(toks))], toks[rng.Intn(len(toks))]
		if _, ok := t.ranks[[2]string{a, b}]; ok || len(a+b) > 8 {
			continue
		}
		t.ranks[[2]string{a, b}] = len(t.ranks)
		if _, ok := t.vocab[a+b]; !ok {
			t.vocab[a+b] = int32(len(t.vocab))
			toks = append(toks, a+b)
		}
	}
	return t
}

// The heap merge gives exactly what the scan did, on words of every length.
func TestBPEMergesAsTheScanDid(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 20; trial++ {
		tk := randomBPE(rng, "abcdé中", 60)
		letters := []rune("abcdé中")
		for w := 0; w < 300; w++ {
			n := 1 + rng.Intn(80)
			word := make([]rune, n)
			for i := range word {
				word[i] = letters[rng.Intn(len(letters))]
			}
			want := naiveBPE(tk, string(word))
			tk.cache = map[string][]int32{}
			if got := tk.bpe(string(word)); fmt.Sprint(got) != fmt.Sprint(want) {
				t.Fatalf("trial %d, %q: %v, the scan gave %v", trial, string(word), got, want)
			}
		}
	}
}

// One word of tens of thousands of characters — a run of text with no space
// in it — merges in well under a second.
func TestBPEOfALongWordIsQuick(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	tk := randomBPE(rng, "abcd", 40)
	word := make([]byte, 30000)
	for i := range word {
		word[i] = "abcd"[rng.Intn(4)]
	}
	start := time.Now()
	tk.bpe(string(word))
	if el := time.Since(start); el > time.Second {
		t.Errorf("a 30,000-character word took %v", el)
	}
}
