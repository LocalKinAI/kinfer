// Package tokenizer reads a Hugging Face tokenizer.json and encodes and
// decodes the way the tokenizers library does, for the byte-level BPE models
// MLX releases ship (Qwen, Llama 3 and their relatives). llama.cpp tokenizes
// GGUF models from the vocabulary inside the file; an MLX model directory has
// only tokenizer.json, so the MLX backend needs this.
//
// What is covered: added tokens (special or not) matched before anything
// else, a Split pre-tokenizer whose pattern is one of the GPT-style ones below,
// ByteLevel, and BPE with merges. And the SentencePiece-style files Gemma's
// releases ship: a normalizer that writes spaces as ▁, no split, and BPE over
// characters that falls back to <0xXX> byte tokens for one it does not know
// (see metaspace). The tokenizers library also applies the
// NFC normalizer these files name; the Go standard library has no Unicode
// normalization, and text that is already NFC — nearly all text — is
// unchanged by it, so this skips it.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// Tokenizer encodes text to token ids and back.
type Tokenizer struct {
	vocab   map[string]int32
	pieces  []string // id → byte-level piece
	ranks   map[[2]string]int
	added   []added // longest first
	special map[int32]bool
	split   func(string) []string

	// metaspace is a SentencePiece-style file: spaces become ▁ before
	// anything is merged, the text is merged as one word, and the pieces are
	// characters, not GPT-2's byte characters. byteFallback spells a character
	// the vocabulary lacks as its UTF-8 bytes, <0xE4><0xB8>…, as the
	// tokenizers library's BPE does with byte_fallback.
	metaspace    bool
	byteFallback bool
	unk          int32

	mu    sync.Mutex
	cache map[string][]int32
}

type added struct {
	content string
	id      int32
	special bool
}

type fileFormat struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        struct {
		Type         string           `json:"type"`
		Vocab        map[string]int32 `json:"vocab"`
		Merges       json.RawMessage  `json:"merges"`
		ByteFallback bool             `json:"byte_fallback"`
		UnkToken     string           `json:"unk_token"`
	} `json:"model"`
}

// The pre-tokenizer patterns this package knows, as they appear in
// tokenizer.json. Go's regexp cannot run them (they use (?!\S)), so each has
// a hand-written scanner that follows the same alternatives in the same
// order.
const (
	qwen35Pattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	qwen2Pattern  = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
)

// Load reads tokenizer.json at path.
func Load(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parse(raw, path)
}

func parse(raw []byte, path string) (*Tokenizer, error) {
	var f fileFormat
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if f.Model.Type != "BPE" {
		return nil, fmt.Errorf("%s: model type %q, only BPE is supported", path, f.Model.Type)
	}
	metaspace := spacesAsMeta(f.Normalizer)
	var split func(string) []string
	var err error
	if metaspace {
		split, err = metaPretokenizer(f.PreTokenizer)
	} else {
		split, err = pretokenizer(f.PreTokenizer)
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	t := &Tokenizer{vocab: f.Model.Vocab, ranks: map[[2]string]int{}, special: map[int32]bool{},
		split: split, cache: map[string][]int32{},
		metaspace: metaspace, byteFallback: f.Model.ByteFallback, unk: -1}
	if id, ok := f.Model.Vocab[f.Model.UnkToken]; ok && f.Model.UnkToken != "" {
		t.unk = id
	}

	// Merges are "a b" strings in older files and ["a", "b"] pairs in newer.
	var asStrings []string
	var asPairs [][2]string
	if err := json.Unmarshal(f.Model.Merges, &asStrings); err == nil {
		for i, m := range asStrings {
			a, b, ok := strings.Cut(m, " ")
			if !ok {
				return nil, fmt.Errorf("%s: merge %d %q has no space", path, i, m)
			}
			t.ranks[[2]string{a, b}] = i
		}
	} else if err := json.Unmarshal(f.Model.Merges, &asPairs); err == nil {
		for i, p := range asPairs {
			t.ranks[p] = i
		}
	} else {
		return nil, fmt.Errorf("%s: merges are neither strings nor pairs", path)
	}

	n := int32(0)
	for _, id := range t.vocab {
		n = max(n, id+1)
	}
	for _, a := range f.AddedTokens {
		n = max(n, a.ID+1)
	}
	t.pieces = make([]string, n)
	for p, id := range t.vocab {
		t.pieces[id] = p
	}
	for _, a := range f.AddedTokens {
		t.pieces[a.ID] = a.Content
		t.added = append(t.added, added{a.Content, a.ID, a.Special})
		if a.Special {
			t.special[a.ID] = true
		}
	}
	sort.Slice(t.added, func(i, j int) bool { return len(t.added[i].content) > len(t.added[j].content) })
	return t, nil
}

func pretokenizer(raw json.RawMessage) (func(string) []string, error) {
	var pt struct {
		Type          string            `json:"type"`
		Pretokenizers []json.RawMessage `json:"pretokenizers"`
	}
	if err := json.Unmarshal(raw, &pt); err != nil {
		return nil, err
	}
	steps := []json.RawMessage{raw}
	if pt.Type == "Sequence" {
		steps = pt.Pretokenizers
	}
	var split func(string) []string
	for _, s := range steps {
		var p struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
			Behavior string `json:"behavior"`
			UseRegex bool   `json:"use_regex"`
		}
		if err := json.Unmarshal(s, &p); err != nil {
			return nil, err
		}
		switch p.Type {
		case "Split":
			if p.Behavior != "Isolated" {
				return nil, fmt.Errorf("Split behaviour %q is not supported", p.Behavior)
			}
			switch p.Pattern.Regex {
			case qwen35Pattern:
				split = func(s string) []string { return scan(s, true) }
			case qwen2Pattern:
				split = func(s string) []string { return scan(s, false) }
			default:
				return nil, fmt.Errorf("pre-tokenizer pattern %q is not supported", p.Pattern.Regex)
			}
		case "ByteLevel":
			if p.UseRegex {
				return nil, fmt.Errorf("ByteLevel with its own regex is not supported")
			}
		default:
			return nil, fmt.Errorf("pre-tokenizer %q is not supported", p.Type)
		}
	}
	if split == nil {
		return nil, fmt.Errorf("no Split pre-tokenizer")
	}
	return split, nil
}

// Encode turns text into token ids. With special true, added tokens written
// in the text (such as <|im_start|> in a rendered chat template) become their
// ids; with it false only non-special added tokens do, and special-token text
// is tokenized as plain text — what keeps a user from injecting a turn
// boundary.
func (t *Tokenizer) Encode(text string, special bool) []int32 {
	var ids []int32
	for len(text) > 0 {
		at, tok := t.nextAdded(text, special)
		if at < 0 {
			ids = t.encodePlain(text, ids)
			break
		}
		ids = t.encodePlain(text[:at], ids)
		ids = append(ids, tok.id)
		text = text[at+len(tok.content):]
	}
	return ids
}

func (t *Tokenizer) nextAdded(text string, special bool) (int, added) {
	best, bestTok := -1, added{}
	for _, a := range t.added {
		if a.special && !special {
			continue
		}
		if i := strings.Index(text, a.content); i >= 0 && (best < 0 || i < best) {
			best, bestTok = i, a
		}
	}
	return best, bestTok
}

func (t *Tokenizer) encodePlain(text string, ids []int32) []int32 {
	if t.metaspace {
		for _, piece := range t.split(strings.ReplaceAll(text, " ", "▁")) {
			ids = append(ids, t.mergeChars(piece)...)
		}
		return ids
	}
	for _, piece := range t.split(text) {
		ids = append(ids, t.bpe(byteLevel(piece))...)
	}
	return ids
}

// bpe merges the characters of one byte-level word by rank, lowest first.
func (t *Tokenizer) bpe(word string) []int32 {
	if id, ok := t.vocab[word]; ok {
		return []int32{id}
	}
	t.mu.Lock()
	if ids, ok := t.cache[word]; ok {
		t.mu.Unlock()
		return ids
	}
	t.mu.Unlock()

	// A word is most often a few characters, but a run of text with no space
	// or mark in it — long Chinese, base64 — is one word, of tens of
	// thousands: merged in a heap (merge), not by scanning after each merge.
	syms := make([]symbol, 0, len(word))
	for _, r := range word {
		syms = append(syms, symbol{s: string(r)})
	}
	t.merge(syms)
	ids := make([]int32, 0, len(syms))
	for i := 0; i >= 0 && i < len(syms); i = syms[i].next {
		if id, ok := t.vocab[syms[i].s]; ok {
			ids = append(ids, id)
		}
	}
	t.mu.Lock()
	if len(t.cache) < 1<<16 {
		t.cache[word] = ids
	}
	t.mu.Unlock()
	return ids
}

// Decode turns ids back into text; bytes that do not form valid UTF-8 come
// out as U+FFFD, as the tokenizers library's "replace".
func (t *Tokenizer) Decode(ids []int32) string {
	return strings.ToValidUTF8(string(t.Bytes(ids)), "�")
}

// Bytes is the raw bytes ids stand for, which may end inside a UTF-8
// character — what streaming needs, holding the tail back until the
// character completes.
func (t *Tokenizer) Bytes(ids []int32) []byte {
	var b []byte
	for _, id := range ids {
		if id < 0 || int(id) >= len(t.pieces) {
			continue
		}
		p := t.pieces[id]
		if t.isAdded(id) {
			b = append(b, p...)
			continue
		}
		if t.metaspace {
			if c, ok := byteToken(p); ok {
				b = append(b, c)
			} else {
				b = append(b, strings.ReplaceAll(p, "▁", " ")...)
			}
			continue
		}
		for _, r := range p {
			if c, ok := unicodeToByte[r]; ok {
				b = append(b, c)
			}
		}
	}
	return b
}

func (t *Tokenizer) isAdded(id int32) bool {
	for _, a := range t.added {
		if a.id == id {
			return true
		}
	}
	return false
}

// ID is the id of a vocabulary or added token, and whether there is one.
func (t *Tokenizer) ID(piece string) (int32, bool) {
	for _, a := range t.added {
		if a.content == piece {
			return a.id, true
		}
	}
	id, ok := t.vocab[piece]
	return id, ok
}

// IsSpecial reports whether id is a special added token.
func (t *Tokenizer) IsSpecial(id int32) bool { return t.special[id] }

// VocabSize is one more than the largest id.
func (t *Tokenizer) VocabSize() int { return len(t.pieces) }

// GPT-2's byte-to-character table: printable bytes stand for themselves, the
// rest shift up past U+0100, so every byte is a visible character.
var byteToUnicode, unicodeToByte = func() ([256]rune, map[rune]byte) {
	var bt [256]rune
	ub := map[rune]byte{}
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			bt[b] = rune(b)
		} else {
			bt[b] = rune(256 + n)
			n++
		}
		ub[bt[b]] = byte(b)
	}
	return bt, ub
}()

func byteLevel(s string) string {
	var sb strings.Builder
	for i := 0; i < len(s); i++ {
		sb.WriteRune(byteToUnicode[s[i]])
	}
	return sb.String()
}

func isLM(r rune, marks bool) bool { return unicode.IsLetter(r) || (marks && unicode.IsMark(r)) }
func isN(r rune) bool              { return unicode.IsNumber(r) }
func isSpace(r rune) bool          { return unicode.IsSpace(r) }
func isNL(r rune) bool             { return r == '\r' || r == '\n' }

// scan splits s the way the GPT-style pattern does, trying its alternatives
// in order at each position (leftmost-first, as Oniguruma does). marks is the
// Qwen3.5 variant, which counts \p{M} with letters.
func scan(s string, marks bool) []string {
	var out []string
	runes := []rune(s)
	n := len(runes)
	i := 0
	for i < n {
		j := match(runes, i, marks)
		if j <= i {
			j = i + 1 // unreachable with these patterns: \s+ or a later class takes any rune
		}
		out = append(out, string(runes[i:j]))
		i = j
	}
	return out
}

func match(r []rune, i int, marks bool) int {
	n := len(r)
	c := r[i]

	// (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if c == '\'' && i+1 < n {
		a := unicode.ToLower(r[i+1])
		if a == 's' || a == 't' || a == 'm' || a == 'd' {
			return i + 2
		}
		if i+2 < n {
			b := unicode.ToLower(r[i+2])
			if (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l') {
				return i + 3
			}
		}
	}

	// [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+   (Qwen2: \p{L}+)
	letters := func(k int) int {
		for k < n && isLM(r[k], marks) {
			k++
		}
		return k
	}
	if !isNL(c) && !unicode.IsLetter(c) && !isN(c) && i+1 < n && isLM(r[i+1], marks) {
		return letters(i + 1)
	}
	if isLM(c, marks) {
		return letters(i)
	}

	// \p{N}
	if isN(c) {
		return i + 1
	}

	// ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
	other := func(x rune) bool { return !isSpace(x) && !isLM(x, marks) && !isN(x) }
	k := i
	if c == ' ' && i+1 < n && other(r[i+1]) {
		k = i + 1
	}
	if other(r[k]) {
		for k < n && other(r[k]) {
			k++
		}
		for k < n && isNL(r[k]) {
			k++
		}
		return k
	}

	// The rest start with whitespace.
	end := i
	for end < n && isSpace(r[end]) {
		end++
	}
	if end == i {
		return i + 1
	}
	// \s*[\r\n]+ : up to the last newline in the run.
	for k := end - 1; k >= i; k-- {
		if isNL(r[k]) {
			return k + 1
		}
	}
	// \s+(?!\S) : the run, less its last space when text follows.
	if end == n {
		return end
	}
	if end-1 > i {
		return end - 1
	}
	// \s+
	return end
}
