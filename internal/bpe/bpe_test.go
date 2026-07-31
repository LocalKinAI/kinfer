package bpe

import "testing"

func TestDecode(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		// The exact symptom seen in the Phase 0 probe output.
		{"leading space", "ĠMerkle", " Merkle"},
		{"space mid-sequence", "Ġtree", " tree"},
		{"newline", "Ċ", "\n"},
		{"double newline", "ĊĊ", "\n\n"},
		{"tab", "ĉ", "\t"},

		// Plain ASCII must survive untouched.
		{"plain word", "Merkle", "Merkle"},
		{"punctuation", "data.", "data."},
		{"digits", "2026", "2026"},
		{"empty", "", ""},

		// A full sentence as the model actually emits it.
		{
			"real fragment",
			"ĠAĠMerkleĠtreeĠisĠaĠdataĠstructure",
			" A Merkle tree is a data structure",
		},

		// SPM/UTF-8 vocabularies store literal text — decoding must not corrupt it.
		{"cjk passthrough", "中文", "中文"},
		{"emoji passthrough", "🙂", "🙂"},
	}

	for _, c := range cases {
		if got := Decode(c.in); got != c.want {
			t.Errorf("%s: Decode(%q) = %q, want %q", c.name, c.in, got, c.want)
		}
	}
}

// A multi-byte character is frequently split across several tokens. Decoding the
// concatenation must reassemble it; decoding piece-by-piece would not.
func TestDecodeAll_MultiBytePieces(t *testing.T) {
	// "中" is UTF-8 e4 b8 ad. In the byte-level alphabet those three bytes are
	// each mapped into the U+0100.. range, and a tokenizer may emit them as
	// separate tokens.
	var pieces []string
	for _, b := range []byte("中文") {
		pieces = append(pieces, string(byteToRune[b]))
	}

	if got := DecodeAll(pieces); got != "中文" {
		t.Errorf("DecodeAll(split bytes) = %q, want %q", got, "中文")
	}
}

// The byte table must be a true bijection, or decoding is lossy.
func TestByteTableIsBijective(t *testing.T) {
	if len(runeToByte) != 256 {
		t.Fatalf("reverse table has %d entries, want 256", len(runeToByte))
	}
	for b := 0; b < 256; b++ {
		if got := runeToByte[byteToRune[b]]; got != byte(b) {
			t.Errorf("byte %d round-trips to %d", b, got)
		}
	}
}
