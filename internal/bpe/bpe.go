// Package bpe decodes the byte-level BPE representation that llama.cpp's
// vocab exposes as raw token text.
//
// Why this exists: gollama.cpp's Token_to_piece is misnamed — it calls
// llama_vocab_get_text (the raw vocab entry) rather than llama_token_to_piece
// (which performs byte-level decoding). So a generated token comes back as
// "ĠMerkle" instead of " Merkle". Every GPT-2-lineage tokenizer (Qwen, Llama 3,
// Mistral, …) uses this encoding, so the fix is a single reverse table.
//
// This is a stopgap. The correct long-term fix is to bind llama_token_to_piece
// directly, which also handles SPM/WPM vocabularies where this table does not
// apply. See PHASE0 notes.
package bpe

import "strings"

// byteToRune is GPT-2's bytes_to_unicode(): a bijection from all 256 byte
// values onto printable runes, so that a tokenizer's vocabulary never contains
// control characters or raw whitespace.
//
// Printable ASCII and two Latin-1 ranges map to themselves; every other byte is
// pushed into the U+0100.. private range in order. That is why a space (0x20)
// shows up as Ġ (U+0120) and a newline (0x0A) as Ċ (U+010A).
var byteToRune [256]rune

// runeToByte is the reverse table used for decoding.
var runeToByte map[rune]byte

func init() {
	// The three ranges GPT-2 leaves untouched.
	direct := func(r rune) bool {
		return (r >= '!' && r <= '~') ||
			(r >= 0xA1 && r <= 0xAC) ||
			(r >= 0xAE && r <= 0xFF)
	}

	n := 0
	for b := 0; b < 256; b++ {
		if direct(rune(b)) {
			byteToRune[b] = rune(b)
			continue
		}
		byteToRune[b] = rune(256 + n)
		n++
	}

	runeToByte = make(map[rune]byte, 256)
	for b, r := range byteToRune {
		runeToByte[r] = byte(b)
	}
}

// Decode converts one raw vocab piece back into real text.
//
// Runes that are not part of the byte-level alphabet are passed through
// unchanged — that covers vocabularies which store literal UTF-8 (SPM models),
// so calling Decode on them is a no-op rather than corruption.
func Decode(piece string) string {
	if piece == "" {
		return ""
	}

	var out []byte
	var passthrough []rune

	flush := func() {
		if len(passthrough) > 0 {
			out = append(out, []byte(string(passthrough))...)
			passthrough = passthrough[:0]
		}
	}

	for _, r := range piece {
		if b, ok := runeToByte[r]; ok {
			flush()
			out = append(out, b)
			continue
		}
		passthrough = append(passthrough, r)
	}
	flush()

	return string(out)
}

// DecodeAll joins and decodes a sequence of pieces. Decoding the concatenation
// (rather than each piece separately) matters for multi-byte characters: a
// single CJK rune is often split across two or three tokens, and decoding them
// in isolation would yield invalid UTF-8 fragments.
func DecodeAll(pieces []string) string {
	return Decode(strings.Join(pieces, ""))
}
