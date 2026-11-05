package tokenizer

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
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
