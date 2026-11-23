package tokenizer

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

// TestAgainstReference compares Encode and Decode with another tokenizer's
// results for the same tokenizer.json: KINFER_TOK_FILE is the file and
// KINFER_TOK_CASES a JSON list of {"text": …, "ids": […]} made with it
// (Hugging Face's tokenizers, add_special_tokens off, special tokens parsed).
func TestAgainstReference(t *testing.T) {
	file, cases := os.Getenv("KINFER_TOK_FILE"), os.Getenv("KINFER_TOK_CASES")
	if file == "" || cases == "" {
		t.Skip("KINFER_TOK_FILE and KINFER_TOK_CASES not both set")
	}
	tok, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(cases)
	if err != nil {
		t.Fatal(err)
	}
	var cs []struct {
		Text string  `json:"text"`
		IDs  []int32 `json:"ids"`
	}
	if err := json.Unmarshal(raw, &cs); err != nil {
		t.Fatal(err)
	}
	bad := 0
	for _, c := range cs {
		got := tok.Encode(c.Text, true)
		if !slices.Equal(got, c.IDs) {
			bad++
			if bad <= 8 {
				t.Errorf("%q:\n got %v\nwant %v", c.Text, got, c.IDs)
			}
			continue
		}
		if back := tok.Decode(c.IDs); back != c.Text {
			bad++
			if bad <= 8 {
				t.Errorf("%q decodes to %q", c.Text, back)
			}
		}
	}
	t.Logf("%d of %d texts encode and decode as the reference", len(cs)-bad, len(cs))
}
