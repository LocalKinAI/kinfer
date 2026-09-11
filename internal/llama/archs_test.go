package llama

import "testing"

func TestHasCStringMatchesWholeStringsOnly(t *testing.T) {
	data := []byte("\x00llama\x00qwen2moe\x00gemma3\x00")

	for _, s := range []string{"llama", "qwen2moe", "gemma3"} {
		if !hasCString(data, s) {
			t.Errorf("hasCString(%q) = false, want true", s)
		}
	}
	// A substring of a name is not that name. "qwen2" inside "qwen2moe" must
	// not read as support for qwen2, or a model would be promised a runtime
	// that cannot load it.
	for _, s := range []string{"qwen2", "lama", "gemma", ""} {
		if hasCString(data, s) {
			t.Errorf("hasCString(%q) = true, want false", s)
		}
	}
}

// The check has to fail closed. A caller that cannot be told the truth must be
// told nothing rather than a guess — and in particular must never be told "not
// supported", which would stop someone trying a model that works.
func TestSupportsReportsWhenTheAnswerIsUnknown(t *testing.T) {
	libOnce.Do(func() {})
	libBytes = nil

	supported, known := Supports("/nonexistent", "llama")
	if supported || known {
		t.Errorf("Supports with no library = (%v, %v), want (false, false)", supported, known)
	}
}
