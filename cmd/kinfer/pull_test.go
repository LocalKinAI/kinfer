package main

import "testing"

// A repo's file name becomes a path under the model's folder only if it stays
// there: folders are kept, a way out — or a backslash, Windows' separator —
// is refused.
func TestRepoPathStaysInTheFolder(t *testing.T) {
	for name, ok := range map[string]bool{
		"model.safetensors":                   true,
		"speech_tokenizer/model.safetensors":  true,
		"../model.safetensors":                false,
		"speech_tokenizer/../../x.json":       false,
		"/etc/passwd":                         false,
		`speech_tokenizer/..\..\..\evil.json`: false,
		`a\b.json`:                            false,
	} {
		if _, got := repoPath(name); got != ok {
			t.Errorf("repoPath(%q) ok = %v, want %v", name, got, ok)
		}
	}
}
