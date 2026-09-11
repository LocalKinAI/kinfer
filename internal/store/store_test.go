package store

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newStore(t *testing.T, files ...string) *Store {
	t.Helper()
	s, err := OpenAt(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for i, f := range files {
		path := filepath.Join(s.Root(), f)
		if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
		// Distinct mtimes so List's ordering is deterministic.
		mt := time.Now().Add(time.Duration(i) * time.Minute)
		if err := os.Chtimes(path, mt, mt); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

func TestList_NewestFirstAndGgufOnly(t *testing.T) {
	s := newStore(t, "old.gguf", "new.gguf")
	if err := os.WriteFile(filepath.Join(s.Root(), "notes.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	models, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 {
		t.Fatalf("List returned %d models, want 2 (non-gguf must be ignored)", len(models))
	}
	if models[0].Name != "new" {
		t.Errorf("newest first: got %q", models[0].Name)
	}
	if models[0].Size != 4 {
		t.Errorf("size = %d, want 4", models[0].Size)
	}
}

func TestList_Empty(t *testing.T) {
	models, err := newStore(t).List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 0 {
		t.Errorf("empty store returned %d models", len(models))
	}
}

func TestResolve(t *testing.T) {
	s := newStore(t, "qwen2.5-0.5b-instruct-q4_k_m.gguf", "llama-3.2-3b.gguf")

	cases := []struct{ in, wantFile string }{
		{"qwen2.5-0.5b-instruct-q4_k_m", "qwen2.5-0.5b-instruct-q4_k_m.gguf"},      // exact name
		{"qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf"}, // filename
		{"QWEN2.5-0.5B-INSTRUCT-Q4_K_M", "qwen2.5-0.5b-instruct-q4_k_m.gguf"},      // case-insensitive
		{"qwen", "qwen2.5-0.5b-instruct-q4_k_m.gguf"},                              // unique prefix
		{"llama", "llama-3.2-3b.gguf"},
	}
	for _, c := range cases {
		got, err := s.Resolve(c.in)
		if err != nil {
			t.Errorf("Resolve(%q): %v", c.in, err)
			continue
		}
		if filepath.Base(got) != c.wantFile {
			t.Errorf("Resolve(%q) = %q, want %q", c.in, filepath.Base(got), c.wantFile)
		}
	}
}

// Loading the wrong 7B model silently is worse than an error.
func TestResolve_AmbiguousPrefixFails(t *testing.T) {
	s := newStore(t, "qwen2.5-0.5b.gguf", "qwen2.5-7b.gguf")

	_, err := s.Resolve("qwen")
	if err == nil {
		t.Fatal("ambiguous prefix should fail")
	}
	if !strings.Contains(err.Error(), "ambiguous") {
		t.Errorf("error should say what went wrong, got: %v", err)
	}
}

func TestResolve_Missing(t *testing.T) {
	s := newStore(t, "qwen.gguf")
	if _, err := s.Resolve("mistral"); err == nil {
		t.Error("Resolve on a missing model should fail")
	}
	if _, err := s.Resolve(""); err == nil {
		t.Error("Resolve(\"\") should fail")
	}
}

func TestResolve_EmptyStoreSuggestsPull(t *testing.T) {
	_, err := newStore(t).Resolve("anything")
	if err == nil || !strings.Contains(err.Error(), "pull") {
		t.Errorf("empty store should point at `pull`, got: %v", err)
	}
}

func TestRemove(t *testing.T) {
	s := newStore(t, "doomed.gguf", "keeper.gguf")

	if err := s.Remove("doomed"); err != nil {
		t.Fatal(err)
	}
	models, _ := s.List()
	if len(models) != 1 || models[0].Name != "keeper" {
		t.Errorf("after Remove, models = %v", names(models))
	}
	if err := s.Remove("doomed"); err == nil {
		t.Error("removing a gone model should fail")
	}
}

// `kinfer rm` must not become a general-purpose file deleter.
func TestRemove_RefusesOutsideStore(t *testing.T) {
	s := newStore(t)

	outside := filepath.Join(t.TempDir(), "precious.gguf")
	if err := os.WriteFile(outside, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := s.Remove(outside); err == nil {
		t.Error("Remove outside the store should be refused")
	}
	if _, err := os.Stat(outside); err != nil {
		t.Error("the outside file was deleted")
	}
}

func TestHumanSize(t *testing.T) {
	cases := map[int64]string{
		512:                    "512 B",
		1024:                   "1.0 KB",
		1024 * 1024:            "1.0 MB",
		5 * 1024 * 1024 * 1024: "5.0 GB",
	}
	for in, want := range cases {
		if got := HumanSize(in); got != want {
			t.Errorf("HumanSize(%d) = %q, want %q", in, got, want)
		}
	}
}
