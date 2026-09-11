package store

import (
	"fmt"
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

// Anything past roughly 50 GB arrives from Hugging Face in pieces, so a store
// that cannot recognise them cannot hold a large model at all: three files look
// like three models, each reporting a fraction of the size, and asking for the
// model by name is ambiguous between its own pieces.

func writeShards(t *testing.T, dir, base string, n int, each int) {
	t.Helper()
	for i := 1; i <= n; i++ {
		name := fmt.Sprintf("%s-%05d-of-%05d.gguf", base, i, n)
		if err := os.WriteFile(filepath.Join(dir, name), make([]byte, each), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestListCollapsesShardsIntoOneModel(t *testing.T) {
	dir := t.TempDir()
	writeShards(t, dir, "Qwen3.8-Flash-Next-UD-Q2_K_XL", 3, 1000)
	if err := os.WriteFile(filepath.Join(dir, "plain.gguf"), make([]byte, 7), 0o644); err != nil {
		t.Fatal(err)
	}

	s, err := OpenAt(dir)
	if err != nil {
		t.Fatal(err)
	}
	models, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 {
		t.Fatalf("List returned %d models, want 2 (one split, one plain): %+v", len(models), models)
	}

	var split *Model
	for i := range models {
		if models[i].Name == "Qwen3.8-Flash-Next-UD-Q2_K_XL" {
			split = &models[i]
		}
	}
	if split == nil {
		t.Fatalf("the split model is not listed under its base name: %+v", models)
	}
	if split.Size != 3000 {
		t.Errorf("size = %d, want the sum of all pieces (3000)", split.Size)
	}
	if !strings.HasSuffix(split.Path, "-00001-of-00003.gguf") {
		t.Errorf("path = %q, want the first piece — llama.cpp finds the rest from it", split.Path)
	}
}

func TestResolveFindsASplitModelByPrefix(t *testing.T) {
	dir := t.TempDir()
	writeShards(t, dir, "Qwen3.8-Flash-Next-UD-Q2_K_XL", 3, 10)
	s, _ := OpenAt(dir)

	got, err := s.Resolve("qwen3.8")
	if err != nil {
		t.Fatalf("Resolve by prefix failed: %v — pieces of one model must not be ambiguous with each other", err)
	}
	if !strings.HasSuffix(got, "-00001-of-00003.gguf") {
		t.Errorf("Resolve returned %q, want the first piece", got)
	}
}

func TestListIgnoresASetMissingItsFirstPiece(t *testing.T) {
	dir := t.TempDir()
	writeShards(t, dir, "Broken", 3, 10)
	if err := os.Remove(filepath.Join(dir, "Broken-00001-of-00003.gguf")); err != nil {
		t.Fatal(err)
	}
	s, _ := OpenAt(dir)
	models, _ := s.List()
	if len(models) != 0 {
		t.Errorf("a set with no first piece was listed as loadable: %+v", models)
	}
}

func TestRemoveDeletesEveryPiece(t *testing.T) {
	dir := t.TempDir()
	writeShards(t, dir, "Big", 3, 10)
	s, _ := OpenAt(dir)

	if err := s.Remove("Big"); err != nil {
		t.Fatal(err)
	}
	left, _ := os.ReadDir(dir)
	if len(left) != 0 {
		// Removing one piece would strand tens of gigabytes that nothing loads.
		t.Errorf("rm left %d pieces behind", len(left))
	}
}

// Linking a model into the store is a reasonable thing to do — it is how one
// file is shared with another runtime instead of copied. Reading the link's own
// size instead of the model's reports twenty gigabytes as a hundred bytes, and
// every size-based decision downstream is then made about nothing.
func TestListFollowsSymlinks(t *testing.T) {
	target := filepath.Join(t.TempDir(), "real-model.gguf")
	if err := os.WriteFile(target, make([]byte, 4096), 0o644); err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	if err := os.Symlink(target, filepath.Join(dir, "linked.gguf")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	s, _ := OpenAt(dir)
	models, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 {
		t.Fatalf("List returned %d models, want 1", len(models))
	}
	if models[0].Size != 4096 {
		t.Errorf("size = %d, want the target's 4096 — a link's own size is not the model's", models[0].Size)
	}
}

func TestListSkipsBrokenSymlinks(t *testing.T) {
	dir := t.TempDir()
	if err := os.Symlink(filepath.Join(dir, "gone.gguf"), filepath.Join(dir, "dangling.gguf")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	s, _ := OpenAt(dir)
	models, _ := s.List()
	if len(models) != 0 {
		t.Errorf("a link with nothing at the end of it was listed: %+v", models)
	}
}

// Ollama's store belongs to Ollama. kinfer reads models out of it so that
// someone who already has forty gigabytes on disk is not made to download them
// again — but it must never write there, and in particular `kinfer rm` must not
// be able to delete another program's data.
func TestRemoveRefusesToDeleteOllamasBlobs(t *testing.T) {
	dir := t.TempDir()
	ollama := filepath.Join(t.TempDir(), "blobs")
	if err := os.MkdirAll(ollama, 0o755); err != nil {
		t.Fatal(err)
	}
	blob := filepath.Join(ollama, "sha256-abc123")
	if err := os.WriteFile(blob, []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}

	s, _ := OpenAt(dir)
	err := s.Remove(blob)
	if err == nil {
		t.Fatal("kinfer rm deleted a file from Ollama's blob store")
	}
	if !strings.Contains(err.Error(), "ollama rm") {
		t.Errorf("error does not point at the right tool: %v", err)
	}
	if _, statErr := os.Stat(blob); statErr != nil {
		t.Error("the blob was removed despite the error")
	}
}

func TestAllPrefersKinfersOwnCopy(t *testing.T) {
	// A model present in both places is kinfer's: someone put it there on
	// purpose, and a borrowed one should not shadow it.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "shared.gguf"), make([]byte, 99), 0o644); err != nil {
		t.Fatal(err)
	}
	s, _ := OpenAt(dir)
	all, err := s.All()
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range all {
		if m.Name == "shared" && m.Source != FromKinfer {
			t.Errorf("shared model came from %q, want %q", m.Source, FromKinfer)
		}
	}
}

// OpenAt names one directory and means it. If it reached into Ollama's store as
// well, a caller asking about a path would get models from somewhere else, and
// every test here would depend on what the machine running it has installed.
func TestOpenAtDoesNotBorrow(t *testing.T) {
	s, err := OpenAt(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	all, err := s.All()
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 0 {
		t.Errorf("OpenAt on an empty directory returned %d models: %+v", len(all), all)
	}
}
