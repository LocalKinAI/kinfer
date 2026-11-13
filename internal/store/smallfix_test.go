package store

import (
	"os"
	"path/filepath"
	"testing"
)

// A tensor whose name gives a layer of a billion is not counted: the slices
// grew to it, gigabytes for a header of a few hundred bytes.
func TestALayerNumberTooLargeIsNotCounted(t *testing.T) {
	const mib = 1 << 20
	raw := ggufFile(map[string]any{"general.architecture": "qwen2", "tokenizer.ggml.tokens": []string{"a"}},
		[]tensorEntry{{"blk.0.attn_q.weight", 0}, {"blk.999999999.attn_q.weight", 1 * mib}})
	p := filepath.Join(t.TempDir(), "x.gguf")
	pad := make([]byte, (len(raw)+31)/32*32-len(raw)+2*mib)
	if err := os.WriteFile(p, append(raw, pad...), 0o644); err != nil {
		t.Fatal(err)
	}
	sh, err := ReadShape(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(sh.LayerBytes) != 1 || len(sh.ExpertBytes) != 1 {
		t.Errorf("LayerBytes has %d entries, want the one real layer", len(sh.LayerBytes))
	}
}

// A file in Ollama's manifests folder that is not three deep is not a model:
// one two deep made an index of -1, a panic on every lookup.
func TestAManifestTooShallowIsNotAModel(t *testing.T) {
	root := t.TempDir()
	manifests := filepath.Join(root, "manifests")
	path := filepath.Join(manifests, "registry.ollama.ai", "stray.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"layers":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if m, ok := readOllamaManifest(root, manifests, path); ok {
		t.Errorf("a stray file was read as %+v", m)
	}
}

// Removing a model takes the unfinished pull of its name, and not one whose
// name differs only in case: that was another model's download.
func TestRemoveLeavesAnotherCasesUnfinishedPull(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "Foo-Q4_K_M.gguf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	writePart(t, dir, "foo-q4_k_m.gguf", 1000, 300) // another model's pull, its name but for case
	s, _ := OpenAt(dir)
	if err := s.Remove("Foo-Q4_K_M"); err != nil {
		t.Fatal(err)
	}
	left, _ := s.Unfinished()
	if len(left) != 1 || left[0].Name != "foo-q4_k_m" {
		t.Errorf("unfinished after removing Foo-Q4_K_M: %+v, want foo-q4_k_m's pull left", left)
	}
	if _, err := os.Stat(filepath.Join(dir, "foo-q4_k_m.gguf.part")); err != nil {
		t.Errorf("its file: %v", err)
	}
}
