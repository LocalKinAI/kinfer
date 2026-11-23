package main

import (
	"testing"

	"github.com/LocalKinAI/kinfer/internal/hub"
	"github.com/LocalKinAI/kinfer/internal/mlxmodel"
)

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

// Inkling 2 is pulled as its publisher keeps it on Hugging Face, but for the
// file of its draft head, which kinfer does not read.
func TestPullInklingLeavesTheDraftHead(t *testing.T) {
	all := []hub.File{
		{Name: "README.md", Size: 54898}, {Name: "chat_template.jinja", Size: 6868}, {Name: "config.json", Size: 2722},
		{Name: "hf_quant_config.json", Size: 67924}, {Name: "model-00001-of-00001.safetensors", Size: 19526019168},
		{Name: "model.safetensors.index.json", Size: 109300}, {Name: "mtp.safetensors", Size: 635801416},
		{Name: "processor_config.json", Size: 461}, {Name: "tokenizer.json", Size: 26178678}, {Name: "tokenizer_config.json", Size: 13235},
	}
	set, ok := hub.MLXFiles(all)
	if !ok {
		t.Fatal("the publisher's files are not taken for a model on MLX")
	}
	if !servedOnMLX(mlxmodel.InklingModelType) {
		t.Fatal("inkling2_loop is not served on MLX")
	}
	got := mlxSet(mlxmodel.InklingModelType, set)
	names := map[string]bool{}
	var size int64
	for _, f := range got {
		names[f.Name] = true
		size += f.Size
	}
	if names["mtp.safetensors"] || !names["model-00001-of-00001.safetensors"] || !names["tokenizer.json"] || !names["processor_config.json"] {
		t.Errorf("pulled: %v", names)
	}
	if got[len(got)-1].Name != "config.json" {
		t.Errorf("config.json, which makes the folder a model, is not last: %v", got[len(got)-1].Name)
	}
	if size > 19_600_000_000 {
		t.Errorf("%d bytes pulled", size)
	}
	if kept := mlxSet("qwen3_5", set); len(kept) != len(set) {
		t.Errorf("another model's pull leaves %d of %d files", len(set)-len(kept), len(set))
	}
}
