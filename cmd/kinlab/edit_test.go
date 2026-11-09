package main

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/mlxmodel"
	"github.com/LocalKinAI/kinfer/internal/safetensors"
	"github.com/LocalKinAI/kinfer/internal/store"
)

func bf16(v ...float32) []byte { return fromFloats(v, "BF16") }

// tinyModel writes a four-layer Qwen3.5 small enough to read by eye: layer 3
// is full attention (two heads of two), the others DeltaNet (two value heads
// of two), and the MTP head has a layer 0 of its own.
func tinyModel(t *testing.T, st *store.Store, name string, down []float32) {
	t.Helper()
	dir := filepath.Join(st.Root(), name)
	os.MkdirAll(dir, 0o755)
	cfg := `{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","hidden_size":2,"intermediate_size":3,
		"num_hidden_layers":4,"full_attention_interval":4,"num_attention_heads":2,"head_dim":2,
		"linear_num_value_heads":2,"linear_value_head_dim":2}}`
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644)
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{}`), 0o644)
	p := "model.language_model.layers."
	err := safetensors.Write(filepath.Join(dir, "model.safetensors"), []safetensors.Tensor{
		{Name: p + "0.mlp.down_proj.weight", Dtype: "BF16", Shape: []int{2, 3}, Data: bf16(down...)},
		{Name: "mtp.layers.0.mlp.down_proj.weight", Dtype: "BF16", Shape: []int{2, 3}, Data: bf16(1, 2, 3, 4, 5, 6)},
		{Name: p + "0.linear_attn.out_proj.weight", Dtype: "BF16", Shape: []int{2, 4}, Data: bf16(1, 2, 3, 4, 5, 6, 7, 8)},
		{Name: p + "3.self_attn.o_proj.weight", Dtype: "BF16", Shape: []int{2, 4}, Data: bf16(1, 2, 3, 4, 5, 6, 7, 8)},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
}

func readEdit(t *testing.T, st *store.Store, name string) map[string][]float32 {
	t.Helper()
	f, err := safetensors.Open(filepath.Join(st.Root(), name, mlxmodel.EditFile))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	out := map[string][]float32{}
	for _, n := range f.Names() {
		b, _ := f.Read(n)
		out[n], _ = floats(b, "BF16")
	}
	return out
}

func TestEditOperations(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	st, _ := store.Open()
	tinyModel(t, st, "base", []float32{1, 2, 3, 4, 5, 6})
	tinyModel(t, st, "other", []float32{3, 2, 1, 8, 5, 2})
	p := "model.language_model.layers."
	for _, c := range []struct {
		name string
		args []string
		want map[string][]float32
	}{
		{"neurons", []string{"-zero-neuron", "0:1"},
			map[string][]float32{p + "0.mlp.down_proj.weight": {1, 0, 3, 4, 0, 6}}},
		{"attn-head", []string{"-zero-head", "3:1"},
			map[string][]float32{p + "3.self_attn.o_proj.weight": {1, 2, 0, 0, 5, 6, 0, 0}}},
		{"delta-head", []string{"-zero-head", "0:0"},
			map[string][]float32{p + "0.linear_attn.out_proj.weight": {0, 0, 3, 4, 0, 0, 7, 8}}},
		{"scaled", []string{"-scale", `layers\.0\.mlp\.down_proj=2`},
			map[string][]float32{p + "0.mlp.down_proj.weight": {2, 4, 6, 8, 10, 12}}},
		{"halfway", []string{"-mix", "other=0.5"},
			map[string][]float32{p + "0.mlp.down_proj.weight": {2, 2, 2, 6, 5, 4}}},
		{"mixed-then-zeroed", []string{"-mix", "other=1", "-zero-neuron", "0:0-1"},
			map[string][]float32{p + "0.mlp.down_proj.weight": {0, 0, 1, 0, 0, 2}}},
	} {
		if err := cmdEdit(append(c.args, "base", c.name)); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		got := readEdit(t, st, c.name)
		if len(got) != len(c.want) {
			t.Errorf("%s changed %d tensors, want %d: %v", c.name, len(got), len(c.want), got)
		}
		for n, w := range c.want {
			if !slices.Equal(got[n], w) {
				t.Errorf("%s: %s = %v, want %v", c.name, n, got[n], w)
			}
		}
	}

	// -only takes the other model's tensors that match, and leaves the rest.
	tinyModel(t, st, "two", []float32{3, 2, 1, 8, 5, 2})
	two := filepath.Join(st.Root(), "two", "model.safetensors")
	tf, _ := safetensors.Open(two)
	var all []safetensors.Tensor
	for _, n := range tf.Names() {
		b, _ := tf.Read(n)
		info := tf.Tensors[n]
		if n == p+"3.self_attn.o_proj.weight" {
			b = bf16(8, 7, 6, 5, 4, 3, 2, 1)
		}
		all = append(all, safetensors.Tensor{Name: n, Dtype: info.Dtype, Shape: info.Shape, Data: b})
	}
	tf.Close()
	safetensors.Write(two, all, nil)
	if err := cmdEdit([]string{"-from", "two", "-only", `layers\.3\.`, "base", "only3"}); err != nil {
		t.Fatal(err)
	}
	if got := readEdit(t, st, "only3"); len(got) != 1 || !slices.Equal(got[p+"3.self_attn.o_proj.weight"], []float32{8, 7, 6, 5, 4, 3, 2, 1}) {
		t.Errorf("-only took %v", got)
	}

	// An edit read as the other model is its base with its edit in place:
	// halfway to "halfway" is a quarter of the way to other.
	if err := cmdEdit([]string{"-mix", "halfway=0.5", "base", "quarter"}); err != nil {
		t.Fatal(err)
	}
	if got := readEdit(t, st, "quarter")[p+"0.mlp.down_proj.weight"]; !slices.Equal(got, []float32{1.5, 2, 2.5, 5, 5, 5}) {
		t.Errorf("halfway to an edit took %v", got)
	}

	// All the way to the other model is the other model, byte for byte.
	if err := cmdEdit([]string{"-from", "other", "base", "from"}); err != nil {
		t.Fatal(err)
	}
	if err := cmdEdit([]string{"-mix", "other=1", "base", "mix1"}); err != nil {
		t.Fatal(err)
	}
	a, _ := os.ReadFile(filepath.Join(st.Root(), "from", mlxmodel.EditFile))
	b, _ := os.ReadFile(filepath.Join(st.Root(), "mix1", mlxmodel.EditFile))
	if !bytes.Equal(a, b) {
		t.Error("-mix other=1 is not -from other")
	}

	for _, bad := range [][]string{
		{"-zero-neuron", "0:3"},                     // there are three neurons
		{"-zero-head", "3:2"},                       // and two heads
		{"-zero-neuron", "9:0"},                     // and four layers
		{"-scale", "nothing-here=2"},                // a pattern that matches nothing
		{"-scale", `down_proj=1`},                   // leaves the model as it was
		{"-from", "other", "-scale", `down_proj=2`}, // two kinds of edit at once
	} {
		if err := cmdEdit(append(bad, "base", "bad")); err == nil {
			t.Errorf("kinfer edit %v made a model", bad)
			os.RemoveAll(filepath.Join(st.Root(), "bad"))
		}
	}
}

// BF16 rounds to nearest, ties to even, and what it holds comes back exactly.
func TestBF16(t *testing.T) {
	v := []float32{1, -2.5, 3.140625, float32(math.Ldexp(1, -70)), 65280}
	got, _ := floats(bf16(v...), "BF16")
	if !slices.Equal(got, v) {
		t.Errorf("round trip %v, want %v", got, v)
	}
	// 1 + 2^-8 is halfway between 1 and the next BF16 up, 1 + 2^-7: even is 1.
	// 1 + 3·2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: even is 1 + 2^-6.
	r, _ := floats(bf16(1+1.0/256, 1+3.0/256), "BF16")
	if r[0] != 1 || r[1] != 1+1.0/64 {
		t.Errorf("ties went to %v, want [1 %v]", r, 1+1.0/64)
	}
}
