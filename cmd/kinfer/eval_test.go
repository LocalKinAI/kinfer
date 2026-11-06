package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/safetensors"
	"github.com/LocalKinAI/kinfer/internal/store"
)

func TestLoadSetReadsArraysAndLines(t *testing.T) {
	dir := t.TempDir()
	arr := filepath.Join(dir, "a.json")
	os.WriteFile(arr, []byte(`[{"uid":"x/1","problem":"1+1?","answer":"2","level":1},{"problem":"2+2?","answer":4}]`), 0o644)
	lines := filepath.Join(dir, "b.jsonl")
	os.WriteFile(lines, []byte("{\"question\":\"3+3?\",\"answer\":\"6\"}\n\n{\"prompt\":\"4+4?\",\"answer\":\"8\"}\n"), 0o644)
	a, err := loadSet(arr)
	if err != nil || len(a) != 2 || a[0].id != "x/1" || a[1].id != "1" || a[1].answer != "4" {
		t.Fatalf("array: %+v, %v", a, err)
	}
	b, err := loadSet(lines)
	if err != nil || len(b) != 2 || b[0].text != "3+3?" || b[1].text != "4+4?" {
		t.Fatalf("lines: %+v, %v", b, err)
	}
	bad := filepath.Join(dir, "c.jsonl")
	os.WriteFile(bad, []byte(`{"problem":"no answer"}`), 0o644)
	if _, err := loadSet(bad); err == nil {
		t.Error("a problem without an answer was accepted")
	}
}

func TestAnswerPartGradesWhatFollowsThinking(t *testing.T) {
	for in, want := range map[string]string{
		`<think>maybe \boxed{1}</think>so \boxed{2}`: `so \boxed{2}`,
		`<think>cut off \boxed{1}`:                   "",
		`plain \boxed{3}`:                            `plain \boxed{3}`,
	} {
		if got := answerPart(in); got != want {
			t.Errorf("answerPart(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestMcNemar(t *testing.T) {
	for _, c := range []struct {
		b, c int
		p    float64
	}{{10, 2, 0.0386}, {5, 5, 1}, {0, 0, 1}, {20, 0, 1.9e-6}} {
		if got := mcnemar(c.b, c.c); math.Abs(got-c.p) > c.p*0.02+1e-9 {
			t.Errorf("mcnemar(%d, %d) = %.4g, want %.4g", c.b, c.c, got, c.p)
		}
	}
}

// An edit made -from another copy holds exactly the tensors that differ,
// and links the rest to the base.
func TestEditFromKeepsOnlyWhatDiffers(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	st, err := store.Open()
	if err != nil {
		t.Fatal(err)
	}
	model := func(name string, w1 byte) {
		dir := filepath.Join(st.Root(), name)
		os.MkdirAll(dir, 0o755)
		os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen3_5"}`), 0o644)
		os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{}`), 0o644)
		err := safetensors.Write(filepath.Join(dir, "model.safetensors"), []safetensors.Tensor{
			{Name: "a.weight", Dtype: "BF16", Shape: []int{2}, Data: []byte{1, 2, 3, 4}},
			{Name: "b.weight", Dtype: "BF16", Shape: []int{2}, Data: []byte{w1, 6, 7, 8}},
		}, nil)
		if err != nil {
			t.Fatal(err)
		}
	}
	model("base", 5)
	model("tuned", 9)
	if err := cmdEdit([]string{"-from", "tuned", "-note", "b from tuned", "base", "edited"}); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(st.Root(), "edited")
	if p, err := os.Readlink(filepath.Join(dir, "model.safetensors")); err != nil || p != filepath.Join(st.Root(), "base", "model.safetensors") {
		t.Errorf("model.safetensors links to %q, %v", p, err)
	}
	f, err := safetensors.Open(filepath.Join(dir, "edit.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if names := f.Names(); len(names) != 1 || names[0] != "b.weight" {
		t.Errorf("the edit holds %v, want only b.weight", names)
	}
	if b, _ := f.Read("b.weight"); len(b) != 4 || b[0] != 9 {
		t.Errorf("b.weight is %v, want tuned's", b)
	}
	var rec editRecord
	b, _ := os.ReadFile(filepath.Join(dir, "edit.json"))
	if json.Unmarshal(b, &rec); rec.Base != "base" || rec.Note != "b from tuned" || len(rec.Tensors) != 1 {
		t.Errorf("edit.json: %+v", rec)
	}
	models, _ := st.List()
	found := false
	for _, m := range models {
		if m.Name == "edited" && m.EditOf == "base" {
			found = true
		}
	}
	if !found {
		t.Errorf("the store does not list the edit as one of base: %+v", models)
	}
	if err := cmdEdit([]string{"-from", "tuned", "edited", "again"}); err == nil {
		t.Error("an edit of an edit was made")
	}
	if err := cmdEdit([]string{"-from", "base", "base", "same"}); err == nil {
		t.Error("an edit that changes nothing was made")
	}
}

// A subset is the same every time, whatever order the set lists its problems
// in, and shares itself among the levels while each has problems left.
func TestPickSubset(t *testing.T) {
	var probs []problem
	for i := 0; i < 50; i++ {
		lv := float64(1 + i%5)
		if i >= 45 {
			lv = 6 // a level of only five
		}
		probs = append(probs, problem{id: fmt.Sprintf("p%d", i), level: lv})
	}
	a := pickSubset(probs, 30)
	rev := append([]problem(nil), probs...)
	slices.Reverse(rev)
	b := pickSubset(rev, 30)
	ids := func(ps []problem) []string {
		var out []string
		for _, p := range ps {
			out = append(out, p.id)
		}
		sort.Strings(out)
		return out
	}
	if !slices.Equal(ids(a), ids(b)) {
		t.Errorf("the subset depends on the set's order: %v vs %v", ids(a), ids(b))
	}
	per := map[any]int{}
	for _, p := range a {
		per[p.level]++
	}
	// Six levels share 30: five each.
	for lv, k := range per {
		if k != 5 {
			t.Errorf("level %v has %d of the subset, want 5: %v", lv, k, per)
		}
	}
	if got := pickSubset(probs, 100); len(got) != len(probs) {
		t.Errorf("a subset larger than the set is %d problems", len(got))
	}
}
