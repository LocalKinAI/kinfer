package hub

import "testing"

func TestParseRef(t *testing.T) {
	cases := []struct {
		in    string
		repo  string
		quant string
		file  string
	}{
		{"Qwen/Qwen2.5-0.5B-Instruct-GGUF", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "", ""},
		{"Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", "Qwen/Qwen2.5-7B-Instruct-GGUF", "Q4_K_M", ""},
		{"hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF:Q5_K_M", "Qwen/Qwen2.5-7B-Instruct-GGUF", "Q5_K_M", ""},
		{"huggingface.co/Qwen/X-GGUF", "Qwen/X-GGUF", "", ""},
		{
			"https://huggingface.co/Qwen/X-GGUF/blob/main/model-q4.gguf",
			"Qwen/X-GGUF", "", "model-q4.gguf",
		},
		{
			"https://huggingface.co/Qwen/X-GGUF/resolve/main/model-q8.gguf",
			"Qwen/X-GGUF", "", "model-q8.gguf",
		},
	}
	for _, c := range cases {
		got, err := ParseRef(c.in)
		if err != nil {
			t.Errorf("ParseRef(%q): %v", c.in, err)
			continue
		}
		if got.Repo != c.repo || got.Quant != c.quant || got.File != c.file {
			t.Errorf("ParseRef(%q) = %+v, want repo=%q quant=%q file=%q",
				c.in, got, c.repo, c.quant, c.file)
		}
	}
}

func TestParseRef_Rejects(t *testing.T) {
	for _, bad := range []string{"", "   ", "just-a-name", "too/many/slashes"} {
		if _, err := ParseRef(bad); err == nil {
			t.Errorf("ParseRef(%q) should fail", bad)
		}
	}
}

func TestPick(t *testing.T) {
	files := []File{
		{Name: "model-q3_k_m.gguf"},
		{Name: "model-q4_k_m.gguf"},
		{Name: "model-q8_0.gguf"},
	}

	// Explicit quantisation.
	if got, err := Pick(files, Ref{Quant: "Q8_0"}); err != nil || got.Name != "model-q8_0.gguf" {
		t.Errorf("Pick(Q8_0) = %v, %v", got.Name, err)
	}
	// Case-insensitive.
	if got, err := Pick(files, Ref{Quant: "q3_k_m"}); err != nil || got.Name != "model-q3_k_m.gguf" {
		t.Errorf("Pick(q3_k_m) = %v, %v", got.Name, err)
	}
	// No quantisation given → Q4_K_M is the default.
	if got, err := Pick(files, Ref{}); err != nil || got.Name != "model-q4_k_m.gguf" {
		t.Errorf("Pick(default) = %v, %v", got.Name, err)
	}
	// Exact file wins.
	if got, err := Pick(files, Ref{File: "model-q8_0.gguf", Quant: "Q4_K_M"}); err != nil || got.Name != "model-q8_0.gguf" {
		t.Errorf("Pick(exact file) = %v, %v", got.Name, err)
	}
	// A quantisation the repo does not have must fail, not fall back silently.
	if _, err := Pick(files, Ref{Quant: "Q2_K"}); err == nil {
		t.Error("Pick with a missing quantisation should fail")
	}
}

// Multi-part models must start at part 1 rather than being called ambiguous.
func TestPick_MultiPartTakesFirst(t *testing.T) {
	files := []File{
		{Name: "big-q4_k_m-00002-of-00003.gguf"},
		{Name: "big-q4_k_m-00001-of-00003.gguf"},
		{Name: "big-q4_k_m-00003-of-00003.gguf"},
	}
	got, err := Pick(files, Ref{Quant: "Q4_K_M"})
	if err != nil {
		t.Fatalf("Pick on a split model: %v", err)
	}
	if got.Name != "big-q4_k_m-00001-of-00003.gguf" {
		t.Errorf("Pick took %q, want part 00001", got.Name)
	}
}

// A blocked huggingface.co is the normal case in some regions; the mirror
// variable must be honoured, and kinfer's own must take precedence.
func TestEndpoint_HonoursMirrorVariables(t *testing.T) {
	if got := endpoint(); got != defaultEndpoint {
		t.Errorf("with no variables set, endpoint = %q, want %q", got, defaultEndpoint)
	}

	t.Setenv("HF_ENDPOINT", "https://hf-mirror.com")
	if got := endpoint(); got != "https://hf-mirror.com" {
		t.Errorf("HF_ENDPOINT ignored: got %q", got)
	}

	// Trailing slashes would produce "//api/models/..." when joined.
	t.Setenv("HF_ENDPOINT", "https://hf-mirror.com/")
	if got := endpoint(); got != "https://hf-mirror.com" {
		t.Errorf("trailing slash not trimmed: got %q", got)
	}

	t.Setenv("KINFER_HF_ENDPOINT", "https://internal.example.com")
	if got := endpoint(); got != "https://internal.example.com" {
		t.Errorf("KINFER_HF_ENDPOINT should win over HF_ENDPOINT, got %q", got)
	}

	// An empty value must not blank out the endpoint.
	t.Setenv("KINFER_HF_ENDPOINT", "   ")
	if got := endpoint(); got != "https://hf-mirror.com" {
		t.Errorf("blank KINFER_HF_ENDPOINT should fall through, got %q", got)
	}
}
