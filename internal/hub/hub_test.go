package hub

import (
	"strings"
	"testing"
)

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
		// A file in a folder, as a link copied from the website has it.
		{
			"https://huggingface.co/unsloth/Big-GGUF/resolve/main/Q2_K/Big-Q2_K-00001-of-00003.gguf?download=true",
			"unsloth/Big-GGUF", "", "Q2_K/Big-Q2_K-00001-of-00003.gguf",
		},
		{
			"https://hf-mirror.com/Qwen/X-GGUF/blob/main/model-q4.gguf",
			"Qwen/X-GGUF", "", "model-q4.gguf",
		},
		{"https://huggingface.co/Qwen/X-GGUF/tree/main", "Qwen/X-GGUF", "", ""},
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
	for _, bad := range []string{"", "   ", "just-a-name", "too/many/slashes", "https://huggingface.co/", "https://huggingface.co/Qwen"} {
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

// A pull of a split model needs every piece, whichever one Pick landed on.
func TestShards(t *testing.T) {
	files := []File{
		{Name: "Q2/big-Q2-00001-of-00003.gguf"},
		{Name: "Q2/big-Q2-00002-of-00003.gguf"},
		{Name: "Q2/big-Q2-00003-of-00003.gguf"},
		{Name: "Q4/big-Q4-00001-of-00002.gguf"},
		{Name: "Q4/big-Q4-00002-of-00002.gguf"},
		{Name: "small-Q8_0.gguf"},
	}
	names := func(fs []File) string { return join(fs) }

	if got := names(Shards(files, files[0])); got != names(files[:3]) {
		t.Errorf("Shards(part 1 of 3) = %s", got)
	}
	if got := names(Shards(files, files[4])); got != names(files[3:5]) {
		t.Errorf("Shards(part 2 of 2) = %s", got)
	}
	if got := names(Shards(files, files[5])); got != files[5].Name {
		t.Errorf("Shards(a file that stands alone) = %s", got)
	}
}

// A vision repo's projector: F16 first, then BF16 — which "f16" must not be
// mistaken for — then F32, then whatever there is.
func TestProjector(t *testing.T) {
	cases := []struct {
		files []string
		want  string
	}{
		{[]string{"m-Q4_K_M.gguf", "mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"}, "mmproj-F16.gguf"},
		{[]string{"m-Q4_K_M.gguf", "mmproj-model-f32.gguf", "mmproj-model-bf16.gguf"}, "mmproj-model-bf16.gguf"},
		{[]string{"m-Q4_K_M.gguf", "mmproj-Q8_0.gguf"}, "mmproj-Q8_0.gguf"},
		{[]string{"m-Q4_K_M.gguf"}, ""},
	}
	for _, c := range cases {
		var files []File
		for _, n := range c.files {
			files = append(files, File{Name: n})
		}
		got, ok := Projector(files)
		if got.Name != c.want || ok != (c.want != "") {
			t.Errorf("Projector(%v) = %q, %v; want %q", c.files, got.Name, ok, c.want)
		}
	}
}

func TestProjectorName(t *testing.T) {
	if got := ProjectorName([]File{{Name: "Qwen3-VL-8B-Q4_K_M.gguf"}}); got != "Qwen3-VL-8B-Q4_K_M.mmproj.gguf" {
		t.Errorf("one file: %q", got)
	}
	set := []File{{Name: "Q2/Big-Q2-00001-of-00002.gguf"}, {Name: "Q2/Big-Q2-00002-of-00002.gguf"}}
	if got := ProjectorName(set); got != "Big-Q2.mmproj.gguf" {
		t.Errorf("a split model: %q, want the name the store gives the set", got)
	}
}

// A projector is never the model a quantization picks.
func TestPickLeavesProjectorsOut(t *testing.T) {
	files := []File{{Name: "mmproj-F16.gguf"}, {Name: "model-F16.gguf"}, {Name: "model-Q8_0.gguf"}}
	got, err := Pick(files, Ref{Repo: "o/r", Quant: "F16"})
	if err != nil || got.Name != "model-F16.gguf" {
		t.Errorf("Pick F16 = %q, %v", got.Name, err)
	}
	if got, _ := Pick([]File{{Name: "mmproj-F16.gguf"}, {Name: "model-Q8_0.gguf"}}, Ref{Repo: "o/r"}); got.Name != "model-Q8_0.gguf" {
		t.Errorf("Pick with no quantization = %q", got.Name)
	}
}

// An MLX repo is pulled as the directory it is, minus what kinfer does not
// read; config.json comes last, so the directory is no model until the end.
func TestMLXFiles(t *testing.T) {
	repo := []File{
		{Name: ".gitattributes"}, {Name: "README.md"}, {Name: "chat_template.jinja"},
		{Name: "config.json"}, {Name: "merges.txt"}, {Name: "model-00001-of-00002.safetensors", Size: 5},
		{Name: "model-00002-of-00002.safetensors", Size: 3}, {Name: "model.safetensors.index.json"},
		{Name: "original/model.safetensors"}, {Name: "tokenizer.json"}, {Name: "tokenizer_config.json"},
	}
	got, ok := MLXFiles(repo)
	if !ok {
		t.Fatal("a repo with config.json, tokenizer.json and safetensors is no MLX model")
	}
	want := "model-00001-of-00002.safetensors model-00002-of-00002.safetensors chat_template.jinja merges.txt " +
		"model.safetensors.index.json tokenizer.json tokenizer_config.json config.json"
	if s := strings.TrimSpace(join(got)); strings.ReplaceAll(s, ", ", " ") != want {
		t.Errorf("MLXFiles = %s\nwant          %s", s, want)
	}
	// Without its config, its tokenizer or its weights, it is no model.
	for _, drop := range []string{"config.json", "tokenizer.json", ".safetensors"} {
		var less []File
		for _, f := range repo {
			if !strings.HasSuffix(f.Name, drop) {
				less = append(less, f)
			}
		}
		if _, ok := MLXFiles(less); ok {
			t.Errorf("without its %s it is still an MLX model", drop)
		}
	}
	if len(GGUFs(repo)) != 0 {
		t.Error("an MLX repo has GGUFs")
	}
}

// Qwen3-TTS keeps its codec in speech_tokenizer/, a model of its own beside
// the talker: its files come too, before config.json; another folder's not.
func TestMLXFilesTakeQwen3TTSsCodec(t *testing.T) {
	repo := []File{
		{Name: "config.json"}, {Name: "generation_config.json"}, {Name: "merges.txt"}, {Name: "vocab.json"},
		{Name: "model.safetensors", Size: 9}, {Name: "speech_tokenizer/config.json"},
		{Name: "speech_tokenizer/model.safetensors", Size: 7}, {Name: "speech_tokenizer/old/model.safetensors"},
		{Name: "samples/voice.wav"},
	}
	got, ok := MLXFiles(repo)
	if !ok {
		t.Fatal("Qwen3-TTS is no MLX model")
	}
	want := "model.safetensors generation_config.json merges.txt vocab.json speech_tokenizer/config.json speech_tokenizer/model.safetensors config.json"
	if s := strings.ReplaceAll(strings.TrimSpace(join(got)), ", ", " "); s != want {
		t.Errorf("MLXFiles = %s\nwant          %s", s, want)
	}
}
