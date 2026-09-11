package hub

import "testing"

// A multimodal repo ships its vision projector as a GGUF of its own. It is
// usually the smallest file there and declares a different architecture
// ("clip"), so mistaking it for the model means both reading the wrong
// architecture and telling someone to download a few hundred megabytes that
// cannot answer a question.
func TestIsModelExcludesWhatShipsBesideTheModel(t *testing.T) {
	for _, n := range []string{
		"mmproj-F16.gguf",
		"mmproj-BF16.gguf",
		"UD-Q2_K_XL/mmproj-F16.gguf",
		"Qwen3.8-Flash-Next-imatrix.gguf",
		"imatrix_unsloth.gguf",
	} {
		if IsModel(n) {
			t.Errorf("IsModel(%q) = true, want false", n)
		}
	}
	for _, n := range []string{
		"Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf",
		"UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00003.gguf",
		"qwen2.5-0.5b-instruct-q4_k_m.gguf",
	} {
		if !IsModel(n) {
			t.Errorf("IsModel(%q) = false, want true", n)
		}
	}
}

func TestGgufArchitecture(t *testing.T) {
	// GGUF: magic, version, tensor count, kv count, then the first key —
	// general.architecture — as a string.
	b := []byte("GGUF")
	b = append(b, 3, 0, 0, 0)
	b = append(b, 0, 0, 0, 0, 0, 0, 0, 0) // tensors
	b = append(b, 1, 0, 0, 0, 0, 0, 0, 0) // one kv
	key := "general.architecture"
	b = append(b, byte(len(key)), 0, 0, 0, 0, 0, 0, 0)
	b = append(b, key...)
	b = append(b, 8, 0, 0, 0) // type 8 = string
	val := "qwen4exp"
	b = append(b, byte(len(val)), 0, 0, 0, 0, 0, 0, 0)
	b = append(b, val...)

	got, err := ggufArchitecture(b)
	if err != nil {
		t.Fatalf("ggufArchitecture: %v", err)
	}
	if got != "qwen4exp" {
		t.Errorf("got %q, want qwen4exp", got)
	}

	if _, err := ggufArchitecture([]byte("not a gguf at all")); err == nil {
		t.Error("a non-GGUF was accepted")
	}
	// A header cut short must fail rather than return whatever it read: the
	// bytes come from a ranged request that can legitimately stop anywhere.
	if _, err := ggufArchitecture(b[:len(b)-4]); err == nil {
		t.Error("a truncated header was accepted")
	}
}

func TestIsModelExcludesDraftHeads(t *testing.T) {
	// A repo can ship speculative-decoding draft heads beside the model — a few
	// dozen tensors, several gigabytes, and useless on their own.
	for _, n := range []string{
		"MTP/mtp-Qwen3.8-Flash-Next-BF16.gguf",
		"MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf",
	} {
		if IsModel(n) {
			t.Errorf("IsModel(%q) = true, want false", n)
		}
	}
}

func TestQuantName(t *testing.T) {
	cases := []struct{ file, base, want string }{
		// A directory holding one split model is named for the quantisation.
		{"UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf",
			"UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL", "UD-Q2_K_XL"},
		// Everything in one directory: the quantisation trails the name.
		{"qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen2.5-0.5b-instruct-q4_k_m", "q4_k_m"},
		{"DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf",
			"DeepSeek-V4.1-Flash-Q2_K", "Q2_K"},
	}
	for _, c := range cases {
		if got := quantName(c.file, c.base); got != c.want {
			t.Errorf("quantName(%q) = %q, want %q", c.file, got, c.want)
		}
	}
}
