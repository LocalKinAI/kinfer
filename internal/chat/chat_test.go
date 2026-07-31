package chat

import (
	"strings"
	"testing"
)

// The prompt must end in the assistant's opening, or the model writes another
// user turn instead of answering.
func TestRender_EndsAtAssistantTurn(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "你是金口约翰。"},
		{Role: "user", Content: "什么是施舍？"},
	}

	for _, name := range []string{"chatml", "llama3", "mistral"} {
		tpl, err := Get(name)
		if err != nil {
			t.Fatalf("Get(%q): %v", name, err)
		}
		got := tpl.Render(msgs)

		if !strings.Contains(got, "什么是施舍？") {
			t.Errorf("%s: user content missing from prompt", name)
		}
		// Mistral has no system role — it folds the system prompt into the first
		// user turn — but the text must survive either way, because that text is
		// the soul.
		if !strings.Contains(got, "你是金口约翰。") {
			t.Errorf("%s: system prompt dropped — souls would not work", name)
		}
		if strings.HasSuffix(got, "<|im_end|>\n") || strings.HasSuffix(got, "<|eot_id|>") {
			t.Errorf("%s: prompt ends on a closed turn, model will not answer", name)
		}
	}
}

func TestRender_ChatMLExact(t *testing.T) {
	tpl, _ := Get("chatml")
	got := tpl.Render([]Message{
		{Role: "system", Content: "S"},
		{Role: "user", Content: "U"},
	})
	want := "<|im_start|>system\nS<|im_end|>\n" +
		"<|im_start|>user\nU<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Errorf("chatml render:\n got %q\nwant %q", got, want)
	}
}

// Multi-turn must preserve order, or the model loses the thread.
func TestRender_MultiTurnOrder(t *testing.T) {
	tpl, _ := Get("chatml")
	got := tpl.Render([]Message{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "reply"},
		{Role: "user", Content: "second"},
	})
	iFirst := strings.Index(got, "first")
	iReply := strings.Index(got, "reply")
	iSecond := strings.Index(got, "second")
	if !(iFirst < iReply && iReply < iSecond) {
		t.Errorf("turns out of order: first=%d reply=%d second=%d", iFirst, iReply, iSecond)
	}
}

func TestTrimStop(t *testing.T) {
	tpl, _ := Get("chatml")

	cases := []struct {
		name    string
		in      string
		want    string
		wantHit bool
	}{
		{"clean stop", "施舍是信仰的试金石。<|im_end|>", "施舍是信仰的试金石。", true},
		{"no stop yet", "施舍是", "施舍是", false},
		{"trailing junk after stop", "答案<|im_end|>\n<|im_start|>user", "答案", true},
		{"endoftext variant", "done<|endoftext|>", "done", true},
		{"empty", "", "", false},
	}
	for _, c := range cases {
		got, hit := tpl.TrimStop(c.in)
		if got != c.want || hit != c.wantHit {
			t.Errorf("%s: TrimStop(%q) = (%q,%v), want (%q,%v)", c.name, c.in, got, hit, c.want, c.wantHit)
		}
	}
}

// When two stop markers are present, cut at the earliest one.
func TestTrimStop_EarliestWins(t *testing.T) {
	tpl, _ := Get("chatml")
	got, hit := tpl.TrimStop("keep<|endoftext|>drop<|im_end|>drop")
	if !hit || got != "keep" {
		t.Errorf("TrimStop = (%q,%v), want (\"keep\",true)", got, hit)
	}
}

func TestDetect(t *testing.T) {
	cases := map[string]string{
		"models/qwen2.5-0.5b-instruct-q4_k_m.gguf": "chatml",
		"Meta-Llama-3-8B-Instruct.Q4_K_M.gguf":     "llama3",
		"llama3.2-3b.gguf":                         "llama3",
		"mistral-7b-instruct-v0.3.Q4_K_M.gguf":     "mistral",
		"mixtral-8x7b.gguf":                        "mistral",
		"some-unknown-model.gguf":                  "chatml", // forgiving default
	}
	for path, want := range cases {
		if got := Detect(path).Name; got != want {
			t.Errorf("Detect(%q) = %q, want %q", path, got, want)
		}
	}
}

func TestGet_Unknown(t *testing.T) {
	if _, err := Get("no-such-template"); err == nil {
		t.Error("Get on unknown template should error")
	}
}
