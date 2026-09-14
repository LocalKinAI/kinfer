package chat

import (
	"github.com/LocalKinAI/kinfer/internal/tools"
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
		got := tpl.Render(msgs, nil)

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
	}, nil)
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
	}, nil)
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

// FromModel needs a loaded model, so what is unit-testable is the fallback:
// a file whose metadata carries no template must still get a sensible guess
// rather than a nil template and a panic on the first request. The path that
// matters — a real GGUF template rendering the format the model was trained
// on — is exercised end to end, and the difference is stark: Gemma 3 rendered
// through its own template stops cleanly, while the same model rendered as
// ChatML leaks "<|im_end" into the reply, because ChatML's marker is not what
// this model was taught to stop at.
func TestFromModelFallsBackWithoutAModel(t *testing.T) {
	cases := []struct{ path, want string }{
		{"/models/llama-3-8b-instruct.gguf", "llama3"},
		{"/models/mistral-7b-instruct.gguf", "mistral"},
		{"/models/qwen2.5-0.5b-instruct.gguf", "chatml"},
	}
	for _, c := range cases {
		got := FromModel(0, c.path)
		if got == nil {
			t.Fatalf("FromModel(0, %q) returned nil", c.path)
		}
		if got.Name != c.want {
			t.Errorf("FromModel(0, %q) = %q, want the filename guess %q", c.path, got.Name, c.want)
		}
		if got.Render(nil, nil) == "" && len(got.Stops) == 0 {
			t.Errorf("%q fell back to something unusable", c.path)
		}
	}
}

func TestFoldDeclaresToolsInTheSystemTurn(t *testing.T) {
	ts := []tools.Tool{{Type: "function", Function: tools.Function{Name: "get_weather"}}}

	// With a system turn, the declaration joins it rather than adding another.
	got := fold([]Message{
		{Role: "system", Content: "You are terse."},
		{Role: "user", Content: "Weather?"},
	}, ts, tools.Hermes)
	if len(got) != 2 {
		t.Fatalf("fold produced %d messages, want 2", len(got))
	}
	if !strings.HasPrefix(got[0].Content, "You are terse.") {
		t.Errorf("the system prompt was replaced rather than extended: %q", got[0].Content)
	}
	if !strings.Contains(got[0].Content, "get_weather") {
		t.Errorf("the tool never reached the prompt: %q", got[0].Content)
	}

	// Without one, a system turn is created — the declaration has nowhere else
	// to go, and a model that never sees it will never call anything.
	got = fold([]Message{{Role: "user", Content: "Weather?"}}, ts, tools.Hermes)
	if len(got) != 2 || got[0].Role != "system" {
		t.Fatalf("fold did not add a system turn: %+v", got)
	}
	if strings.HasPrefix(got[0].Content, "\n") {
		t.Errorf("a synthesised system turn starts with blank lines: %q", got[0].Content)
	}
}

func TestFoldRewritesToolResultsAsUserTurns(t *testing.T) {
	// No instruct model has a "tool" role; Qwen's own template turns results
	// into a user turn wrapped in <tool_response>.
	got := fold([]Message{
		{Role: "user", Content: "Weather?"},
		{Role: "assistant", ToolCalls: []tools.Call{{Name: "get_weather", Arguments: []byte(`{"city":"Berlin"}`)}}},
		{Role: "tool", Content: `{"temp": 7}`},
	}, nil, tools.Hermes)

	if got[1].Role != "assistant" || !strings.Contains(got[1].Content, "<tool_call>") {
		t.Errorf("the assistant's call did not become text: %+v", got[1])
	}
	if len(got[1].ToolCalls) != 0 {
		t.Error("tool calls were left on the message as well as rendered into it")
	}
	if got[2].Role != "user" {
		t.Errorf("a tool result kept role %q; no model knows that role", got[2].Role)
	}
	if !strings.Contains(got[2].Content, "<tool_response>") || !strings.Contains(got[2].Content, `"temp": 7`) {
		t.Errorf("the tool result was not wrapped: %q", got[2].Content)
	}
}

func TestFoldWithoutToolsChangesNothing(t *testing.T) {
	in := []Message{{Role: "system", Content: "s"}, {Role: "user", Content: "u"}}
	got := fold(in, nil, tools.Hermes)
	if len(got) != 2 || got[0].Content != "s" || got[1].Content != "u" {
		t.Errorf("fold altered a plain conversation: %+v", got)
	}
}

// The Qwen3 family disables thinking with an empty think block written into
// the prompt. That text has to come from the template — it is the model's own
// convention — and must not be appended to a model that has no such switch.
func TestNoThinkSuffixComesFromTheTemplate(t *testing.T) {
	qwen := `{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- else %}
    {{- '<think>\n' }}
{%- endif %}`
	if got := noThinkSuffix(qwen); got != "<think>\n\n</think>\n\n" {
		t.Errorf("Qwen3 template: suffix = %q", got)
	}
	llama3 := "{% for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>{{ m.content }}<|eot_id|>{% endfor %}"
	if got := noThinkSuffix(llama3); got != "" {
		t.Errorf("a template with no thinking switch got suffix %q", got)
	}
}

// RenderNoThink must change nothing for a model without the switch, and for
// one with it must end the prompt inside a closed, empty think block — so the
// model's first token is the answer, not the start of its reasoning.
func TestRenderNoThinkEndsInAnEmptyThinkBlock(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "Why do rivers meander?"}}

	plain := &Template{Name: "plain", render: chatml.render}
	if a, b := plain.Render(msgs, nil), plain.RenderNoThink(msgs, nil); a != b {
		t.Errorf("a template with no switch rendered differently with NoThink:\n%q\n%q", a, b)
	}

	thinker := &Template{Name: "thinker", render: chatml.render, noThink: "<think>\n\n</think>\n\n"}
	got := thinker.RenderNoThink(msgs, nil)
	want := "<|im_start|>user\nWhy do rivers meander?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
	if got != want {
		t.Errorf("RenderNoThink:\n got %q\nwant %q", got, want)
	}
	if thinker.Render(msgs, nil) == got {
		t.Error("Render and RenderNoThink are the same on a thinking model — nothing was injected")
	}
}
