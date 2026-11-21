package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/tools"
)

// fromJinjaText is FromTemplate without llama.cpp: what kinfer reads from the
// template's text, over a ChatML renderer, which is what
// llama_chat_apply_template makes of a Qwen-lineage template.
func fromJinjaText(tmpl string) *Template {
	return &Template{
		Name:             "gguf",
		toolFmt:          tools.Detect(tmpl),
		noThink:          noThinkSuffix(tmpl),
		thinkOn:          thinkOnSuffix(tmpl),
		thinksByDefault:  thinksByDefault(tmpl),
		rendersReasoning: rendersReasoning(tmpl),
		efforts:          readEfforts(tmpl),
		render:           chatml.render,
	}
}

func readTemplate(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "reasoning", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// The prompts in cases.json are the models' own templates rendered by Jinja
// (the templates are beside it, as the GGUF files carry them): kinfer's prompt
// for the same conversation, effort and thinking must be the same text.
func TestReasoningMatchesTheModelsOwnTemplates(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "reasoning", "cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name     string
		Template string
		Messages []struct {
			Role     string
			Content  string
			Thinking string
			Calls    []tools.Call
		}
		Thinking Thinking
		Want     string
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("no cases")
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			tpl := fromJinjaText(readTemplate(t, c.Template))
			var msgs []Message
			for _, m := range c.Messages {
				msgs = append(msgs, Message{Role: m.Role, Content: m.Content, Thinking: m.Thinking, ToolCalls: m.Calls})
			}
			if got := tpl.RenderWith(msgs, nil, c.Thinking); got != c.Want {
				t.Errorf("prompt differs from the template's own\n--- kinfer\n%s\n--- template\n%s", got, c.Want)
			}
		})
	}
}

func TestEffortLevelsAreReadFromTheTemplate(t *testing.T) {
	e := readEfforts(readTemplate(t, "qwen3.8-27b.jinja"))
	if e == nil {
		t.Fatal("Qwen3.8's template has reasoning effort levels")
	}
	for _, lv := range []string{"xhigh", "medium", "low"} {
		if _, ok := e.levels[lv]; !ok {
			t.Errorf("level %q not read", lv)
		}
	}
	if e.levels["medium"] != "" {
		t.Errorf("medium writes no instruction in this template, got %q", e.levels["medium"])
	}
	for effort, want := range map[string]string{"low": "set to low", "minimal": "set to low", "20": "set to low",
		"high": "set to xhigh", "max": "set to xhigh", "90": "set to xhigh", "xhigh": "set to xhigh"} {
		got, ok := e.instruction(effort)
		if !ok || !strings.Contains(got, want) {
			t.Errorf("effort %q: %q, want one that says %q", effort, got, want)
		}
	}
	if got, ok := e.instruction("50"); !ok || got != "" {
		t.Errorf("50 is medium, which writes nothing: %q %v", got, ok)
	}
	if _, ok := e.instruction("banana"); ok {
		t.Error("a level nobody has is no level")
	}
	if readEfforts(readTemplate(t, "ornith-1.5-35b.jinja")) != nil {
		t.Error("ornith's template has no reasoning effort")
	}
}

// What did not change: a caller that names no effort and sends no reasoning
// back gets the prompt it got before.
func TestNoEffortAndNoReasoningIsThePromptAsBefore(t *testing.T) {
	for _, name := range []string{"qwen3.8-27b.jinja", "qwen3.8-flash-next.jinja", "ornith-1.5-35b.jinja"} {
		tpl := fromJinjaText(readTemplate(t, name))
		msgs := []Message{{Role: "system", Content: "s"}, {Role: "user", Content: "q1"},
			{Role: "assistant", Content: "a1"}, {Role: "user", Content: "q2"}}
		for _, th := range []Thinking{{}, {On: true}, {Off: true}} {
			got := tpl.RenderWith(msgs, nil, th)
			if strings.Contains(got, "Reasoning effort") || strings.Contains(got, "<think>\n\n</think>\n\na1") || strings.Contains(got, "</think>\n\na1") {
				t.Errorf("%s %+v: the prompt gained something nobody asked for:\n%s", name, th, got)
			}
		}
	}
}

// The default keeps reasoning for the turns since the user's last question;
// KeepAll for every turn, as ornith's template (which has no switch) and
// Qwen3.8's default do.
func TestKeepAllGivesEveryTurnItsReasoning(t *testing.T) {
	tpl := fromJinjaText(readTemplate(t, "ornith-1.5-35b.jinja"))
	msgs := []Message{{Role: "user", Content: "q1"}, {Role: "assistant", Content: "a1", Thinking: "t1"},
		{Role: "user", Content: "q2"}}
	if got := tpl.RenderWith(msgs, nil, Thinking{}); strings.Contains(got, "t1") {
		t.Errorf("reasoning from before the last question came back unasked:\n%s", got)
	}
	got := tpl.RenderWith(msgs, nil, Thinking{KeepAll: true})
	if !strings.Contains(got, "<|im_start|>assistant\n<think>\nt1\n</think>\n\na1<|im_end|>") {
		t.Errorf("KeepAll: the earlier turn's reasoning is not in its think block:\n%s", got)
	}
}

func TestATemplateWithoutTheBranchesIsLeftAlone(t *testing.T) {
	tpl := fromJinjaText("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}")
	msgs := []Message{{Role: "user", Content: "q"}, {Role: "assistant", Content: "a", Thinking: "t"}, {Role: "tool", Content: "r"}}
	if got := tpl.RenderWith(msgs, nil, Thinking{Effort: "low", KeepAll: true}); strings.Contains(got, "<think>") || strings.Contains(got, "Reasoning") {
		t.Errorf("a template with no reasoning branches got them:\n%s", got)
	}
}
