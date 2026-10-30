package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/tools"
)

// dsCase is one of DeepSeek's encoder fixtures, read into kinfer's types.
type dsCase struct {
	msgs  []Message
	tools []tools.Tool
	th    Thinking
	want  string
}

func loadDSCase(t *testing.T, n string) dsCase {
	t.Helper()
	dir := filepath.Join("testdata", "deepseek-v4.1")
	raw, err := os.ReadFile(filepath.Join(dir, "test_input_"+n+".json"))
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(filepath.Join(dir, "test_output_"+n+".txt"))
	if err != nil {
		t.Fatal(err)
	}

	// A fixture is either a bare message list or an object carrying one.
	var c struct {
		ThinkingMode    string            `json:"thinking_mode"`
		ReasoningEffort json.RawMessage   `json:"reasoning_effort"`
		Tools           []tools.Tool      `json:"tools"`
		Messages        []json.RawMessage `json:"messages"`
	}
	if raw[0] == '[' {
		if err := json.Unmarshal(raw, &c.Messages); err != nil {
			t.Fatal(err)
		}
	} else if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}

	out := dsCase{tools: c.Tools, want: string(want)}
	// DeepSeek's harness renders in chat mode unless the case asks to think.
	out.th.Off = c.ThinkingMode != "thinking"
	var effort string
	if json.Unmarshal(c.ReasoningEffort, &effort) == nil {
		out.th.Effort = effort
	}

	for _, rm := range c.Messages {
		var m struct {
			Role             string          `json:"role"`
			Content          json.RawMessage `json:"content"`
			ReasoningContent string          `json:"reasoning_content"`
			ToolCalls        []struct {
				Function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		}
		if err := json.Unmarshal(rm, &m); err != nil {
			t.Fatal(err)
		}
		msg := Message{Role: m.Role, Thinking: m.ReasoningContent}
		msg.Content = dsContent(t, m.Content)
		for _, tc := range m.ToolCalls {
			// Arguments stay as the OpenAI wire sends them, a string holding JSON.
			msg.ToolCalls = append(msg.ToolCalls, tools.Call{Name: tc.Function.Name, Arguments: tc.Function.Arguments})
		}
		out.msgs = append(out.msgs, msg)
	}
	return out
}

// dsContent flattens content the way DeepSeek's encoder does: text blocks and
// image placeholders joined by a blank line.
func dsContent(t *testing.T, raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		t.Fatalf("content: %v", err)
	}
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		} else {
			parts = append(parts, "<｜deepseek_image｜>")
		}
	}
	return strings.Join(parts, "\n\n")
}

func diffAt(a, b string) string {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++
	}
	lo := max(0, i-60)
	return "first difference at byte " + strconv.Itoa(i) + "\n  got:  …" + clip(a, lo, i+60) + "\n  want: …" + clip(b, lo, i+60)
}

func clip(s string, lo, hi int) string { return s[min(lo, len(s)):min(hi, len(s))] }

// The conversations DeepSeek publishes with its encoder, rendered byte for
// byte. These are the prompts the model was trained to read.
func TestDeepseekV41_MatchesDeepSeekEncoderFixtures(t *testing.T) {
	// 1: thinking, two functions, a call with its reasoning, a result, and an
	// answer — every turn keeps its reasoning because functions are declared.
	// 2: chat mode, two turns, reasoning dropped.
	// Both end on an assistant turn, as training data does, so the body before
	// any reply is opened is what compares.
	for _, n := range []string{"1", "2"} {
		c := loadDSCase(t, n)
		got, open := deepseekV41Body(c.msgs, c.tools, c.th)
		if got != c.want {
			t.Errorf("case %s:\n%s", n, diffAt(got, c.want))
		}
		if open {
			t.Errorf("case %s: a conversation ending on an assistant turn reported the reply already opened", n)
		}
	}

	// 4: chat mode with a latest_reminder after the system prompt. Its last
	// user turn carries a task token kinfer has no field for; everything up to
	// that token is the generation prompt kinfer writes.
	c := loadDSCase(t, "4")
	want := c.want[:strings.Index(c.want, "<｜action｜>")]
	if got := renderDeepseekV41(c.msgs[:5], nil, c.th); got != want {
		t.Errorf("case 4:\n%s", diffAt(got, want))
	}

	// 5: thinking at effort "max", ending in an open think block.
	c = loadDSCase(t, "5")
	if got := renderDeepseekV41(c.msgs, nil, c.th); got != c.want {
		t.Errorf("case 5:\n%s", diffAt(got, c.want))
	}
}

const effortLine = "Reasoning Effort: %s (range 1-100, the higher the value, the more thorough the reasoning)\n\n"

// The assertions of DeepSeek's encoder tests (encoding/test_encoding.py), as
// they apply to what kinfer can send.
func TestDeepseekV41_ReasoningEffort(t *testing.T) {
	q := []Message{{Role: "user", Content: "question"}}
	for effort, budget := range map[string]string{
		"": "75", "low": "50", "high": "75", "max": "100", "1": "1", "42": "42", "100": "100",
		// DeepSeek's server takes these too.
		"minimal": "50", "medium": "75", "xhigh": "75", "ultra": "100",
		// Out of range is the default, not an error.
		"0": "75", "101": "75",
	} {
		got := renderDeepseekV41(q, nil, Thinking{Effort: effort})
		want := dsBOS + dsSystem + strings.Replace(effortLine, "%s", budget, 1) + "<｜User｜>question<｜Assistant｜><think>"
		if got != want {
			t.Errorf("effort %q:\n%s", effort, diffAt(got, want))
		}
	}

	// Chat mode has neither the effort line nor, without a system message, the
	// system token.
	got := renderDeepseekV41([]Message{{Role: "user", Content: "hello"}}, nil, Thinking{Off: true, Effort: "max"})
	if want := dsBOS + "<｜User｜>hello<｜Assistant｜></think>"; got != want {
		t.Errorf("chat mode:\n%s", diffAt(got, want))
	}
}

func TestDeepseekV41_SystemMessages(t *testing.T) {
	got := renderDeepseekV41([]Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "hello"},
	}, nil, Thinking{Off: true})
	if want := dsBOS + "<｜System｜>You are a helpful assistant.<｜User｜>hello<｜Assistant｜></think>"; got != want {
		t.Errorf("leading system:\n%s", diffAt(got, want))
	}

	// A system message mid-conversation gets its own token and opens the reply.
	got = renderDeepseekV41([]Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "q1"},
		{Role: "assistant", Content: "a1", Thinking: "r1"},
		{Role: "system", Content: "mid sys"},
	}, nil, Thinking{Effort: "88"})
	want := dsBOS + dsSystem + strings.Replace(effortLine, "%s", "88", 1) +
		"sys<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜><｜System｜>mid sys<｜Assistant｜><think>"
	if got != want {
		t.Errorf("mid-conversation system:\n%s", diffAt(got, want))
	}
}

// Without functions declared, a finished turn's reasoning is dropped, but the
// turn still running — across its tool results — keeps it.
func TestDeepseekV41_DropsFinishedTurnsReasoning(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "q1"},
		{Role: "assistant", Content: "a1", Thinking: "r1"},
		{Role: "user", Content: "q2"},
		{Role: "assistant", Thinking: "r2", ToolCalls: []tools.Call{{Name: "lookup", Arguments: json.RawMessage(`"{}"`)}}},
		{Role: "tool", Content: "value"},
	}
	got := renderDeepseekV41(msgs, nil, Thinking{})
	if !strings.Contains(got, "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>") || strings.Contains(got, "r1") {
		t.Errorf("a finished turn's reasoning survived:\n%s", got)
	}
	if !strings.Contains(got, "<think>r2</think>") {
		t.Errorf("the running turn lost its reasoning:\n%s", got)
	}
	if !strings.HasSuffix(got, "<tool_result>value</tool_result><｜Assistant｜><think>") {
		t.Errorf("does not open the reply after the tool result:\n%s", got)
	}

	next := renderDeepseekV41(append(msgs, Message{Role: "user", Content: "q3"}), nil, Thinking{})
	if strings.Contains(next, "r2") || !strings.HasSuffix(next, "q3<｜Assistant｜><think>") {
		t.Errorf("next turn:\n%s", next)
	}
}

// V41_TOOL_CALL_OUTPUT in DeepSeek's tests: an assistant turn with reasoning,
// prose and one call, as the model writes it — and as kinfer must write it back.
const dsToolCallTurn = "  reason  </think>summary\n\n" +
	"<｜DSML｜ calls>\n" +
	"<｜DSML｜ invoke name=\"lookup\">\n" +
	"<｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter>\n" +
	"<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n" +
	"</｜DSML｜ invoke>\n" +
	"</｜DSML｜ calls><｜end▁of▁sentence｜>"

func TestDeepseekV41_ToolCallRoundTrip(t *testing.T) {
	q := Message{Role: "user", Content: "question"}
	asked := Message{Role: "assistant", Content: "summary", Thinking: "  reason  ",
		ToolCalls: []tools.Call{{Name: "lookup", Arguments: json.RawMessage(`"{\"query\":\"value\",\"limit\":2}"`)}}}
	body, _ := deepseekV41Body([]Message{q, asked}, nil, Thinking{})
	prefix := dsBOS + dsSystem + strings.Replace(effortLine, "%s", "75", 1) + "<｜User｜>question<｜Assistant｜><think>"
	if want := prefix + dsToolCallTurn; body != want {
		t.Fatalf("render:\n%s", diffAt(body, want))
	}

	// What the server sees: the reply as generated (it stops at the end of
	// sentence) with the <think> the prompt opened reported in front of it.
	reply := "<think>" + strings.TrimSuffix(dsToolCallTurn, dsEOS)
	thinking, answer := SplitThinking(reply)
	content, calls := deepseekV41.ToolFormat().Parse(answer)
	if thinking != "reason" || content != "summary" || len(calls) != 1 || calls[0].Name != "lookup" {
		t.Fatalf("parse: thinking %q, content %q, calls %+v", thinking, content, calls)
	}
	var args map[string]any
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil || args["query"] != "value" || args["limit"] != float64(2) {
		t.Fatalf("arguments %s (%v)", calls[0].Arguments, err)
	}

	// Sent back as the next turn's history, the call renders as it was written.
	parsed := Message{Role: "assistant", Content: content, Thinking: "  reason  ", ToolCalls: calls}
	if again, _ := deepseekV41Body([]Message{q, parsed}, nil, Thinking{}); again != body {
		t.Errorf("round trip:\n%s", diffAt(again, body))
	}
}

func TestDeepseekV41_ToolPromptAndTemplate(t *testing.T) {
	fn := tools.Tool{Type: "function", Function: tools.Function{Name: "f"}}
	got := renderDeepseekV41([]Message{{Role: "user", Content: "question"}}, []tools.Tool{fn}, Thinking{Off: true})
	// No system message: one is made for the tool section, which keeps its
	// leading blank line.
	if !strings.HasPrefix(got, dsBOS+"<｜System｜>\n\n## Tools\n\n") {
		t.Errorf("tool section placement:\n%s", got)
	}
	// A function without parameters is offered with {}, as DeepSeek's server
	// offers it.
	if !strings.Contains(got, `{"name": "f", "description": "", "parameters": {}}`) {
		t.Errorf("schema line:\n%s", got)
	}
	if !strings.Contains(got, "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">") || strings.Contains(got, "<｜DSML｜tool_calls>") {
		t.Errorf("tool instructions are not V4.1's spaced form:\n%s", got)
	}

	tpl, err := Get("deepseek-v4.1")
	if err != nil {
		t.Fatal(err)
	}
	if tpl.ToolFormat() != tools.DSML {
		t.Errorf("tool format %v", tpl.ToolFormat())
	}
	if !tpl.OpensThink(Thinking{}) || tpl.OpensThink(Thinking{Off: true}) {
		t.Error("OpensThink must follow the thinking switch")
	}
	if tpl.Render(nil, nil) != tpl.RenderWith(nil, nil, Thinking{}) ||
		tpl.RenderNoThink(nil, nil) != tpl.RenderWith(nil, nil, Thinking{Off: true}) {
		t.Error("Render and RenderNoThink must go through the V4.1 renderer")
	}
}

// Consecutive user turns and tool results become one user turn; "developer"
// is a system message.
func TestDeepseekV41_MergesTurns(t *testing.T) {
	got := renderDeepseekV41([]Message{
		{Role: "developer", Content: "dev"},
		{Role: "user", Content: "a"},
		{Role: "user", Content: "b"},
		{Role: "assistant", ToolCalls: []tools.Call{{Name: "f", Arguments: json.RawMessage(`{}`)}, {Name: "g", Arguments: json.RawMessage(`{"x":[1,0.00001,true,null,"é<>"]}`)}}},
		{Role: "tool", Content: "r1"},
		{Role: "tool", Content: "r2"},
		{Role: "user", Content: "c"},
	}, nil, Thinking{Off: true})
	want := dsBOS + "<｜System｜>dev<｜User｜>a\n\nb<｜Assistant｜></think>\n\n" +
		"<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n\n</｜DSML｜ invoke>\n" +
		"<｜DSML｜ invoke name=\"g\">\n<｜DSML｜ parameter name=\"x\" string=\"false\">[1, 1e-05, true, null, \"é<>\"]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n" +
		"</｜DSML｜ calls><｜end▁of▁sentence｜>" +
		"<｜User｜><tool_result>r1</tool_result>\n\n<tool_result>r2</tool_result>\n\nc<｜Assistant｜></think>"
	if got != want {
		t.Errorf("\n%s", diffAt(got, want))
	}
}
