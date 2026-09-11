package tools

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestParseExtractsCallsAndKeepsProseInOrder(t *testing.T) {
	reply := `Let me look that up.
<tool_call>
{"name": "get_weather", "arguments": {"city": "Berlin"}}
</tool_call>
And the other one.
<tool_call>
{"name": "get_time", "arguments": {"tz": "CET"}}
</tool_call>`

	content, calls := Parse(reply)

	if len(calls) != 2 {
		t.Fatalf("got %d calls, want 2", len(calls))
	}
	if calls[0].Name != "get_weather" || calls[1].Name != "get_time" {
		t.Errorf("call names = %q, %q", calls[0].Name, calls[1].Name)
	}

	var args map[string]string
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil {
		t.Fatalf("arguments are not valid JSON: %v", err)
	}
	if args["city"] != "Berlin" {
		t.Errorf("arguments = %v, want city=Berlin", args)
	}

	// Prose must stay in the order the model wrote it — writing the text
	// around a tag out of order would reverse the reply.
	if i, j := strings.Index(content, "look that up"), strings.Index(content, "other one"); i < 0 || j < 0 || i > j {
		t.Errorf("prose is missing or reordered: %q", content)
	}
	if strings.Contains(content, openTag) {
		t.Errorf("a parsed call was left in the content: %q", content)
	}
}

func TestParseLeavesMalformedCallsVisible(t *testing.T) {
	// A half-written call is something the caller should see, not something to
	// silently drop — it is the difference between "the model said nothing" and
	// "the model was cut off mid-call".
	for _, reply := range []string{
		"before <tool_call>\n{not json}\n</tool_call> after",
		"before <tool_call>\n{\"name\": \"x\"",
		"before <tool_call>\n{\"arguments\": {}}\n</tool_call>", // no name
	} {
		content, calls := Parse(reply)
		if len(calls) != 0 {
			t.Errorf("Parse(%q) invented %d calls", reply, len(calls))
		}
		if !strings.Contains(content, "before") {
			t.Errorf("Parse(%q) lost the surrounding text: %q", reply, content)
		}
	}
}

func TestParsePassesPlainRepliesThrough(t *testing.T) {
	content, calls := Parse("Just an ordinary answer.")
	if calls != nil {
		t.Errorf("a reply with no tags produced %d calls", len(calls))
	}
	if content != "Just an ordinary answer." {
		t.Errorf("content = %q, want it untouched", content)
	}
}

// TestDeclareMatchesTheTrainedWording: the model was fine-tuned on this exact
// text. A paraphrase is a different prompt.
func TestDeclareMatchesTheTrainedWording(t *testing.T) {
	got := Declare([]Tool{{
		Type:     "function",
		Function: Function{Name: "f", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)},
	}})

	for _, want := range []string{
		"# Tools",
		"You may call one or more functions to assist with the user query.",
		"You are provided with function signatures within <tools></tools> XML tags:",
		"<tools>", "</tools>",
		`{"name": <function-name>, "arguments": <args-json-object>}`,
		`"name":"f"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("Declare() is missing %q\ngot:\n%s", want, got)
		}
	}

	if Declare(nil) != "" {
		t.Error("Declare(nil) should add nothing to the system prompt")
	}
}

func TestRenderCallsRoundTrips(t *testing.T) {
	calls := []Call{{Name: "f", Arguments: json.RawMessage(`{"a":1}`)}}
	_, back := Parse(RenderCalls(calls))
	if len(back) != 1 || back[0].Name != "f" {
		t.Fatalf("round trip produced %v", back)
	}
	if string(back[0].Arguments) != `{"a":1}` {
		t.Errorf("arguments changed on the round trip: %s", back[0].Arguments)
	}
}

func TestSupportedReadsTheModelsOwnTemplate(t *testing.T) {
	if !Supported("…within <tool_call></tool_call> XML tags…") {
		t.Error("a template that names <tool_call> was reported unsupported")
	}
	if Supported("<|im_start|>system\n{{ content }}<|im_end|>") {
		t.Error("a template with no tool convention was reported supported")
	}
}
