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

	content, calls := Hermes.Parse(reply)

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
		content, calls := Hermes.Parse(reply)
		if len(calls) != 0 {
			t.Errorf("Hermes.Parse(%q) invented %d calls", reply, len(calls))
		}
		if !strings.Contains(content, "before") {
			t.Errorf("Hermes.Parse(%q) lost the surrounding text: %q", reply, content)
		}
	}
}

func TestParsePassesPlainRepliesThrough(t *testing.T) {
	content, calls := Hermes.Parse("Just an ordinary answer.")
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
	got := Hermes.Declare([]Tool{{
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
			t.Errorf("Hermes.Declare() is missing %q\ngot:\n%s", want, got)
		}
	}

	if Hermes.Declare(nil) != "" {
		t.Error("Hermes.Declare(nil) should add nothing to the system prompt")
	}
}

func TestRenderCallsRoundTrips(t *testing.T) {
	calls := []Call{{Name: "f", Arguments: json.RawMessage(`{"a":1}`)}}
	_, back := Hermes.Parse(Hermes.RenderCalls(calls))
	if len(back) != 1 || back[0].Name != "f" {
		t.Fatalf("round trip produced %v", back)
	}
	if string(back[0].Arguments) != `{"a":1}` {
		t.Errorf("arguments changed on the round trip: %s", back[0].Arguments)
	}
}

func TestDetectReadsTheModelsOwnTemplate(t *testing.T) {
	if Detect("…within <tool_call></tool_call> XML tags…") != Hermes {
		t.Error("a template that names <tool_call> was reported unsupported")
	}
	if Detect("<|im_start|>system\n{{ content }}<|im_end|>") != None {
		t.Error("a template with no tool convention was reported supported")
	}
}

// The XML convention, which is what made ornith-1.5:35b answer in garbage when
// it was told to use the JSON one.

func TestDetectPrefersTheXMLConventionWhenBothTagsAppear(t *testing.T) {
	// Both conventions wrap calls in <tool_call>, so only the inner form tells
	// them apart. Checking the outer tag first would misread every XML model as
	// Hermes — which is exactly the bug this replaced.
	xml := `…ONLY reply in the following format:\n<tool_call>\n<function=example>\n</function>\n</tool_call>`
	if got := Detect(xml); got != FunctionXML {
		t.Errorf("Detect(xml template) = %v, want function-xml", got)
	}
	if got := Detect("…json object within <tool_call></tool_call> XML tags…"); got != Hermes {
		t.Errorf("Detect(hermes template) = %v, want hermes", got)
	}
}

func TestFunctionXMLParse(t *testing.T) {
	reply := `Let me check.
<tool_call>
<function=get_weather>
<parameter=city>
Berlin
</parameter>
<parameter=days>
3
</parameter>
</function>
</tool_call>`

	content, calls := FunctionXML.Parse(reply)
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if calls[0].Name != "get_weather" {
		t.Errorf("name = %q", calls[0].Name)
	}

	var args map[string]any
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil {
		t.Fatalf("arguments are not valid JSON: %v", err)
	}
	if args["city"] != "Berlin" {
		t.Errorf(`city = %v, want "Berlin"`, args["city"])
	}
	// A number written as a parameter should arrive as a number, not "3".
	if n, ok := args["days"].(float64); !ok || n != 3 {
		t.Errorf("days = %#v, want the number 3", args["days"])
	}
	if !strings.Contains(content, "Let me check") {
		t.Errorf("prose was lost: %q", content)
	}
}

func TestFunctionXMLRoundTrips(t *testing.T) {
	calls := []Call{{Name: "get_weather", Arguments: json.RawMessage(`{"city":"Berlin"}`)}}
	rendered := FunctionXML.RenderCalls(calls)
	if !strings.Contains(rendered, "<function=get_weather>") || !strings.Contains(rendered, "<parameter=city>") {
		t.Fatalf("RenderCalls produced the wrong convention:\n%s", rendered)
	}
	_, back := FunctionXML.Parse(rendered)
	if len(back) != 1 || back[0].Name != "get_weather" {
		t.Fatalf("round trip produced %v", back)
	}
	var args map[string]string
	if json.Unmarshal(back[0].Arguments, &args) != nil || args["city"] != "Berlin" {
		t.Errorf("arguments did not survive the round trip: %s", back[0].Arguments)
	}
}

func TestEachFormatDeclaresItsOwnWording(t *testing.T) {
	ts := []Tool{{Type: "function", Function: Function{Name: "f"}}}

	hermes, xml := Hermes.Declare(ts), FunctionXML.Declare(ts)
	if !strings.Contains(hermes, "You may call one or more functions") {
		t.Error("the Hermes declaration lost its trained wording")
	}
	if !strings.Contains(xml, "You have access to the following functions") {
		t.Error("the XML declaration lost its trained wording")
	}
	if !strings.Contains(xml, "<function=example_function_name>") {
		t.Error("the XML declaration does not show the format it asks for")
	}
	if hermes == xml {
		t.Error("both formats declared the same thing; one model would be told the wrong convention")
	}
	if None.Declare(ts) != "" {
		t.Error("a model with no convention was given a declaration anyway")
	}
}

func TestNoneParsesNothing(t *testing.T) {
	reply := "<tool_call>\n{\"name\":\"f\",\"arguments\":{}}\n</tool_call>"
	content, calls := None.Parse(reply)
	if calls != nil {
		t.Error("a model with no convention produced a call")
	}
	if content != reply {
		t.Error("the reply was altered")
	}
}
