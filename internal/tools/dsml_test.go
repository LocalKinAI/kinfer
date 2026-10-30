package tools

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPythonJSON_MatchesJSONDumps(t *testing.T) {
	// Each right-hand side is what Python 3's json.dumps(json.loads(lhs),
	// ensure_ascii=False) prints.
	for in, want := range map[string]string{
		`{"b":1,"a":[1,2]}`:             `{"b": 1, "a": [1, 2]}`, // order kept, spaced separators
		`{"x":1.0,"y":1e2,"z":100}`:     `{"x": 1.0, "y": 100.0, "z": 100}`,
		`[0.00001,1e-7,0.0001,0.5]`:     `[1e-05, 1e-07, 0.0001, 0.5]`,
		`[1e16,1e15,123456789.25]`:      `[1e+16, 1000000000000000.0, 123456789.25]`,
		`[-0,-0.0,0.0,1e400,1e-400]`:    `[0, -0.0, 0.0, Infinity, 0.0]`,
		`"é <a> & \"q\" \\ \n\t\u0001"`: `"é <a> & \"q\" \\ \n\t\u0001"`,
		`{"k":{"a":1,"a":2}}`:           `{"k": {"a": 2}}`, // a repeated key keeps the last value
		`{}`:                            `{}`,
		`[true,false,null]`:             `[true, false, null]`,
	} {
		got, err := PythonJSON([]byte(in))
		if err != nil || got != want {
			t.Errorf("PythonJSON(%s) = %s, %v; want %s", in, got, err, want)
		}
	}
	if _, err := PythonJSON([]byte(`{"a":1} x`)); err == nil {
		t.Error("trailing data accepted")
	}
}

func TestDSML_ParsesBothSpellings(t *testing.T) {
	// The spaced V4.1 form, and V4's unspaced one, which DeepSeek's server
	// parser also accepts.
	v41 := "Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n" +
		"<｜DSML｜ parameter name=\"location\" string=\"true\">Tokyo</｜DSML｜ parameter>\n" +
		"<｜DSML｜ parameter name=\"unit\" string=\"false\">{\"value\": \"celsius\"}</｜DSML｜ parameter>\n" +
		"</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"get_weather\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
	v4 := strings.NewReplacer("｜DSML｜ calls", "｜DSML｜tool_calls", "｜DSML｜ invoke", "｜DSML｜invoke",
		"｜DSML｜ parameter", "｜DSML｜parameter").Replace(v41)

	for name, reply := range map[string]string{"v4.1": v41, "v4": v4} {
		content, calls := DSML.Parse(reply)
		if content != "Checking." || len(calls) != 2 {
			t.Fatalf("%s: content %q, %d calls", name, content, len(calls))
		}
		if calls[0].Name != "get_weather" || string(calls[0].Arguments) != `{"location": "Tokyo", "unit": {"value": "celsius"}}` {
			t.Errorf("%s: first call %s %s", name, calls[0].Name, calls[0].Arguments)
		}
		if string(calls[1].Arguments) != "{}" {
			t.Errorf("%s: an empty invoke has arguments %s", name, calls[1].Arguments)
		}
	}
}

func TestDSML_ArgumentsAreAlwaysValidJSON(t *testing.T) {
	reply := "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n" +
		// JSON text inside a string value must be escaped, not inserted.
		"<｜DSML｜ parameter name=\"s\" string=\"true\">{\"value\": \"Paris\"}\nline</｜DSML｜ parameter>\n" +
		// Flagged as JSON but not JSON: kept as a string rather than breaking the call.
		"<｜DSML｜ parameter name=\"n\" string=\"false\">three</｜DSML｜ parameter>\n" +
		// A repeated parameter keeps its first value.
		"<｜DSML｜ parameter name=\"s\" string=\"true\">again</｜DSML｜ parameter>\n" +
		"</｜DSML｜ invoke>\n</｜DSML｜ calls>"
	_, calls := DSML.Parse(reply)
	if len(calls) != 1 {
		t.Fatalf("%d calls", len(calls))
	}
	var args map[string]any
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil {
		t.Fatalf("arguments %s: %v", calls[0].Arguments, err)
	}
	if args["s"] != "{\"value\": \"Paris\"}\nline" || args["n"] != "three" || len(args) != 2 {
		t.Errorf("arguments %s", calls[0].Arguments)
	}
}

func TestDSML_KeepsWhatItCannotRead(t *testing.T) {
	// Cut off mid-call: the caller sees it.
	cut := "Let me look.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"q\" string=\"true\">x"
	if content, calls := DSML.Parse(cut); len(calls) != 0 || content != strings.TrimSpace(cut) {
		t.Errorf("unterminated block: %q, %v", content, calls)
	}
	// A block with no readable invoke stays visible too.
	junk := "a\n\n<｜DSML｜ calls>\nnonsense\n</｜DSML｜ calls>"
	if content, calls := DSML.Parse(junk); len(calls) != 0 || !strings.Contains(content, "nonsense") {
		t.Errorf("unreadable block: %q, %v", content, calls)
	}
	// Prose after a block is kept.
	after := "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\nDone."
	if content, calls := DSML.Parse(after); len(calls) != 1 || content != "Done." {
		t.Errorf("prose after the block: %q, %v", content, calls)
	}
	// No block at all is plain prose.
	if content, calls := DSML.Parse("  just text  "); content != "just text" || calls != nil {
		t.Errorf("plain reply: %q, %v", content, calls)
	}
}

func TestDSML_RenderCallsTakesArgumentsInEitherForm(t *testing.T) {
	want := "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n" +
		"<｜DSML｜ parameter name=\"q\" string=\"true\">v</｜DSML｜ parameter>\n" +
		"<｜DSML｜ parameter name=\"n\" string=\"false\">2</｜DSML｜ parameter>\n" +
		"</｜DSML｜ invoke>\n</｜DSML｜ calls>"
	// An object (Ollama), a string holding one (OpenAI), and the same string
	// encoded twice, which DeepSeek's encoder also unwraps.
	for _, args := range []string{`{"q":"v","n":2}`, `"{\"q\":\"v\",\"n\":2}"`, `"\"{\\\"q\\\":\\\"v\\\",\\\"n\\\":2}\""`} {
		if got := DSML.RenderCalls([]Call{{Name: "f", Arguments: json.RawMessage(args)}}); got != want {
			t.Errorf("arguments %s:\n got %q\nwant %q", args, got, want)
		}
	}
	// Not an object at all: one parameter named "arguments".
	got := DSML.RenderCalls([]Call{{Name: "f", Arguments: json.RawMessage(`"plain"`)}})
	if !strings.Contains(got, `<｜DSML｜ parameter name="arguments" string="true">plain</｜DSML｜ parameter>`) {
		t.Errorf("non-object arguments: %q", got)
	}
	if DSML.RenderCalls(nil) != "" {
		t.Error("no calls rendered a block")
	}
}

func TestCallSplitter_HoldsBackDSML(t *testing.T) {
	reply := "Sure.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
	var s CallSplitter
	var shown strings.Builder
	// One rune at a time, the way a stream can split a multi-byte token.
	for _, r := range reply {
		shown.WriteString(s.Next(string(r)))
	}
	shown.WriteString(s.Flush())
	if got := shown.String(); got != "Sure.\n\n" {
		t.Errorf("shown %q; the call leaked into the stream", got)
	}
}
