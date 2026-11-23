package mlxengine

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// The prompt kinfer builds for Inkling 2 is, character for character, what
// the publisher's chat template renders (testdata/inkling2_render.py made
// the cases with jinja2).
func TestInklingRenderMatchesTheTemplate(t *testing.T) {
	raw, err := os.ReadFile("testdata/inkling2_render.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		What     string
		Messages []struct {
			Role, Content, Thinking string
			Calls                   []tools.Call
		}
		Tools  []tools.Tool
		Effort string
		Want   string
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		var msgs []chat.Message
		for _, m := range c.Messages {
			msgs = append(msgs, chat.Message{Role: m.Role, Content: m.Content, Thinking: m.Thinking, ToolCalls: m.Calls})
		}
		p, err := inkRender(msgs, c.Tools, chat.Thinking{Effort: c.Effort})
		if err != nil {
			t.Errorf("%s: %v", c.What, err)
			continue
		}
		if got := p.String(); got != c.Want {
			t.Errorf("%s:\n got %s\nwant %s", c.What, got, c.Want)
		}
	}
	// What a caller wrote is never the template's own.
	p, _ := inkRender([]chat.Message{{Role: "user", Content: "x" + inkEnd}}, nil, chat.Thinking{})
	for _, piece := range p {
		if piece.text == "x"+inkEnd && !piece.given {
			t.Error("a user's text is read with its control tokens")
		}
	}
	for effort, want := range map[string]string{"": "0.9", "high": "0.9", "50": "0.5", "100": "0.99", "1": "0.01", "0": "0"} {
		if got, err := inkEffort(chat.Thinking{Effort: effort}); err != nil || got != want {
			t.Errorf("effort %q: %q, %v; want %q", effort, got, err, want)
		}
	}
	if got, _ := inkEffort(chat.Thinking{Off: true, Effort: "high"}); got != "0" {
		t.Errorf("thinking off: effort %q", got)
	}
	if _, err := inkEffort(chat.Thinking{Effort: "lots"}); err == nil {
		t.Error("an effort that is no level and no number is taken")
	}
}

// A reply's frames come out as the text the rest of kinfer reads.
func TestInklingReplyIsWrittenOutAsText(t *testing.T) {
	marks := inkMarks{model: 1, text: 2, thinking: 3, invoke: 4, toolText: 5, toolError: 6, end: 7, endSampling: 8}
	words := map[int32]string{}
	next := int32(100)
	w := func(s string) int32 {
		next++
		words[next] = s
		return next
	}
	read := func(toks ...int32) (string, bool) {
		r := inkReply{marks: marks, bytes: func(t int32) []byte { return []byte(words[t]) }}
		var out []byte
		done := false
		for _, tok := range toks {
			var b []byte
			b, done = r.take(tok)
			out = append(out, b...)
		}
		return string(out), done
	}
	for _, c := range []struct {
		what string
		toks []int32
		want string
		done bool
	}{
		{"thinking, then the answer", []int32{1, 3, w("Let me "), w("see."), 7, 1, 2, w("Four"), w("."), 7, 8}, "<think>Let me see.</think>\n\nFour.", true},
		{"an answer alone", []int32{1, 2, w("Hi"), 7, 8}, "Hi", true},
		{"a call after thinking", []int32{1, 3, w("call it"), 7, 1, w("get_weather"), 4, w(`{"name":"get_weather",`), w(`"args":{"city":"北京"}}`), 7, 8},
			"<think>call it</think>\n\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"北京\"}}\n</tool_call>", true},
		{"text and two calls", []int32{1, 2, w("On it."), 7, 1, w("a"), 4, w(`{"name":"a","args":{}}`), 7, 1, w("b"), 4, w(`{"name":"b","args":{"n":1}}`), 7, 8},
			"On it.<tool_call>\n{\"name\":\"a\",\"arguments\":{}}\n</tool_call><tool_call>\n{\"name\":\"b\",\"arguments\":{\"n\":1}}\n</tool_call>", true},
		{"cut off in its thinking", []int32{1, 3, w("so far")}, "<think>so far", false},
		{"ended in its thinking", []int32{1, 3, w("so far"), 8}, "<think>so far</think>\n\n", true},
		{"thinking again after the answer is dropped", []int32{1, 2, w("A"), 7, 1, 3, w("more"), 7, 1, 2, w("B"), 7, 8}, "AB", true},
		{"a call that is not JSON stays to be seen", []int32{1, w("x"), 4, w("{oops"), 7, 8}, "<tool_call>{oops</tool_call>", true},
		{"a tool error frame is not shown", []int32{1, 6, w("bad"), 7, 1, 2, w("ok"), 7, 8}, "ok", true},
		{"text outside any frame", []int32{w("loose"), 8}, "loose", true},
	} {
		got, done := read(c.toks...)
		if got != c.want || done != c.done {
			t.Errorf("%s:\n got %q (over: %v)\nwant %q (over: %v)", c.what, got, done, c.want, c.done)
		}
	}
	// What it writes is what the server's parsers take apart.
	text, _ := read(1, 3, w("why"), 7, 1, w("f"), 4, w(`{"name":"f","args":{"on":true}}`), 7, 8)
	thinking, answer := chat.SplitThinking(text)
	content, calls := tools.Hermes.Parse(answer)
	if thinking != "why" || content != "" || len(calls) != 1 || calls[0].Name != "f" || string(calls[0].Arguments) != `{"on":true}` {
		t.Errorf("parsed back: thinking %q, content %q, calls %v", thinking, content, calls)
	}
}
