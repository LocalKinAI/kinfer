package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// thinkProseCall is a reply with all three things an agent's turn can hold:
// thinking, prose, and a tool call — split across fragments the way a model
// actually emits them, marker halves included.
var thinkProseCall = []string{
	"<think>", "which city", "</think>", "\n\nLet me ", "look.", " <tool_c",
	"all>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Berlin\"}}\n</tool_call>",
}

// recordedEvent is one server-sent event out of a recorded stream.
type recordedEvent struct {
	Name string
	Data map[string]any
}

func recordedEvents(t *testing.T, body string) []recordedEvent {
	t.Helper()
	var out []recordedEvent
	for _, chunk := range strings.Split(strings.TrimSpace(body), "\n\n") {
		var ev recordedEvent
		for _, line := range strings.Split(chunk, "\n") {
			switch {
			case strings.HasPrefix(line, "event: "):
				ev.Name = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev.Data); err != nil {
					t.Fatalf("bad data line %q: %v", line, err)
				}
			}
		}
		out = append(out, ev)
	}
	return out
}

// compactNames lists the event names with runs of the same name collapsed, so
// a test can state an order without depending on how many deltas there were.
func compactNames(events []recordedEvent) []string {
	var out []string
	for _, ev := range events {
		if len(out) == 0 || out[len(out)-1] != ev.Name {
			out = append(out, ev.Name)
		}
	}
	return out
}

// Claude Code sends a great deal the reference does not describe — a system
// array with cache_control, adaptive thinking, output_config,
// context_management, metadata, ?beta=true on the path — and must get back
// thinking, prose and the call, each as its own block, with no call syntax in
// the text.
func TestAnthropicStreamsThinkingProseAndToolUse(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	srv.open = func(p string, _ engine.Options) (generator, error) {
		return &fakeEngine{path: p, tools: true, promptTokens: 42, frags: thinkProseCall}, nil
	}
	body := `{"model":"alpha","max_tokens":32000,"stream":true,
	  "system":[{"type":"text","text":"You are Claude Code."},{"type":"text","text":"Be brief.","cache_control":{"type":"ephemeral"}}],
	  "thinking":{"type":"adaptive","display":"omitted"},
	  "output_config":{"effort":"high"},
	  "context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]},
	  "metadata":{"user_id":"x"},
	  "messages":[{"role":"user","content":[{"type":"text","text":"weather in Berlin?"}]}],
	  "tools":[{"name":"get_weather","description":"Weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}]}`
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages?beta=true", strings.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}

	events := recordedEvents(t, w.Body.String())
	names := compactNames(events)
	if names[0] != "message_start" || names[len(names)-1] != "message_stop" {
		t.Fatalf("stream must open with message_start and close with message_stop: %v", names)
	}
	var thinking, prose, toolName, toolInput, stop string
	blocks := map[float64]string{}
	for _, ev := range events {
		if ev.Name != ev.Data["type"] {
			t.Errorf("event %q carries type %v", ev.Name, ev.Data["type"])
		}
		switch ev.Name {
		case "content_block_start":
			cb := ev.Data["content_block"].(map[string]any)
			blocks[ev.Data["index"].(float64)] = cb["type"].(string)
			if cb["type"] == "tool_use" {
				toolName, _ = cb["name"].(string)
			}
		case "content_block_delta":
			d := ev.Data["delta"].(map[string]any)
			switch d["type"] {
			case "thinking_delta":
				thinking += d["thinking"].(string)
			case "text_delta":
				prose += d["text"].(string)
			case "input_json_delta":
				toolInput += d["partial_json"].(string)
			}
		case "message_delta":
			stop, _ = ev.Data["delta"].(map[string]any)["stop_reason"].(string)
			if u := ev.Data["usage"].(map[string]any); u["input_tokens"] != float64(42) {
				t.Errorf("usage = %v; want the prompt's 42 input tokens", u)
			}
		}
	}
	if thinking != "which city" {
		t.Errorf("thinking = %q", thinking)
	}
	if strings.TrimSpace(prose) != "Let me look." || strings.HasPrefix(prose, "\n") {
		t.Errorf("prose = %q; want it streamed without the template's leading newlines", prose)
	}
	if strings.Contains(prose, "tool_c") {
		t.Errorf("call syntax leaked into a text block: %q", prose)
	}
	var input map[string]string
	if toolName != "get_weather" || json.Unmarshal([]byte(toolInput), &input) != nil || input["city"] != "Berlin" {
		t.Errorf("tool_use = %q %q; want get_weather {city: Berlin}", toolName, toolInput)
	}
	if stop != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", stop)
	}
	if blocks[0] != "thinking" || blocks[1] != "text" || blocks[2] != "tool_use" {
		t.Errorf("blocks = %v; want thinking, text, tool_use in that order", blocks)
	}
}

// The third turn of a captured Claude Code conversation: a system array, a
// user turn of two text blocks, a system turn inside messages, an assistant
// turn of thinking plus tool_use, and the tool_result that answers it.
func TestAnthropicHistoryBecomesTemplateTurns(t *testing.T) {
	raw := `{"model":"alpha",
	  "system":[{"type":"text","text":"base prompt"}],
	  "messages":[
	    {"role":"user","content":[{"type":"text","text":"list files"},{"type":"text","text":"in here"}]},
	    {"role":"system","content":"a reminder"},
	    {"role":"assistant","content":[{"type":"thinking","thinking":"use ls"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}]},
	    {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"a.txt\nb.txt","cache_control":{"type":"ephemeral"}}]}
	  ]}`
	var req anthropicRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}
	msgs, err := anthropicToChat(&req)
	if err != nil {
		t.Fatal(err)
	}
	want := []struct{ role, content, call string }{
		{"system", "base prompt", ""},
		{"user", "list files\nin here", ""},
		{"system", "a reminder", ""},
		{"assistant", "", "Bash"},
		{"tool", "a.txt\nb.txt", ""},
	}
	if len(msgs) != len(want) {
		t.Fatalf("got %d turns %+v, want %d", len(msgs), msgs, len(want))
	}
	for i, w := range want {
		m, call := msgs[i], ""
		if len(m.ToolCalls) > 0 {
			call = m.ToolCalls[0].Name
		}
		if m.Role != w.role || m.Content != w.content || call != w.call {
			t.Errorf("turn %d = %s %q call=%q; want %s %q call=%q", i, m.Role, m.Content, call, w.role, w.content, w.call)
		}
	}
	if got := string(msgs[3].ToolCalls[0].Arguments); got != `{"command":"ls"}` {
		t.Errorf("arguments = %s", got)
	}
}

func TestAnthropicNonStreamingReply(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	srv.open = func(p string, _ engine.Options) (generator, error) {
		return &fakeEngine{path: p, promptTokens: 7, frags: []string{"Hello", " there"}}, nil
	}
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
		`{"model":"alpha","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var got struct {
		Type       string `json:"type"`
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type, Text string
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Type != "message" || got.StopReason != "end_turn" || len(got.Content) != 1 ||
		got.Content[0].Type != "text" || got.Content[0].Text != "Hello there" {
		t.Errorf("reply = %+v", got)
	}
	if got.Usage.InputTokens != 7 || got.Usage.OutputTokens != 2 {
		t.Errorf("usage = %+v; want 7 in, 2 out", got.Usage)
	}
}

// Errors come in this dialect's envelope with the status the other dialects
// give the same failure — a streamed request included, when it failed before
// its first token and a status could still be sent.
func TestAnthropicErrorsUseItsEnvelope(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	srv.open = func(p string, _ engine.Options) (generator, error) {
		return &fakeEngine{path: p, err: &engine.PromptTooLongError{Tokens: 40000, Limit: 32768, Slots: 4}}, nil
	}
	cases := []struct {
		body   string
		status int
		kind   string
	}{
		{`{"model":"alpha","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`, http.StatusRequestEntityTooLarge, "request_too_large"},
		{`{"model":"alpha","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`, http.StatusRequestEntityTooLarge, "request_too_large"},
		{`{"model":"nope","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`, http.StatusNotFound, "not_found_error"},
		{`{"model":`, http.StatusBadRequest, "invalid_request_error"},
	}
	for _, c := range cases {
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(c.body)))
		var env struct {
			Type  string `json:"type"`
			Error struct {
				Type string `json:"type"`
			} `json:"error"`
		}
		_ = json.Unmarshal(w.Body.Bytes(), &env)
		if w.Code != c.status || env.Type != "error" || env.Error.Type != c.kind {
			t.Errorf("%s\n  → %d %s; want %d %s", c.body, w.Code, w.Body.String(), c.status, c.kind)
		}
	}
}

// Declaring tools to a model whose template has no call convention is refused,
// as in the OpenAI dialect, rather than answered with prose where a call was
// expected.
func TestAnthropicToolsNeedAToolTemplate(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
		`{"model":"alpha","max_tokens":10,"messages":[{"role":"user","content":"hi"}],
		  "tools":[{"name":"f","input_schema":{"type":"object"}}]}`)))
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "tool_call") {
		t.Errorf("status %d %s; want 400 naming the missing convention", w.Code, w.Body.String())
	}
}
