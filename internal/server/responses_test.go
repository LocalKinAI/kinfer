package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// Codex drives a turn from the event stream: it needs every item opened and
// closed in order, the text done event carrying the whole text, and
// response.completed carrying the output and the usage. The request is shaped
// like the one Codex was captured sending, namespace and hosted tools included.
func TestResponsesStreamsReasoningTextAndCallsInOrder(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	srv.open = func(p string, _ engine.Options) (generator, error) {
		return &fakeEngine{path: p, tools: true, promptTokens: 42, frags: thinkProseCall}, nil
	}
	body := `{"model":"alpha","stream":true,"store":false,"tool_choice":"auto","parallel_tool_calls":true,
	  "reasoning":{"summary":"auto"},"include":["reasoning.encrypted_content"],"prompt_cache_key":"k",
	  "client_metadata":{"x":"y"},"instructions":"You are Codex.",
	  "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in Berlin?"}]}],
	  "tools":[{"type":"function","name":"get_weather","description":"Weather","strict":false,
	            "parameters":{"type":"object","properties":{"city":{"type":"string"}}}},
	           {"type":"namespace","name":"multi_agent_v1","description":"x","tools":[]},
	           {"type":"web_search","external_web_access":true}]}`
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}

	events := recordedEvents(t, w.Body.String())
	want := []string{
		"response.created", "response.in_progress",
		"response.output_item.added", "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done", "response.output_item.done",
		"response.output_item.added", "response.content_part.added", "response.output_text.delta", "response.output_text.done", "response.content_part.done", "response.output_item.done",
		"response.output_item.added", "response.function_call_arguments.delta", "response.function_call_arguments.done", "response.output_item.done",
		"response.completed",
	}
	if got := compactNames(events); !reflect.DeepEqual(got, want) {
		t.Fatalf("events:\n got  %v\n want %v", got, want)
	}

	var doneText, args string
	for i, ev := range events {
		if ev.Name != ev.Data["type"] {
			t.Errorf("event %q carries type %v", ev.Name, ev.Data["type"])
		}
		if ev.Data["sequence_number"] != float64(i) {
			t.Errorf("event %d (%s) has sequence_number %v", i, ev.Name, ev.Data["sequence_number"])
		}
		switch ev.Name {
		case "response.output_text.done":
			doneText, _ = ev.Data["text"].(string)
		case "response.function_call_arguments.done":
			args, _ = ev.Data["arguments"].(string)
		}
	}
	if strings.TrimSpace(doneText) != "Let me look." || strings.Contains(doneText, "tool_c") {
		t.Errorf("output_text.done = %q", doneText)
	}
	var input map[string]string
	if json.Unmarshal([]byte(args), &input) != nil || input["city"] != "Berlin" {
		t.Errorf("arguments = %q", args)
	}

	completed := events[len(events)-1].Data["response"].(map[string]any)
	var kinds []string
	for _, o := range completed["output"].([]any) {
		kinds = append(kinds, o.(map[string]any)["type"].(string))
	}
	if !reflect.DeepEqual(kinds, []string{"reasoning", "message", "function_call"}) || completed["status"] != "completed" {
		t.Errorf("completed = %v %v", completed["status"], kinds)
	}
	if u := completed["usage"].(map[string]any); u["input_tokens"] != float64(42) {
		t.Errorf("usage = %v; want 42 input tokens", u)
	}
}

// The second request of a captured Codex turn: instructions, a developer
// message, the user's, then function_call, reasoning, function_call_output —
// in that order, reasoning between a call and its result.
func TestResponsesInputBecomesTemplateTurns(t *testing.T) {
	raw := `{"model":"alpha","instructions":"base prompt",
	  "input":[
	    {"type":"message","role":"developer","content":[{"type":"input_text","text":"sandbox rules"},{"type":"input_text","text":"more rules"}]},
	    {"type":"message","role":"user","content":[{"type":"input_text","text":"list files"}]},
	    {"type":"function_call","name":"exec_command","call_id":"functions.exec_command:0","arguments":"{\"cmd\":\"ls -la\"}"},
	    {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"use ls"}],"encrypted_content":"opaque"},
	    {"type":"function_call_output","call_id":"functions.exec_command:0","output":"a.txt\nb.txt"}
	  ],
	  "tools":[{"type":"function","name":"exec_command","parameters":{"type":"object"}},
	           {"type":"namespace","name":"multi_agent_v1","tools":[]},{"type":"web_search"}]}`
	var req responsesRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}
	msgs, err := responsesToChat(&req)
	if err != nil {
		t.Fatal(err)
	}
	want := []struct{ role, content, call string }{
		{"system", "base prompt", ""},
		{"system", "sandbox rules\nmore rules", ""},
		{"user", "list files", ""},
		{"assistant", "", "exec_command"},
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
	if tools := responsesTools(req.Tools); len(tools) != 1 || tools[0].Function.Name != "exec_command" {
		t.Errorf("tools = %+v; want only the function tool", tools)
	}
}

func TestResponsesStringInputNonStreaming(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	srv.open = func(p string, _ engine.Options) (generator, error) {
		return &fakeEngine{path: p, promptTokens: 3, frags: []string{"Hello", " there"}}, nil
	}
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"alpha","input":"hi"}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var got struct {
		Object, Status string
		Output         []struct {
			Type    string
			Content []struct{ Type, Text string }
		}
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		}
	}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Object != "response" || got.Status != "completed" || len(got.Output) != 1 ||
		got.Output[0].Type != "message" || got.Output[0].Content[0].Text != "Hello there" {
		t.Errorf("response = %+v", got)
	}
	if got.Usage.InputTokens != 3 || got.Usage.OutputTokens != 2 {
		t.Errorf("usage = %+v", got.Usage)
	}
}

// A reply a limit cut off ends in response.incomplete with the reason, not in
// response.completed: an agent that takes a truncated turn for a finished one
// acts on half an answer.
func TestResponsesTruncatedReplyIsIncomplete(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	srv.open = func(p string, _ engine.Options) (generator, error) {
		return &fakeEngine{path: p, truncated: true, frags: []string{"half an"}}, nil
	}
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"alpha","input":"hi","stream":true}`)))
	events := recordedEvents(t, w.Body.String())
	last := events[len(events)-1]
	resp, _ := last.Data["response"].(map[string]any)
	if last.Name != "response.incomplete" || resp["status"] != "incomplete" {
		t.Errorf("last event = %s %v; want response.incomplete", last.Name, resp["status"])
	}
}
