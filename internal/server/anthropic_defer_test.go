package server

import (
	"encoding/json"
	"strings"
	"testing"
)

// A deferred tool is kept from the model until a tool search's result names
// it or the model has called it; the search's result reads as the definitions
// of what it found.
func TestAnthropicDeferredTools(t *testing.T) {
	const body = `{"model":"m","max_tokens":10,
	  "tools":[
	    {"name":"Read","description":"read a file","input_schema":{"type":"object"}},
	    {"name":"ToolSearch","description":"find tools","input_schema":{"type":"object"}},
	    {"name":"mcp__mail__send","description":"send a mail","defer_loading":true,"input_schema":{"type":"object","properties":{"to":{"type":"string"}}}},
	    {"name":"mcp__mail__list","description":"list mail","defer_loading":true,"input_schema":{"type":"object"}},
	    {"name":"mcp__cal__add","description":"add an event","defer_loading":true,"input_schema":{"type":"object"}}],
	  "messages":[%s]}`
	names := func(msgs string) (string, []string) {
		var req anthropicRequest
		if err := json.Unmarshal([]byte(strings.Replace(body, "%s", msgs, 1)), &req); err != nil {
			t.Fatal(err)
		}
		chat, err := anthropicToChat(&req)
		if err != nil {
			t.Fatal(err)
		}
		var got []string
		for _, tl := range anthropicTools(req.Tools, loadedTools(&req)) {
			got = append(got, tl.Function.Name)
		}
		return chat[len(chat)-1].Content, got
	}

	_, got := names(`{"role":"user","content":"send a mail"}`)
	if strings.Join(got, ",") != "Read,ToolSearch" {
		t.Errorf("before any search the model is shown %v", got)
	}

	found := `{"role":"user","content":"send a mail"},
	  {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"ToolSearch","input":{"query":"mail"}}]},
	  {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[
	     {"type":"tool_reference","tool_name":"mcp__mail__send"},{"type":"tool_reference","tool_name":"mcp__nobody__knows"}]}]}`
	last, got := names(found)
	if strings.Join(got, ",") != "Read,ToolSearch,mcp__mail__send" {
		t.Errorf("after the search the model is shown %v", got)
	}
	if !strings.Contains(last, `"name":"mcp__mail__send"`) || !strings.Contains(last, `"to":{"type":"string"}`) || strings.Contains(last, "nobody") {
		t.Errorf("the search's result reads %q", last)
	}

	// A tool the model has called stays, though the turn that found it is gone
	// (a compacted conversation).
	_, got = names(`{"role":"user","content":"again"},
	  {"role":"assistant","content":[{"type":"tool_use","id":"t2","name":"mcp__cal__add","input":{}}]},
	  {"role":"user","content":[{"type":"tool_result","tool_use_id":"t2","content":"added"}]}`)
	if strings.Join(got, ",") != "Read,ToolSearch,mcp__cal__add" {
		t.Errorf("a tool already called: %v", got)
	}

	// Anthropic's own tool search runs on Anthropic's side: asked for here,
	// nothing would ever load a tool, so every tool is shown.
	var req anthropicRequest
	json.Unmarshal([]byte(`{"tools":[{"type":"tool_search_tool_regex_20251119","name":"tool_search_tool_regex"},
	  {"name":"a","defer_loading":true,"input_schema":{"type":"object"}}],"messages":[]}`), &req)
	if ts := anthropicTools(req.Tools, loadedTools(&req)); len(ts) != 1 || ts[0].Function.Name != "a" {
		t.Errorf("with Anthropic's own search asked for: %v", ts)
	}
}
