package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// ─── OpenAI Responses dialect ────────────────────────────────────────────────
//
// POST /v1/responses: the only API Codex speaks to a custom provider since
// 0.154 refused `wire_api = "chat"`. Written against what Codex was captured
// sending, and shaped like what it was seen accepting (Ollama's /v1/responses):
//
//   - `instructions` is the whole ~17K-character base prompt, and a "developer"
//     message follows it in `input`;
//   - history arrives as items in the order function_call, reasoning,
//     function_call_output — a call's result is not always the item after it;
//   - the tools include a "namespace" group and a hosted "web_search" alongside
//     the plain function tools;
//   - store is false and every turn resends the whole conversation, so
//     previous_response_id never has to be honoured.

type responsesRequest struct {
	Model           string          `json:"model"`
	Instructions    string          `json:"instructions"`
	Input           json.RawMessage `json:"input"`
	Tools           []responsesTool `json:"tools"`
	ToolChoice      json.RawMessage `json:"tool_choice"`
	Stream          bool            `json:"stream"`
	MaxOutputTokens *int            `json:"max_output_tokens"`
	Temperature     *float64        `json:"temperature"`
	TopP            *float64        `json:"top_p"`

	Reasoning *struct {
		Effort string `json:"effort"`
	} `json:"reasoning"`
}

type responsesTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// responsesItem is every input item kind this server reads, flattened.
type responsesItem struct {
	Type      string          `json:"type"`
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	Name      string          `json:"name"`
	Arguments string          `json:"arguments"`
	Input     string          `json:"input"`
	Output    json.RawMessage `json:"output"`
}

// responsesText reads content that is either a string or an array of parts.
func responsesText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", err
	}
	var out []string
	for _, p := range parts {
		switch p.Type {
		case "input_text", "output_text", "text":
			out = append(out, p.Text)
		case "input_image":
			out = append(out, omittedAttachment("image"))
		case "input_file":
			out = append(out, omittedAttachment("file"))
		}
	}
	return strings.Join(out, "\n"), nil
}

// responsesToChat flattens a Responses request into the turns a chat template
// renders.
func responsesToChat(req *responsesRequest) ([]chat.Message, error) {
	var out []chat.Message
	if req.Instructions != "" {
		out = append(out, chat.Message{Role: "system", Content: req.Instructions})
	}
	if len(req.Input) == 0 {
		return out, nil
	}
	var s string
	if json.Unmarshal(req.Input, &s) == nil {
		return append(out, chat.Message{Role: "user", Content: s}), nil
	}
	var items []responsesItem
	if err := json.Unmarshal(req.Input, &items); err != nil {
		return nil, fmt.Errorf("input: %w", err)
	}

	for i, it := range items {
		kind := it.Type
		if kind == "" && it.Role != "" {
			kind = "message" // the shorthand form: {role, content}, no type
		}
		switch kind {
		case "message":
			text, err := responsesText(it.Content)
			if err != nil {
				return nil, fmt.Errorf("input[%d]: %w", i, err)
			}
			role := "user"
			switch it.Role {
			case "system", "developer":
				role = "system"
			case "assistant":
				role = "assistant"
			}
			out = append(out, chat.Message{Role: role, Content: text})

		case "function_call", "custom_tool_call":
			args := json.RawMessage(it.Arguments)
			if kind == "custom_tool_call" {
				args, _ = json.Marshal(map[string]string{"input": it.Input})
			}
			call := tools.Call{Name: it.Name, Arguments: jsonObject(args)}
			// A call belongs to the assistant turn that made it: the last turn
			// so far, when that was the assistant's — a reasoning item between
			// the two is dropped, not made a turn.
			if n := len(out); n > 0 && out[n-1].Role == "assistant" {
				out[n-1].ToolCalls = append(out[n-1].ToolCalls, call)
			} else {
				out = append(out, chat.Message{Role: "assistant", ToolCalls: []tools.Call{call}})
			}

		case "function_call_output", "custom_tool_call_output":
			text, err := responsesText(it.Output)
			if err != nil {
				return nil, fmt.Errorf("input[%d]: %w", i, err)
			}
			out = append(out, chat.Message{Role: "tool", Content: text})
		}
		// A reasoning item is the model's earlier working, encrypted for a
		// server that is not this one; web_search_call and its kind record
		// hosted tools that never ran here. Neither is a turn.
	}
	return out, nil
}

// responsesTools keeps the plain function tools. A "namespace" groups tools
// under a name the model would have to be taught to call through, a "custom"
// tool takes freeform input under a grammar, and web_search and file_search run
// on OpenAI's side. Declared to a local model, each only invites a call nothing
// here can answer.
func responsesTools(in []responsesTool) []tools.Tool {
	var out []tools.Tool
	for _, t := range in {
		if t.Type != "function" || t.Name == "" {
			continue
		}
		params := t.Parameters
		if len(params) == 0 || string(params) == "null" {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out = append(out, tools.Tool{Type: "function", Function: tools.Function{
			Name: t.Name, Description: t.Description, Parameters: params,
		}})
	}
	return out
}

func responsesUsage(st engine.Stats) map[string]any {
	return map[string]any{
		"input_tokens":          st.PromptTokens,
		"input_tokens_details":  map[string]any{"cached_tokens": 0},
		"output_tokens":         st.EvalTokens,
		"output_tokens_details": map[string]any{"reasoning_tokens": 0},
		"total_tokens":          st.PromptTokens + st.EvalTokens,
	}
}

// responsesStatus is "incomplete", with the reason, for a reply a limit cut off,
// and "completed" otherwise.
func responsesStatus(st engine.Stats) (string, any) {
	if st.Truncated || st.Timeout {
		return "incomplete", map[string]string{"reason": "max_output_tokens"}
	}
	return "completed", nil
}

func responseObject(id, model string, created int64, status string, output []any, usage, incomplete any) map[string]any {
	return map[string]any{
		"id": id, "object": "response", "created_at": created, "status": status, "model": model,
		"output": output, "usage": usage, "incomplete_details": incomplete, "error": nil,
	}
}

func outputTextPart(text string) map[string]any {
	return map[string]any{"type": "output_text", "text": text, "annotations": []any{}, "logprobs": []any{}}
}

func messageItem(id, text, status string) map[string]any {
	content := []any{}
	if status == "completed" {
		content = append(content, outputTextPart(text))
	}
	return map[string]any{"type": "message", "id": id, "status": status, "role": "assistant", "content": content}
}

func reasoningItem(id, text string) map[string]any {
	summary := []any{}
	if text != "" {
		summary = append(summary, map[string]any{"type": "summary_text", "text": text})
	}
	return map[string]any{"type": "reasoning", "id": id, "summary": summary}
}

func functionCallItem(id, callID, name, args, status string) map[string]any {
	return map[string]any{
		"type": "function_call", "id": id, "call_id": callID, "name": name, "arguments": args, "status": status,
	}
}

func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req responsesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{
			"message": err.Error(), "type": "invalid_request_error"}})
		return
	}
	msgs, err := responsesToChat(&req)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{
			"message": err.Error(), "type": "invalid_request_error"}})
		return
	}

	eng, release, err := s.acquire(req.Model)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]any{"error": map[string]string{
			"message": err.Error(), "type": "invalid_request_error"}})
		return
	}
	defer release()

	params := engine.DefaultGenParams()
	if !toolChoiceNone(req.ToolChoice) {
		params.Tools = responsesTools(req.Tools)
	}
	params.NoThink = req.Reasoning != nil && req.Reasoning.Effort == "none"
	if req.Temperature != nil {
		params.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = float32(*req.TopP)
	}
	if req.MaxOutputTokens != nil {
		params.MaxTokens = *req.MaxOutputTokens
	}
	if len(params.Tools) > 0 && eng.ToolFormat() == tools.None {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{
			"message": "this model's chat template does not use the <tool_call> convention, so it cannot answer with tool calls",
			"type":    "invalid_request_error"}})
		return
	}

	id := newID("resp")
	created := time.Now().Unix()

	if !req.Stream {
		rep, err := generate(r.Context(), eng, msgs, params, nil, nil)
		if err != nil {
			writeOpenAIGenError(w, err)
			return
		}
		output := []any{}
		if rep.Thinking != "" {
			output = append(output, reasoningItem(newID("rs"), rep.Thinking))
		}
		if text := strings.TrimSpace(rep.Content); text != "" {
			output = append(output, messageItem(newID("msg"), text, "completed"))
		}
		for _, c := range rep.Calls {
			output = append(output, functionCallItem(newID("fc"), newID("call"), c.Name, string(jsonObject(c.Arguments)), "completed"))
		}
		status, incomplete := responsesStatus(rep.Stats)
		writeJSON(w, http.StatusOK, responseObject(id, req.Model, created, status, output, responsesUsage(rep.Stats), incomplete))
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]string{"message": "streaming unsupported"}})
		return
	}
	seq := 0
	send := func(data map[string]any) {
		data["sequence_number"] = seq
		seq++
		payload, _ := json.Marshal(data)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", data["type"], payload)
		flusher.Flush()
	}

	output := []any{}
	streamed := false
	begin := func() {
		if streamed {
			return
		}
		streamed = true
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		send(map[string]any{"type": "response.created", "response": responseObject(id, req.Model, created, "in_progress", []any{}, nil, nil)})
		send(map[string]any{"type": "response.in_progress", "response": responseObject(id, req.Model, created, "in_progress", []any{}, nil, nil)})
	}

	outputIndex := -1
	open := "" // the item being streamed: "reasoning", "message", or none
	itemID := ""
	var text strings.Builder
	closeItem := func() {
		switch open {
		case "reasoning":
			send(map[string]any{"type": "response.reasoning_summary_text.done",
				"item_id": itemID, "output_index": outputIndex, "summary_index": 0, "text": text.String()})
			item := reasoningItem(itemID, text.String())
			send(map[string]any{"type": "response.output_item.done", "output_index": outputIndex, "item": item})
			output = append(output, item)
		case "message":
			send(map[string]any{"type": "response.output_text.done",
				"item_id": itemID, "output_index": outputIndex, "content_index": 0, "text": text.String(), "logprobs": []any{}})
			send(map[string]any{"type": "response.content_part.done",
				"item_id": itemID, "output_index": outputIndex, "content_index": 0, "part": outputTextPart(text.String())})
			item := messageItem(itemID, text.String(), "completed")
			send(map[string]any{"type": "response.output_item.done", "output_index": outputIndex, "item": item})
			output = append(output, item)
		}
		open = ""
		text.Reset()
	}

	onThinking := func(t string) {
		begin()
		if open != "reasoning" {
			closeItem()
			outputIndex++
			itemID = newID("rs")
			open = "reasoning"
			send(map[string]any{"type": "response.output_item.added", "output_index": outputIndex, "item": reasoningItem(itemID, "")})
		}
		text.WriteString(t)
		send(map[string]any{"type": "response.reasoning_summary_text.delta",
			"item_id": itemID, "output_index": outputIndex, "summary_index": 0, "delta": t})
	}
	onContent := func(t string) {
		if open != "message" {
			// The newlines a template leaves after a think block are not the
			// start of anything the model said.
			t = strings.TrimLeft(t, " \t\r\n")
			if t == "" {
				return
			}
			begin()
			closeItem()
			outputIndex++
			itemID = newID("msg")
			open = "message"
			send(map[string]any{"type": "response.output_item.added", "output_index": outputIndex, "item": messageItem(itemID, "", "in_progress")})
			send(map[string]any{"type": "response.content_part.added",
				"item_id": itemID, "output_index": outputIndex, "content_index": 0, "part": outputTextPart("")})
		}
		text.WriteString(t)
		send(map[string]any{"type": "response.output_text.delta",
			"item_id": itemID, "output_index": outputIndex, "content_index": 0, "delta": t, "logprobs": []any{}})
	}

	rep, err := generate(r.Context(), eng, msgs, params, onThinking, onContent)
	if err != nil {
		if !streamed {
			writeOpenAIGenError(w, err)
			return
		}
		// The header is gone; response.failed is this dialect's way of saying
		// the reply did not end, rather than closing as though it had.
		log.Printf("generation failed: %v", err)
		failed := responseObject(id, req.Model, created, "failed", output, nil, nil)
		failed["error"] = map[string]string{"code": "server_error", "message": err.Error()}
		send(map[string]any{"type": "response.failed", "response": failed})
		return
	}

	begin()
	closeItem()
	for _, c := range rep.Calls {
		// A call is only known once it is complete, so its arguments go out in
		// one delta.
		outputIndex++
		fid, callID, args := newID("fc"), newID("call"), string(jsonObject(c.Arguments))
		send(map[string]any{"type": "response.output_item.added", "output_index": outputIndex,
			"item": functionCallItem(fid, callID, c.Name, "", "in_progress")})
		send(map[string]any{"type": "response.function_call_arguments.delta", "item_id": fid, "output_index": outputIndex, "delta": args})
		send(map[string]any{"type": "response.function_call_arguments.done", "item_id": fid, "output_index": outputIndex, "arguments": args})
		item := functionCallItem(fid, callID, c.Name, args, "completed")
		send(map[string]any{"type": "response.output_item.done", "output_index": outputIndex, "item": item})
		output = append(output, item)
	}

	status, incomplete := responsesStatus(rep.Stats)
	final := "response.completed"
	if status == "incomplete" {
		final = "response.incomplete"
	}
	send(map[string]any{"type": final,
		"response": responseObject(id, req.Model, created, status, output, responsesUsage(rep.Stats), incomplete)})
}
