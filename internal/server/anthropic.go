package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"strconv"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// ─── Anthropic dialect ───────────────────────────────────────────────────────
//
// POST /v1/messages: what Claude Code speaks to a server named in
// ANTHROPIC_BASE_URL. Written against what Claude Code was captured sending,
// and shaped like what it was seen accepting (Ollama's /v1/messages, which it
// runs against unmodified), because both differ from the reference:
//
//   - the path arrives as /v1/messages?beta=true;
//   - `system` is an array of text blocks, and turns with role "system" appear
//     inside `messages` too, which the reference says cannot happen;
//   - thinking, output_config, context_management, metadata and cache_control
//     markers ride along on every turn, so none of them may be a 400;
//   - the stream carries no ping, and thinking blocks no signature.

type anthropicRequest struct {
	Model       string             `json:"model"`
	System      json.RawMessage    `json:"system"`
	Messages    []anthropicMessage `json:"messages"`
	Tools       []anthropicTool    `json:"tools"`
	ToolChoice  json.RawMessage    `json:"tool_choice"`
	MaxTokens   *int               `json:"max_tokens"`
	Temperature *float64           `json:"temperature"`
	TopP        *float64           `json:"top_p"`
	Stream      bool               `json:"stream"`

	Thinking *struct {
		Type string `json:"type"`
	} `json:"thinking"`

	// OutputConfig.Format is a JSON Schema the reply must match. Claude Code
	// asks for one when it names a session.
	OutputConfig *struct {
		Format *struct {
			Type   string          `json:"type"`
			Schema json.RawMessage `json:"schema"`
		} `json:"format"`
	} `json:"output_config"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// anthropicBlock is every content block kind this server reads, flattened.
type anthropicBlock struct {
	Type    string          `json:"type"`
	Text    string          `json:"text"`
	Name    string          `json:"name"`
	Input   json.RawMessage `json:"input"`
	Content json.RawMessage `json:"content"`
	IsError bool            `json:"is_error"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// anthropicBlocks reads content in either shape the API allows wherever content
// appears: a plain string, or an array of blocks.
func anthropicBlocks(raw json.RawMessage) ([]anthropicBlock, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return []anthropicBlock{{Type: "text", Text: s}}, nil
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, err
	}
	return blocks, nil
}

// anthropicText is the readable text of some content: text blocks joined,
// attachments named.
func anthropicText(raw json.RawMessage) (string, error) {
	blocks, err := anthropicBlocks(raw)
	if err != nil {
		return "", err
	}
	var parts []string
	for _, b := range blocks {
		switch b.Type {
		case "text":
			parts = append(parts, b.Text)
		case "image", "document":
			parts = append(parts, omittedAttachment(b.Type))
		}
	}
	return strings.Join(parts, "\n"), nil
}

// anthropicToChat flattens a Messages request into the turns a chat template
// renders. The two disagree about where a tool result lives: here it is a block
// inside a user turn, and a template wants it as a "tool" turn of its own — in
// order, ahead of any text that follows it.
func anthropicToChat(req *anthropicRequest) ([]chat.Message, error) {
	var out []chat.Message

	system, err := anthropicText(req.System)
	if err != nil {
		return nil, fmt.Errorf("system: %w", err)
	}
	if oc := req.OutputConfig; oc != nil && oc.Format != nil && oc.Format.Type == "json_schema" && len(oc.Format.Schema) > 0 {
		// Nothing here holds a model to a schema while it samples, so the
		// schema is stated instead. Claude Code uses this for a session title:
		// a model that ignores it costs a title, not a turn.
		system = strings.TrimSpace(system + "\n\nRespond with only a JSON object matching this JSON Schema, and nothing else:\n" + string(oc.Format.Schema))
	}
	if system != "" {
		out = append(out, chat.Message{Role: "system", Content: system})
	}

	for i, m := range req.Messages {
		blocks, err := anthropicBlocks(m.Content)
		if err != nil {
			return nil, fmt.Errorf("messages[%d]: %w", i, err)
		}

		if m.Role == "assistant" {
			msg := chat.Message{Role: "assistant"}
			var text []string
			for _, b := range blocks {
				switch b.Type {
				case "text":
					text = append(text, b.Text)
				case "tool_use":
					msg.ToolCalls = append(msg.ToolCalls, tools.Call{Name: b.Name, Arguments: jsonObject(b.Input)})
				}
				// thinking and redacted_thinking are the model's working, not
				// what it said, and a local template has nowhere to put them.
			}
			msg.Content = strings.Join(text, "\n")
			if msg.Content != "" || len(msg.ToolCalls) > 0 {
				out = append(out, msg)
			}
			continue
		}

		// "user" — and "system", as Claude Code sends it mid-conversation, which
		// stays a system turn where it was put.
		role := "user"
		if m.Role == "system" {
			role = "system"
		}
		var text []string
		flush := func() {
			if len(text) > 0 {
				out = append(out, chat.Message{Role: role, Content: strings.Join(text, "\n")})
				text = nil
			}
		}
		for _, b := range blocks {
			switch b.Type {
			case "text":
				text = append(text, b.Text)
			case "image", "document":
				text = append(text, omittedAttachment(b.Type))
			case "tool_result":
				flush()
				result, err := anthropicText(b.Content)
				if err != nil {
					return nil, fmt.Errorf("messages[%d]: tool_result: %w", i, err)
				}
				if b.IsError {
					result = "Error: " + result
				}
				out = append(out, chat.Message{Role: "tool", Content: result})
			}
		}
		flush()
	}
	return out, nil
}

// anthropicTools keeps the tools a local model can be asked to call. Server
// tools — web_search_…, code_execution_… — have no input schema and run on
// Anthropic's side; there is nothing here to run them.
func anthropicTools(in []anthropicTool) []tools.Tool {
	var out []tools.Tool
	for _, t := range in {
		if t.Name == "" || len(t.InputSchema) == 0 {
			continue
		}
		out = append(out, tools.Tool{Type: "function", Function: tools.Function{
			Name: t.Name, Description: t.Description, Parameters: t.InputSchema,
		}})
	}
	return out
}

// anthropicStopReason is why a reply ended, in this dialect's vocabulary. A
// timeout is "max_tokens" for the reason openAIFinishReason gives: the reply is
// incomplete, and a limit rather than the model cut it off.
func anthropicStopReason(r reply) string {
	switch {
	case len(r.Calls) > 0:
		return "tool_use"
	case r.Stats.Truncated || r.Stats.Timeout:
		return "max_tokens"
	default:
		return "end_turn"
	}
}

func anthropicUsage(st engine.Stats) map[string]any {
	return map[string]any{"input_tokens": st.PromptTokens, "output_tokens": st.EvalTokens}
}

func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req anthropicRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	msgs, err := anthropicToChat(&req)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	eng, release, err := s.acquire(req.Model)
	if err != nil {
		writeAnthropicError(w, http.StatusNotFound, "not_found_error", err.Error())
		return
	}
	defer release()

	params := engine.DefaultGenParams()
	if !toolChoiceNone(req.ToolChoice) {
		params.Tools = anthropicTools(req.Tools)
	}
	params.NoThink = req.Thinking != nil && req.Thinking.Type == "disabled"
	if req.Temperature != nil {
		params.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = float32(*req.TopP)
	}
	if req.MaxTokens != nil {
		params.MaxTokens = *req.MaxTokens
	}
	if len(params.Tools) > 0 && eng.ToolFormat() == tools.None {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error",
			"this model's chat template does not use the <tool_call> convention, so it cannot answer with tool calls")
		return
	}

	id := newID("msg")

	if !req.Stream {
		rep, err := generate(r.Context(), eng, msgs, params, nil, nil)
		if err != nil {
			writeAnthropicGenError(w, err)
			return
		}
		content := []any{}
		if rep.Thinking != "" {
			content = append(content, map[string]any{"type": "thinking", "thinking": rep.Thinking})
		}
		if text := strings.TrimSpace(rep.Content); text != "" {
			content = append(content, map[string]any{"type": "text", "text": text})
		}
		for _, c := range rep.Calls {
			content = append(content, map[string]any{
				"type": "tool_use", "id": newID("toolu"), "name": c.Name, "input": jsonObject(c.Arguments),
			})
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": id, "type": "message", "role": "assistant", "model": req.Model, "content": content,
			"stop_reason": anthropicStopReason(rep), "stop_sequence": nil, "usage": anthropicUsage(rep.Stats),
		})
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "streaming unsupported")
		return
	}
	send := func(data map[string]any) {
		payload, _ := json.Marshal(data)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", data["type"], payload)
		flusher.Flush()
	}

	// Nothing is written until there is something to write, as in the other
	// dialects, so a request that fails before its first token still gets a
	// status code rather than a 200 whose body says it failed.
	streamed := false
	begin := func() {
		if streamed {
			return
		}
		streamed = true
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		send(map[string]any{"type": "message_start", "message": map[string]any{
			"id": id, "type": "message", "role": "assistant", "model": req.Model, "content": []any{},
			"stop_reason": nil, "stop_sequence": nil, "usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
		}})
	}

	index := -1
	open := "" // the block being streamed: "thinking", "text", or none
	closeBlock := func() {
		if open != "" {
			send(map[string]any{"type": "content_block_stop", "index": index})
			open = ""
		}
	}
	startBlock := func(kind string, block map[string]any) {
		closeBlock()
		index++
		open = kind
		send(map[string]any{"type": "content_block_start", "index": index, "content_block": block})
	}

	onThinking := func(t string) {
		begin()
		if open != "thinking" {
			startBlock("thinking", map[string]any{"type": "thinking", "thinking": ""})
		}
		send(map[string]any{"type": "content_block_delta", "index": index,
			"delta": map[string]any{"type": "thinking_delta", "thinking": t}})
	}
	onContent := func(t string) {
		if open != "text" {
			// A reply opens with the newlines a template leaves after its think
			// block. A text block that starts with them is starting with that,
			// not with anything the model said.
			t = strings.TrimLeft(t, " \t\r\n")
			if t == "" {
				return
			}
			begin()
			startBlock("text", map[string]any{"type": "text", "text": ""})
		}
		send(map[string]any{"type": "content_block_delta", "index": index,
			"delta": map[string]any{"type": "text_delta", "text": t}})
	}

	rep, err := generate(r.Context(), eng, msgs, params, onThinking, onContent)
	if err != nil {
		if !streamed {
			writeAnthropicGenError(w, err)
			return
		}
		// The header is gone. Say so in the stream, in the shape its clients
		// read, rather than closing it as though the reply had ended.
		log.Printf("generation failed: %v", err)
		closeBlock()
		send(map[string]any{"type": "error", "error": map[string]string{"type": "api_error", "message": err.Error()}})
		return
	}

	begin()
	closeBlock()
	for _, c := range rep.Calls {
		// A call is only known once it is complete, so its input goes out as one
		// delta — the whole object, which is valid partial JSON too.
		index++
		send(map[string]any{"type": "content_block_start", "index": index, "content_block": map[string]any{
			"type": "tool_use", "id": newID("toolu"), "name": c.Name, "input": map[string]any{},
		}})
		send(map[string]any{"type": "content_block_delta", "index": index,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": string(jsonObject(c.Arguments))}})
		send(map[string]any{"type": "content_block_stop", "index": index})
	}
	send(map[string]any{"type": "message_delta",
		"delta": map[string]any{"stop_reason": anthropicStopReason(rep), "stop_sequence": nil},
		"usage": anthropicUsage(rep.Stats)})
	send(map[string]any{"type": "message_stop"})
}

// writeAnthropicError is this dialect's error envelope. Its clients switch on
// error.type, so the type matters as much as the status.
func writeAnthropicError(w http.ResponseWriter, status int, kind, message string) {
	writeJSON(w, status, map[string]any{
		"type": "error", "error": map[string]string{"type": kind, "message": message},
	})
}

// writeAnthropicGenError is writeGenError in the Anthropic dialect: 413 for a
// prompt that will never fit, 503 with Retry-After for a server that is full,
// 500 for anything else.
func writeAnthropicGenError(w http.ResponseWriter, err error) {
	log.Printf("generation failed: %v", err)

	var big *engine.PromptTooLongError
	if errors.As(err, &big) {
		writeAnthropicError(w, http.StatusRequestEntityTooLarge, "request_too_large", big.Error())
		return
	}
	var stale *engine.TimeoutError
	if errors.As(err, &stale) && stale.BeforeStarting() {
		writeAnthropicError(w, http.StatusServiceUnavailable, "overloaded_error", stale.Error())
		return
	}
	var busy *engine.BusyError
	if errors.As(err, &busy) {
		if busy.RetryAfter > 0 {
			w.Header().Set("Retry-After", strconv.Itoa(int(math.Ceil(busy.RetryAfter.Seconds()))))
		}
		writeAnthropicError(w, http.StatusServiceUnavailable, "overloaded_error", busy.Error())
		return
	}
	writeAnthropicError(w, http.StatusInternalServerError, "api_error", err.Error())
}
