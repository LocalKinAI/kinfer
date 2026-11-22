package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"regexp"
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

	// ToolName is a tool_reference block's: the tool a client-side tool
	// search found, in the result of that search.
	ToolName string `json:"tool_name"`

	// Source is an image block's picture: {"type": "base64", "data": …}, or
	// a "url", which kinfer does not fetch.
	Source *struct {
		Type string `json:"type"`
		Data string `json:"data"`
		URL  string `json:"url"`
	} `json:"source"`
}

type anthropicTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`

	// DeferLoading marks a tool the model is not to be shown until a tool
	// search has found it: Claude Code sends its MCP tools so, hundreds of
	// them in some setups, and names the ones found in tool_reference blocks.
	DeferLoading bool `json:"defer_loading"`
}

// loadedTools are the deferred tools this conversation has loaded: the ones a
// tool search's result names (tool_reference) and the ones the model has
// already called.
func loadedTools(req *anthropicRequest) map[string]bool {
	loaded := map[string]bool{}
	for _, m := range req.Messages {
		blocks, err := anthropicBlocks(m.Content)
		if err != nil {
			continue // anthropicToChat reports it
		}
		for _, b := range blocks {
			switch b.Type {
			case "tool_use":
				loaded[b.Name] = true
			case "tool_result":
				for _, name := range toolReferences(b.Content) {
					loaded[name] = true
				}
			}
		}
	}
	return loaded
}

// toolReferences are the tools a tool result's content names as found.
func toolReferences(content json.RawMessage) []string {
	blocks, err := anthropicBlocks(content)
	if err != nil {
		return nil
	}
	var names []string
	for _, b := range blocks {
		if b.Type == "tool_reference" && b.ToolName != "" {
			names = append(names, b.ToolName)
		}
	}
	return names
}

// referencedTools is what a tool search's result reads as to the model: the
// definitions of the tools it found, as JSON, where the result stands in the
// conversation. A name the request does not declare is left out.
func referencedTools(names []string, declared []anthropicTool) string {
	type def struct {
		Type        string          `json:"type"`
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	}
	var defs []def
	for _, name := range names {
		for _, t := range declared {
			if t.Name == name {
				d := def{Type: "function", Name: name, Description: t.Description}
				var b bytes.Buffer
				if len(t.InputSchema) > 0 && json.Compact(&b, t.InputSchema) == nil {
					d.Parameters = b.Bytes()
				}
				defs = append(defs, d)
				break
			}
		}
	}
	if len(defs) == 0 {
		return ""
	}
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	enc.Encode(defs)
	return strings.TrimSuffix(out.String(), "\n")
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

// billingStamp is the part of the line Claude Code opens its system prompt with
// that changes on every request, the cch:
//
//	x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You are Claude Code, …
//
// It means nothing to a model, but as the prompt's first tokens it kept the
// prefix pool from matching past it, so every turn of a Claude Code session —
// tens of thousands of tokens of system prompt and tools — was read whole.
// llama.cpp's server pins it the same way (#21793).
var billingStamp = regexp.MustCompile(`(?m)^(x-anthropic-billing-header:[^\n]*?\bcch=)[0-9A-Za-z]+;`)

func pinBillingStamp(system string) string {
	return billingStamp.ReplaceAllString(system, "${1}00000;")
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
	system = pinBillingStamp(system)
	if oc := req.OutputConfig; oc != nil && oc.Format != nil && oc.Format.Type == "json_schema" && len(oc.Format.Schema) > 0 {
		// The schema is stated, for a model whose replies cannot be held to
		// one while it samples (MLX), and for one that can, which then knows
		// what the properties are for. Claude Code uses this for a session title:
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
		var pics [][]byte
		flush := func() {
			if len(text) > 0 {
				out = append(out, chat.Message{Role: role, Content: strings.Join(text, "\n"), Images: pics})
				text, pics = nil, nil
			}
		}
		for _, b := range blocks {
			switch b.Type {
			case "text":
				text = append(text, b.Text)
			case "image":
				if b.Source == nil {
					return nil, fmt.Errorf("messages[%d]: an image block without a source", i)
				}
				src := b.Source.Data
				if b.Source.Type == "url" {
					src = b.Source.URL
				}
				pic, err := picture(src)
				if err != nil {
					return nil, fmt.Errorf("messages[%d]: %w", i, err)
				}
				text = append(text, engine.ImageMarker)
				pics = append(pics, pic)
			case "document":
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
				if defs := referencedTools(toolReferences(b.Content), req.Tools); defs != "" {
					result = strings.TrimSpace(result + "\n\n" + defs)
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
//
// A deferred tool (defer_loading) is kept out until the conversation has
// loaded it: with every one of a setup's MCP tools written into the prompt, a
// turn carried tens of thousands of tokens of tools nobody had asked for. When
// the request asks for Anthropic's own tool search (tool_search_tool_…), which
// nothing here runs, no tool could ever be loaded, so all are kept.
func anthropicTools(in []anthropicTool, loaded map[string]bool) []tools.Tool {
	var out []tools.Tool
	deferring := true
	for _, t := range in {
		if strings.HasPrefix(t.Type, "tool_search_tool") {
			deferring = false
		}
	}
	for _, t := range in {
		if t.Name == "" || len(t.InputSchema) == 0 {
			continue
		}
		if t.DeferLoading && deferring && !loaded[t.Name] {
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

// anthropicUsage counts as Anthropic does: input_tokens is the part of the
// prompt that was read, cache_read_input_tokens the part that was held from
// before, and the prompt is the two together. Claude Code adds them up for how
// full the context is, so the sum must stay the whole prompt.
func anthropicUsage(st engine.Stats) map[string]any {
	cached := min(st.CachedTokens, st.PromptTokens)
	return map[string]any{"input_tokens": st.PromptTokens - cached, "output_tokens": st.EvalTokens,
		"cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}
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
		status := acquireStatus(err)
		writeAnthropicError(w, status, anthropicKind(status), err.Error())
		return
	}
	defer release()

	params := engine.DefaultGenParams()
	if !toolChoiceNone(req.ToolChoice) {
		params.Tools = anthropicTools(req.Tools, loadedTools(&req))
	}
	if oc := req.OutputConfig; oc != nil && oc.Format != nil && oc.Format.Type == "json_schema" && len(oc.Format.Schema) > 0 {
		// Stated in the system prompt (anthropicToChat), and held to where
		// the model's replies can be. A schema that cannot be held to is
		// left stated: Claude Code asks this way for a session's title.
		if _, err := tools.JSONGrammar(oc.Format.Schema); err == nil {
			params.Format = oc.Format.Schema
		}
	}
	params.NoThink = req.Thinking != nil && req.Thinking.Type == "disabled"
	params.Think = req.Thinking != nil && req.Thinking.Type != "" && req.Thinking.Type != "disabled"
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
// handleAnthropicCountTokens answers POST /v1/messages/count_tokens: how many
// tokens the request's system prompt, history and tools come to, rendered as
// /v1/messages would render them for the model. Claude Code asks it to measure a
// conversation; it went unanswered (404), and Claude Code estimated instead.
// The model is loaded if it is not, as it would be for the turn being sized.
func (s *Server) handleAnthropicCountTokens(w http.ResponseWriter, r *http.Request) {
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
		status := acquireStatus(err)
		writeAnthropicError(w, status, anthropicKind(status), err.Error())
		return
	}
	defer release()

	params := engine.DefaultGenParams()
	if !toolChoiceNone(req.ToolChoice) {
		params.Tools = anthropicTools(req.Tools, loadedTools(&req))
	}
	params.NoThink = req.Thinking != nil && req.Thinking.Type == "disabled"
	params.Think = req.Thinking != nil && req.Thinking.Type != "" && req.Thinking.Type != "disabled"
	n, err := eng.CountTokens(msgs, params)
	if err != nil {
		status, kind := http.StatusInternalServerError, "api_error"
		if pictureFault(err) {
			status, kind = http.StatusBadRequest, "invalid_request_error"
		}
		writeAnthropicError(w, status, kind, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"input_tokens": n})
}

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
	if pictureFault(err) {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
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

// anthropicKind is the error type Anthropic's API gives a status: a model that
// would not load is the server's own failure, api_error.
func anthropicKind(status int) string {
	switch status {
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusInternalServerError:
		return "api_error"
	}
	return "not_found_error"
}
