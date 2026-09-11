// Package tools lets a model ask for a function to be run.
//
// It implements the convention Qwen and the Hermes-style models were trained
// on: the available functions are declared inside the system prompt, and the
// model answers with JSON wrapped in <tool_call> tags.
//
//	<tool_call>
//	{"name": "get_weather", "arguments": {"city": "Berlin"}}
//	</tool_call>
//
// Doing it here rather than through llama.cpp is forced.
// llama_chat_apply_template takes only {role, content} — it has no tools
// parameter at all — so the tool sections of a model's Jinja template are
// unreachable through the C API. What is reachable is the convention those
// templates encode, and this package reproduces it exactly as Qwen's own
// template writes it.
//
// That convention is not universal. Llama 3.1 emits <|python_tag|>, Mistral
// emits [TOOL_CALLS], and a model trained on either will not answer in tags it
// has never seen. Supported reports whether a model's own template uses this
// one, so a request can be refused rather than silently answered with prose
// where a caller expected a function call.
package tools

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Tool is one function a model may ask to call, in the shape both the Ollama
// and OpenAI dialects use.
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function describes the callable.
type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Call is a model's request to run one function.
type Call struct {
	Name string `json:"name"`

	// Arguments stays raw. It is generated JSON: re-encoding it would change
	// key order and number formatting for no benefit, and a caller that wants
	// it as a map can unmarshal it itself.
	Arguments json.RawMessage `json:"arguments"`
}

const (
	openTag  = "<tool_call>"
	closeTag = "</tool_call>"
)

// Supported reports whether a model's chat template uses this convention.
//
// The check is the template's own text: a model trained to emit <tool_call>
// says so in the template it ships with. A model that does not will answer in
// prose no matter how the functions are declared, and a caller waiting for a
// function call would wait forever.
func Supported(chatTemplate string) bool {
	return strings.Contains(chatTemplate, openTag)
}

// Declare renders the tool section that belongs at the end of the system
// prompt, in the exact wording Qwen's template uses — down to the line breaks,
// because the model was fine-tuned on this text and not on a paraphrase of it.
func Declare(ts []Tool) string {
	if len(ts) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n")
	b.WriteString("You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
	for _, t := range ts {
		raw, err := json.Marshal(t)
		if err != nil {
			continue // a tool that will not marshal cannot be offered
		}
		b.WriteString("\n")
		b.Write(raw)
	}
	b.WriteString("\n</tools>\n\nFor each function call, return a json object with function name ")
	b.WriteString("and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n")
	b.WriteString(`{"name": <function-name>, "arguments": <args-json-object>}`)
	b.WriteString("\n</tool_call>")
	return b.String()
}

// RenderCalls turns an assistant's tool calls back into the text form the model
// produced, so a later turn sees its own earlier output verbatim.
func RenderCalls(calls []Call) string {
	var b strings.Builder
	for _, c := range calls {
		args := string(c.Arguments)
		if args == "" {
			args = "{}"
		}
		fmt.Fprintf(&b, "\n%s\n{\"name\": %q, \"arguments\": %s}\n%s", openTag, c.Name, args, closeTag)
	}
	return b.String()
}

// Parse splits a reply into the prose the model wrote and the calls it asked
// for. Text outside the tags is returned as content; malformed blocks are left
// in the content rather than dropped, because a half-written call is something
// the caller should see rather than something to pretend did not happen.
func Parse(reply string) (content string, calls []Call) {
	if !strings.Contains(reply, openTag) {
		return reply, nil
	}

	var text strings.Builder
	rest := reply
	for {
		i := strings.Index(rest, openTag)
		if i < 0 {
			text.WriteString(rest)
			break
		}
		j := strings.Index(rest[i:], closeTag)
		if j < 0 {
			// Unterminated: the reply was cut off mid-call. Keep it visible.
			text.WriteString(rest)
			break
		}
		j += i

		// Whatever came before the tag is prose, and it comes first.
		text.WriteString(rest[:i])

		body := strings.TrimSpace(rest[i+len(openTag) : j])
		var c Call
		if err := json.Unmarshal([]byte(body), &c); err == nil && c.Name != "" {
			calls = append(calls, c)
		} else {
			text.WriteString(rest[i : j+len(closeTag)])
		}
		rest = rest[j+len(closeTag):]
	}
	return strings.TrimSpace(text.String()), calls
}
