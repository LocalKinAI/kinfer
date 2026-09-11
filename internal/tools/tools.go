// Package tools lets a model ask for a function to be run.
//
// There is no single convention for this. Qwen and the Hermes lineage declare
// the functions in the system prompt and answer with JSON wrapped in tags:
//
//	<tool_call>
//	{"name": "get_weather", "arguments": {"city": "Berlin"}}
//	</tool_call>
//
// Others were trained on a nested XML form instead:
//
//	<tool_call>
//	<function=get_weather>
//	<parameter=city>
//	Berlin
//	</parameter>
//	</function>
//	</tool_call>
//
// Which one a model knows is not a matter of taste. Declaring functions in the
// wrong convention makes it fight its own training: ornith-1.5:35b, told to
// answer in JSON when it was taught the XML form, produced
// `{"name":_get_weather"` — deterministically, four runs out of four. It was
// trying to comply with an instruction that contradicted everything it had
// learned.
//
// So the convention is read out of the model's own chat template, and both the
// declaration and the parser follow from it. The declarations below reproduce
// each family's template wording exactly, because a model was fine-tuned on
// that text and not on a paraphrase of it.
//
// Doing this here rather than through llama.cpp is forced:
// llama_chat_apply_template takes only {role, content}, with no tools
// parameter, so a template's tool branches are unreachable through the C API.
// What is reachable is the conventions they encode.
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
	// key order and number formatting for no benefit.
	Arguments json.RawMessage `json:"arguments"`
}

// Format is a model family's tool-call convention.
type Format int

const (
	// None means the model's template describes no tool convention. Declaring
	// functions to it produces prose where a caller expects a call.
	None Format = iota

	// Hermes is JSON inside <tool_call> tags: Qwen, Hermes, and most of what
	// has been converted since.
	Hermes

	// FunctionXML is a nested <function=name><parameter=key> form, also inside
	// <tool_call> tags.
	FunctionXML
)

func (f Format) String() string {
	switch f {
	case Hermes:
		return "hermes"
	case FunctionXML:
		return "function-xml"
	default:
		return "none"
	}
}

const (
	openTag  = "<tool_call>"
	closeTag = "</tool_call>"
)

// Detect reads the convention out of a model's chat template.
//
// The template is where a model states how it was taught to answer, and the
// two conventions are distinguishable: only the XML one mentions <function=.
// Checking in that order matters, because both wrap their calls in
// <tool_call>.
func Detect(chatTemplate string) Format {
	switch {
	case strings.Contains(chatTemplate, "<function="):
		return FunctionXML
	case strings.Contains(chatTemplate, openTag):
		return Hermes
	default:
		return None
	}
}

// Declare renders the tool section that belongs at the end of the system
// prompt, in the wording this format's models were trained on.
func (f Format) Declare(ts []Tool) string {
	if len(ts) == 0 || f == None {
		return ""
	}

	var b strings.Builder
	b.WriteString("\n\n# Tools\n\n")
	switch f {
	case Hermes:
		b.WriteString("You may call one or more functions to assist with the user query.\n\n")
		b.WriteString("You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		writeToolJSON(&b, ts)
		b.WriteString("\n</tools>\n\nFor each function call, return a json object with function name ")
		b.WriteString("and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n")
		b.WriteString(`{"name": <function-name>, "arguments": <args-json-object>}`)
		b.WriteString("\n</tool_call>")
	case FunctionXML:
		b.WriteString("You have access to the following functions:\n\n<tools>")
		writeToolJSON(&b, ts)
		b.WriteString("\n</tools>")
		b.WriteString("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n")
		b.WriteString("<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n")
		b.WriteString("<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\n")
		b.WriteString("multiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n")
		b.WriteString("- Function calls MUST follow the specified format: an inner <function=...></function> block ")
		b.WriteString("must be nested within <tool_call></tool_call> XML tags\n")
		b.WriteString("- Required parameters MUST be specified\n")
		b.WriteString("- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n")
		b.WriteString("- If there is no function call available, answer the question like normal with your current knowledge ")
		b.WriteString("and do not tell the user about function calls\n</IMPORTANT>")
	}
	return b.String()
}

func writeToolJSON(b *strings.Builder, ts []Tool) {
	for _, t := range ts {
		raw, err := json.Marshal(t)
		if err != nil {
			continue // a tool that will not marshal cannot be offered
		}
		b.WriteString("\n")
		b.Write(raw)
	}
}

// RenderCalls turns an assistant's tool calls back into the text the model
// produced, so a later turn sees its own earlier output verbatim.
func (f Format) RenderCalls(calls []Call) string {
	var b strings.Builder
	for _, c := range calls {
		args := string(c.Arguments)
		if args == "" {
			args = "{}"
		}
		switch f {
		case FunctionXML:
			fmt.Fprintf(&b, "\n%s\n<function=%s>\n", openTag, c.Name)
			var fields map[string]json.RawMessage
			if json.Unmarshal([]byte(args), &fields) == nil {
				for k, v := range fields {
					fmt.Fprintf(&b, "<parameter=%s>\n%s\n</parameter>\n", k, scalarText(v))
				}
			}
			fmt.Fprintf(&b, "</function>\n%s", closeTag)
		default:
			fmt.Fprintf(&b, "\n%s\n{\"name\": %q, \"arguments\": %s}\n%s", openTag, c.Name, args, closeTag)
		}
	}
	return b.String()
}

// Parse splits a reply into the prose the model wrote and the calls it asked
// for.
//
// Text outside the tags is returned as content, and a malformed block is left
// in the content rather than dropped: a reply cut off mid-call is something the
// caller should see, not something to pretend did not happen. That choice is
// what made this package's own mismatch legible — a model answering in the
// wrong convention showed up as visible garbage instead of silence.
func (f Format) Parse(reply string) (content string, calls []Call) {
	if f == None || !strings.Contains(reply, openTag) {
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
		if c, ok := f.parseBody(body); ok {
			calls = append(calls, c)
		} else {
			text.WriteString(rest[i : j+len(closeTag)])
		}
		rest = rest[j+len(closeTag):]
	}
	return strings.TrimSpace(text.String()), calls
}

func (f Format) parseBody(body string) (Call, bool) {
	if f == FunctionXML {
		return parseFunctionXML(body)
	}
	var c Call
	if err := json.Unmarshal([]byte(body), &c); err != nil || c.Name == "" {
		return Call{}, false
	}
	return c, true
}

// parseFunctionXML reads <function=name><parameter=key>value</parameter>…
func parseFunctionXML(body string) (Call, bool) {
	const fnOpen = "<function="
	i := strings.Index(body, fnOpen)
	if i < 0 {
		return Call{}, false
	}
	rest := body[i+len(fnOpen):]
	j := strings.IndexByte(rest, '>')
	if j < 0 {
		return Call{}, false
	}
	name := strings.TrimSpace(rest[:j])
	if name == "" {
		return Call{}, false
	}
	rest = rest[j+1:]

	args := map[string]json.RawMessage{}
	for {
		const pOpen, pClose = "<parameter=", "</parameter>"
		a := strings.Index(rest, pOpen)
		if a < 0 {
			break
		}
		rest = rest[a+len(pOpen):]
		b := strings.IndexByte(rest, '>')
		if b < 0 {
			break
		}
		key := strings.TrimSpace(rest[:b])
		rest = rest[b+1:]

		c := strings.Index(rest, pClose)
		if c < 0 {
			break
		}
		args[key] = jsonValue(strings.TrimSpace(rest[:c]))
		rest = rest[c+len(pClose):]
	}

	raw, err := json.Marshal(args)
	if err != nil {
		return Call{}, false
	}
	return Call{Name: name, Arguments: raw}, true
}

// jsonValue keeps a parameter's text as the JSON type it looks like — a number
// stays a number — and falls back to a string, which is what most values are.
func jsonValue(s string) json.RawMessage {
	switch s {
	case "true", "false", "null":
		return json.RawMessage(s)
	}
	if len(s) > 0 && (s[0] == '{' || s[0] == '[' || s[0] == '-' || (s[0] >= '0' && s[0] <= '9')) {
		var probe any
		if json.Unmarshal([]byte(s), &probe) == nil {
			return json.RawMessage(s)
		}
	}
	raw, _ := json.Marshal(s)
	return raw
}

// scalarText renders a JSON value as the plain text the XML form expects.
func scalarText(v json.RawMessage) string {
	var s string
	if json.Unmarshal(v, &s) == nil {
		return s
	}
	return string(v)
}
