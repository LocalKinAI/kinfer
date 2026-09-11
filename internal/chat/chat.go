// Package chat turns a message list into the exact prompt string an instruct
// model was fine-tuned on, and knows when that model has finished talking.
//
// This is the difference between continuation and conversation. Feed a bare
// question to an instruct model and it will happily continue the sentence; feed
// it the template it was trained on and it answers as the assistant, honouring
// the system prompt. For kinfer that distinction is the whole point: a LocalKin
// soul IS a system prompt, so without templating none of the 300+ personas work.
//
// The authoritative template lives in the GGUF's own metadata, and FromModel
// uses it: llama_chat_apply_template matches that string against the families
// llama.cpp implements in C++ and applies the right one. There are 54 of them,
// and no Jinja engine is involved — the Jinja text is matched, not executed.
//
// The three hand-written families below remain as a fallback, for a file whose
// metadata carries no template or one llama.cpp does not recognise. They were
// the only option while kinfer went through gollama.cpp, which exposed neither
// the metadata reader nor the template function.
package chat

import (
	"fmt"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// Message is one turn. Role is "system", "user", "assistant" or "tool".
type Message struct {
	Role    string
	Content string

	// ToolCalls are the functions an assistant turn asked to have run.
	ToolCalls []tools.Call
}

// Template renders a conversation and reports the model's stop markers.
type Template struct {
	// Name identifies the family, e.g. "chatml".
	Name string

	// Stops are literal strings that mean "the assistant is done".
	//
	// They are a backstop, not the mechanism: generation ends when
	// llama_vocab_is_eog reports an end-of-generation token. Matching text
	// catches a template that leaks its own marker as ordinary output, which
	// the hand-written families below are known to do. A template read from
	// GGUF metadata leaves this empty — its family is not known ahead of time,
	// so there is no marker to match.
	Stops []string

	render func(msgs []Message) string

	// toolFmt is the tool-call convention this model was trained on, read from
	// its own template. Declaring functions in another one makes the model
	// fight its training rather than answer.
	toolFmt tools.Format
}

// Render builds the prompt, ending in the assistant's opening so the model
// continues as the assistant rather than inventing another user turn.
//
// Available functions are folded into the conversation before it reaches the
// template. llama_chat_apply_template takes only {role, content}, with no tools
// parameter, so the tool branches of a model's Jinja template cannot be reached
// through the C API — but the convention they encode can be reproduced, and
// then the template has nothing unusual left to render.
func (t *Template) Render(msgs []Message, ts []tools.Tool) string {
	return t.render(fold(msgs, ts, t.toolFmt))
}

// fold rewrites a conversation into plain {role, content} turns.
//
// Three transformations, each one exactly what Qwen's own template does:
//
//   - the function signatures join the end of the system prompt, adding one if
//     the conversation has none
//   - an assistant turn's tool calls become <tool_call> blocks in its text,
//     which is the form the model itself produced them in
//   - a tool result becomes a user turn wrapped in <tool_response>, because no
//     instruct model has a "tool" role of its own
func fold(msgs []Message, ts []tools.Tool, f tools.Format) []Message {
	decl := f.Declare(ts)
	out := make([]Message, 0, len(msgs)+1)

	if decl != "" && (len(msgs) == 0 || msgs[0].Role != "system") {
		out = append(out, Message{Role: "system", Content: strings.TrimPrefix(decl, "\n\n")})
		decl = ""
	}

	for _, m := range msgs {
		switch {
		case m.Role == "system" && decl != "":
			m.Content += decl
			decl = ""
		case m.Role == "assistant" && len(m.ToolCalls) > 0:
			m.Content += f.RenderCalls(m.ToolCalls)
			m.ToolCalls = nil
		case m.Role == "tool":
			m.Role = "user"
			m.Content = "<tool_response>\n" + m.Content + "\n</tool_response>"
		}
		out = append(out, m)
	}
	return out
}

// ToolFormat is the tool-call convention this model knows, or tools.None when
// its template describes none. A model that knows none will answer in prose
// however the functions are declared, so a caller expecting a call should be
// told rather than left waiting.
func (t *Template) ToolFormat() tools.Format { return t.toolFmt }

// TrimStop cuts the output at the first stop marker and reports whether one was
// found. Generation loops call this every step: stop markers arrive as ordinary
// text, so without trimming the model's "<|im_end|>" would be shown to the user.
func (t *Template) TrimStop(s string) (string, bool) {
	cut := -1
	for _, stop := range t.Stops {
		if i := strings.Index(s, stop); i >= 0 && (cut < 0 || i < cut) {
			cut = i
		}
	}
	if cut < 0 {
		return s, false
	}
	return s[:cut], true
}

// chatml is used by Qwen, Yi, and most models converted after 2024.
var chatml = &Template{
	Name:  "chatml",
	Stops: []string{"<|im_end|>", "<|endoftext|>"},
	// The ChatML fallback is reached mainly for Qwen-lineage models, which is
	// where this convention comes from.
	toolFmt: tools.Hermes,
	render: func(msgs []Message) string {
		var b strings.Builder
		for _, m := range msgs {
			fmt.Fprintf(&b, "<|im_start|>%s\n%s<|im_end|>\n", m.Role, m.Content)
		}
		b.WriteString("<|im_start|>assistant\n")
		return b.String()
	},
}

// llama3 covers Llama 3.x and its derivatives.
var llama3 = &Template{
	Name:  "llama3",
	Stops: []string{"<|eot_id|>", "<|end_of_text|>"},
	render: func(msgs []Message) string {
		var b strings.Builder
		b.WriteString("<|begin_of_text|>")
		for _, m := range msgs {
			fmt.Fprintf(&b, "<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", m.Role, m.Content)
		}
		b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
		return b.String()
	},
}

// mistral has no system role: the system prompt is folded into the first user
// turn, which is what Mistral's own reference implementation does.
var mistral = &Template{
	Name:  "mistral",
	Stops: []string{"</s>", "[INST]"},
	render: func(msgs []Message) string {
		var b strings.Builder
		b.WriteString("<s>")

		var system string
		for _, m := range msgs {
			switch m.Role {
			case "system":
				system = m.Content
			case "user":
				content := m.Content
				if system != "" {
					content = system + "\n\n" + content
					system = ""
				}
				fmt.Fprintf(&b, "[INST] %s [/INST]", content)
			case "assistant":
				fmt.Fprintf(&b, " %s</s>", m.Content)
			}
		}
		return b.String()
	},
}

var families = map[string]*Template{
	chatml.Name:  chatml,
	llama3.Name:  llama3,
	mistral.Name: mistral,
}

// Get returns a template by family name.
func Get(name string) (*Template, error) {
	if t, ok := families[strings.ToLower(name)]; ok {
		return t, nil
	}
	var have []string
	for n := range families {
		have = append(have, n)
	}
	return nil, fmt.Errorf("unknown chat template %q (have: %s)", name, strings.Join(have, ", "))
}

// FromModel returns the template a model was actually fine-tuned on, read from
// its GGUF metadata, falling back to a guess from the filename.
//
// The guess is what kinfer used to do for every model, and it is wrong for
// anything outside the three built-in families — a Gemma or Phi model would be
// handed ChatML and answer, badly, in a format it never saw in training. The
// metadata knows; ask it.
//
// The template is exercised once here rather than trusted: llama.cpp rejects a
// template it does not implement, and finding that out on the first request
// would turn a startup problem into a serving one.
func FromModel(model llama.Model, path string) *Template {
	if model == 0 {
		return Detect(path)
	}
	tmpl := llama.ChatTemplate(model)
	if tmpl == "" {
		return Detect(path)
	}

	probe := []Message{{Role: "system", Content: "s"}, {Role: "user", Content: "u"}}
	if _, err := applyLlama(tmpl, probe); err != nil {
		return Detect(path)
	}

	return &Template{
		Name:    "gguf",
		toolFmt: tools.Detect(tmpl),
		// No stop strings. Generation ends on an end-of-generation token, which
		// llama_vocab_is_eog reports for whatever terminator this model uses —
		// a surer signal than matching text, and the only one available when
		// the family is not known ahead of time.
		Stops: nil,
		render: func(msgs []Message) string {
			out, err := applyLlama(tmpl, msgs)
			if err != nil {
				// Rendering was proved to work at load time, so an error here
				// means a message this conversation carries is the problem.
				// Falling back keeps the request answerable.
				return chatml.render(msgs)
			}
			return out
		},
	}
}

func applyLlama(tmpl string, msgs []Message) (string, error) {
	roles := make([]string, len(msgs))
	contents := make([]string, len(msgs))
	for i, m := range msgs {
		roles[i], contents[i] = m.Role, m.Content
	}
	return llama.ApplyChatTemplate(tmpl, roles, contents, true)
}

// Detect guesses the family from a model's filename.
//
// A guess, and a last resort: prefer FromModel, which reads the template the
// model was actually trained on. This runs only when a GGUF carries no template
// or llama.cpp does not implement the one it carries. ChatML is the default
// because it is both the most common and the most forgiving — a model that
// wants a different template still produces coherent text from ChatML, whereas
// the reverse often collapses into gibberish.
func Detect(modelPath string) *Template {
	name := strings.ToLower(modelPath)
	switch {
	case strings.Contains(name, "llama-3"), strings.Contains(name, "llama3"):
		return llama3
	case strings.Contains(name, "mistral"), strings.Contains(name, "mixtral"):
		return mistral
	default:
		return chatml
	}
}
