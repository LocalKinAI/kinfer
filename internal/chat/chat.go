// Package chat turns a message list into the exact prompt string an instruct
// model was fine-tuned on, and knows when that model has finished talking.
//
// This is the difference between continuation and conversation. Feed a bare
// question to an instruct model and it will happily continue the sentence; feed
// it the template it was trained on and it answers as the assistant, honouring
// the system prompt. For kinfer that distinction is the whole point: a LocalKin
// soul IS a system prompt, so without templating none of the 300+ personas work.
//
// Templates are built in rather than read from the GGUF. llama.cpp stores a
// Jinja template in the file's metadata, but gollama.cpp exposes neither the
// metadata reader nor llama_chat_apply_template, and shipping a Jinja engine to
// render one string is a poor trade. The three families below cover the models
// kinfer actually runs; add a family when a model needs it.
package chat

import (
	"fmt"
	"strings"
)

// Message is one turn. Role is "system", "user", or "assistant".
type Message struct {
	Role    string
	Content string
}

// Template renders a conversation and reports the model's stop markers.
type Template struct {
	// Name identifies the family, e.g. "chatml".
	Name string

	// Stops are literal strings that mean "the assistant is done". The vocab's
	// EOS token id would be cleaner, but gollama.cpp exposes no way to read it,
	// so kinfer matches on the decoded text instead.
	Stops []string

	render func(msgs []Message) string
}

// Render builds the prompt, ending in the assistant's opening so the model
// continues as the assistant rather than inventing another user turn.
func (t *Template) Render(msgs []Message) string { return t.render(msgs) }

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

// Detect guesses the family from a model's filename.
//
// A guess, not a lookup — the authoritative template lives in the GGUF metadata
// we cannot read yet. ChatML is the default because it is both the most common
// and the most forgiving: a model that wants a different template still produces
// coherent text from ChatML, whereas the reverse often collapses into gibberish.
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
