package mlxengine

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// How Inkling 2 is spoken to and how it answers.
//
// A prompt is a row of frames, each a role token — with a tool's name after
// it for a tool's result or the model's call — a content token saying what
// the frame holds, the content, and <|end_message|>. A reply is the same:
// usually a frame of thinking, then one of text or calls of tools, then
// <|content_model_end_sampling|>. The rest of kinfer knows a reply as text
// with the thinking in <think> tags and calls in <tool_call> tags, so the
// reply is written out that way as it is read (inkReply).
//
// This follows the chat template Thinking Machines publish with the model.
const (
	inkUser        = "<|message_user|>"
	inkModel       = "<|message_model|>"
	inkSystem      = "<|message_system|>"
	inkTool        = "<|message_tool|>"
	inkText        = "<|content_text|>"
	inkThinking    = "<|content_thinking|>"
	inkDeclare     = "<|content_xml|>"
	inkInvoke      = "<|content_invoke_tool_json|>"
	inkToolText    = "<|content_invoke_tool_text|>"
	inkToolError   = "<|content_tool_error|>"
	inkEnd         = "<|end_message|>"
	inkEndSampling = "<|content_model_end_sampling|>"
)

// inkPiece is a stretch of a prompt: the template's own, whose tokens are
// read as tokens, or something a caller wrote, which is text whatever it
// spells.
type inkPiece struct {
	text  string
	given bool
}

type inkPrompt []inkPiece

func (p *inkPrompt) own(s string)   { *p = append(*p, inkPiece{text: s}) }
func (p *inkPrompt) given(s string) { *p = append(*p, inkPiece{text: s, given: true}) }

func (p inkPrompt) String() string {
	var b strings.Builder
	for _, x := range p {
		b.WriteString(x.text)
	}
	return b.String()
}

// inkEffort is the number the prompt's "Thinking effort level" line gives:
// 0 for no thinking — the model still thinks a little — to 0.99.
func inkEffort(th chat.Thinking) (string, error) {
	if th.Off {
		return "0", nil
	}
	switch e := strings.ToLower(strings.TrimSpace(th.Effort)); e {
	case "", "high":
		return "0.9", nil
	case "none":
		return "0", nil
	case "minimal":
		return "0.1", nil
	case "low":
		return "0.2", nil
	case "medium":
		return "0.7", nil
	case "xhigh", "max":
		return "0.99", nil
	default:
		// A number from 1 to 100, as other models' budgets are given.
		n, err := strconv.ParseFloat(e, 64)
		if err != nil || n < 0 || n > 100 {
			return "", fmt.Errorf("unknown reasoning effort %q", th.Effort)
		}
		if n == 0 {
			return "0", nil
		}
		return strconv.FormatFloat(min(n/100, 0.99), 'f', -1, 64), nil
	}
}

// inkRender is the prompt for a conversation.
func inkRender(msgs []chat.Message, ts []tools.Tool, th chat.Thinking) (inkPrompt, error) {
	var p inkPrompt
	if len(ts) > 0 {
		decl, err := tools.InklingDeclare(ts)
		if err != nil {
			return nil, err
		}
		p.own(inkSystem + "tool_declare" + inkDeclare)
		p.given(decl)
		p.own(inkEnd)
	}
	effort, err := inkEffort(th)
	if err != nil {
		return nil, err
	}
	said := false
	sayEffort := func() {
		p.own(inkSystem + inkText + "Thinking effort level: " + effort + inkEnd)
		said = true
	}
	// A tool's result is headed by the tool's name: the next of the calls the
	// turn before it made.
	var calls []tools.Call
	for _, m := range msgs {
		var role string
		switch m.Role {
		case "user":
			role = inkUser
		case "assistant":
			role = inkModel
		case "system":
			role = inkSystem
		case "tool":
			role = inkTool
		default:
			return nil, fmt.Errorf("unknown message role %q", m.Role)
		}
		if !said && m.Role != "system" {
			sayEffort()
		}
		if len(m.Images) > 0 || strings.Contains(m.Content, engine.ImageMarker) {
			return nil, fmt.Errorf("pictures are not read by kinfer's Inkling 2 yet")
		}
		head := func(content string) {
			p.own(role)
			if m.Role == "tool" && len(calls) > 0 {
				p.given(calls[0].Name)
			}
			p.own(content)
		}
		if m.Role == "assistant" {
			calls = m.ToolCalls
			if m.Thinking != "" {
				head(inkThinking)
				p.given(m.Thinking)
				p.own(inkEnd)
			}
		}
		if m.Content != "" || !(m.Role == "assistant" && (len(m.ToolCalls) > 0 || m.Thinking != "")) {
			head(inkText)
			p.given(m.Content)
			p.own(inkEnd)
		}
		if m.Role == "tool" && len(calls) > 0 {
			calls = calls[1:]
		}
		if m.Role == "assistant" {
			for _, c := range m.ToolCalls {
				body, err := tools.InklingInvoke(c)
				if err != nil {
					return nil, err
				}
				p.own(inkModel)
				p.given(c.Name)
				p.own(inkInvoke)
				p.given(body)
				p.own(inkEnd)
			}
			p.own(inkEndSampling)
		}
	}
	if !said {
		sayEffort()
	}
	return p, nil
}

// inkMarks are the tokens a reply's frames are made of.
type inkMarks struct {
	model, text, thinking, invoke, toolText, toolError, end, endSampling int32
}

// inkReply writes a reply out, token by token, as the rest of kinfer reads
// one: thinking in <think> tags before anything else, text as it is, and each
// call of a tool as <tool_call>{"name": …, "arguments": …}</tool_call>.
type inkReply struct {
	marks   inkMarks
	bytes   func(tok int32) []byte
	mode    int
	name    []byte // what followed the role token of this frame: a tool's name
	body    []byte // a call being written
	thought bool   // the thinking has been opened
	closed  bool   // and closed
	said    bool   // text or a call has been written
}

const (
	inkBetween = iota
	inkHeader
	inkInThinking
	inkInText
	inkInCall
	inkDropped
)

// take reads one token of the reply and returns what it adds to the text,
// and whether the reply is over.
func (r *inkReply) take(tok int32) (out []byte, done bool) {
	closeThought := func() {
		if r.thought && !r.closed {
			out = append(out, "</think>\n\n"...)
			r.closed = true
		}
	}
	switch tok {
	case r.marks.model:
		r.mode, r.name = inkHeader, nil
	case r.marks.thinking:
		// Thinking after the answer began has nowhere to go.
		if r.said || r.closed {
			r.mode = inkDropped
			break
		}
		if !r.thought {
			out = append(out, "<think>"...)
			r.thought = true
		}
		r.mode = inkInThinking
	case r.marks.text:
		closeThought()
		r.mode = inkInText
	case r.marks.invoke:
		closeThought()
		r.mode, r.body = inkInCall, nil
	case r.marks.toolText, r.marks.toolError:
		r.mode = inkDropped
	case r.marks.end:
		if r.mode == inkInCall {
			out = append(out, r.call()...)
			r.said = true
		}
		r.mode = inkBetween
	case r.marks.endSampling:
		if r.mode == inkInCall && len(r.body) > 0 {
			out = append(out, r.call()...)
		}
		closeThought()
		return out, true
	default:
		b := r.bytes(tok)
		switch r.mode {
		case inkHeader:
			r.name = append(r.name, b...)
		case inkInThinking:
			out = append(out, b...)
		case inkInCall:
			r.body = append(r.body, b...)
		case inkDropped:
		default:
			// Text, in a frame or — a reply cut loose from its frames — not.
			closeThought()
			out = append(out, b...)
			r.said = r.said || len(b) > 0
		}
	}
	return out, false
}

// call is the call just read, as kinfer's own tags give one; what is not a
// call is left as the model wrote it, to be seen.
func (r *inkReply) call() []byte {
	c, err := tools.InklingCall(string(r.body), string(r.name))
	if err != nil {
		return []byte("<tool_call>" + string(r.body) + "</tool_call>")
	}
	b, _ := json.Marshal(c)
	return []byte("<tool_call>\n" + string(b) + "\n</tool_call>")
}
