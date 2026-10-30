package chat

import (
	"strconv"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/tools"
)

// DeepSeek V4.1 has a prompt format llama.cpp does not implement, defined by
// the Python encoder that ships with the model (encoding/encoding.py in
// deepseek-ai/DeepSeek-V4.1-Flash-1017). This is that encoder's rendering,
// ported rule for rule and checked against the five cases DeepSeek publishes
// beside it (testdata/deepseek-v4.1). DeepSeek's server library,
// deepseek-recipe, renders the same prompts; where the two differ, this
// follows the model's own encoder and the difference is noted.
//
// What makes V4.1 more than another set of tags:
//
//   - A system token, written only when there is a system message — or, in
//     thinking mode, a "Reasoning Effort: N" line, which it leads.
//   - The prompt opens the reply's think block: it ends in <｜Assistant｜><think>,
//     so the model starts reasoning straight away and writes only </think>.
//     With thinking off it ends in <｜Assistant｜></think> instead.
//   - Earlier reasoning is dropped once a newer user turn has begun — unless
//     functions are on the table, in which case every turn keeps it.
//   - No tool role: results become <tool_result> blocks in a user turn, joined
//     with any text that follows them.

const (
	dsBOS       = "<｜begin▁of▁sentence｜>"
	dsEOS       = "<｜end▁of▁sentence｜>"
	dsSystem    = "<｜System｜>"
	dsUser      = "<｜User｜>"
	dsAssistant = "<｜Assistant｜>"
	dsReminder  = "<｜latest_reminder｜>"
)

var deepseekV41 = &Template{
	Name:       "deepseek-v4.1",
	toolFmt:    tools.DSML,
	opensThink: true,
	renderAll:  renderDeepseekV41,
	// No stop strings: <｜end▁of▁sentence｜> is the vocabulary's end of
	// generation, and nothing else ends a turn.
}

// renderDeepseekV41 renders a conversation for the model's next reply.
func renderDeepseekV41(msgs []Message, ts []tools.Tool, th Thinking) string {
	body, open := deepseekV41Body(msgs, ts, th)
	if !open {
		// A conversation that ends on an assistant turn still wants a reply,
		// which DeepSeek's server opens the way a user turn would.
		body += dsAssistant + dsMarker(!th.Off)
	}
	return body
}

type dsMsg struct {
	role     string
	content  string
	thinking string
	calls    []tools.Call
	toolOnly bool // a user turn made of tool results alone, which starts no new turn
}

// deepseekV41Body is the prompt exactly as DeepSeek's encoder writes it, and
// whether it already ends by opening the assistant's reply.
func deepseekV41Body(msgs []Message, ts []tools.Tool, th Thinking) (string, bool) {
	ms := mergeDeepseekTurns(msgs)

	// The tool section joins the first system message, which is created when
	// the conversation has none.
	if len(ts) > 0 {
		if len(ms) == 0 || ms[0].role != "system" {
			ms = append([]dsMsg{{role: "system"}}, ms...)
		}
		ms[0].content += tools.DSML.Declare(ts)
	}

	thinking := !th.Off
	// With functions on the table, every turn's reasoning stays.
	drop := len(ts) == 0

	// The last user turn: tool results do not start one, and a system message
	// after the first does.
	lastUser := -1
	for i := len(ms) - 1; i >= 0; i-- {
		if (ms[i].role == "user" && !ms[i].toolOnly) || (ms[i].role == "system" && i > 0) {
			lastUser = i
			break
		}
	}

	var b strings.Builder
	b.WriteString(dsBOS)
	headerAt := -1
	for i, m := range ms {
		if i == 0 {
			if thinking {
				b.WriteString(dsSystem)
				b.WriteString("Reasoning Effort: " + dsEffort(th.Effort) +
					" (range 1-100, the higher the value, the more thorough the reasoning)\n\n")
			} else if m.role == "system" {
				b.WriteString(dsSystem)
			}
		}

		switch m.role {
		case "system":
			if i > 0 {
				b.WriteString(dsSystem)
			}
			b.WriteString(m.content)
		case "latest_reminder":
			b.WriteString(dsReminder + m.content)
		case "assistant":
			if thinking && (!drop || i > lastUser) {
				b.WriteString(m.thinking + thinkClose)
			}
			b.WriteString(m.content)
			b.WriteString(tools.DSML.RenderCalls(m.calls))
			b.WriteString(dsEOS)
		default: // user, and the tool results merged into one
			b.WriteString(dsUser + m.content)
		}

		// A turn that the assistant answers next — or that ends the
		// conversation — opens the assistant's reply after itself.
		if i+1 < len(ms) && ms[i+1].role != "assistant" && ms[i+1].role != "latest_reminder" {
			continue
		}
		if m.role == "user" || (m.role == "system" && i > 0) {
			b.WriteString(dsAssistant + dsMarker(thinking && (!drop || i >= lastUser)))
			headerAt = i
		}
	}

	// Open when the last header has nothing after it but latest reminders,
	// which the encoder writes into the reply it has already opened.
	open := headerAt >= 0
	for _, m := range ms[headerAt+1:] {
		if m.role != "latest_reminder" {
			open = false
		}
	}
	return b.String(), open
}

// mergeDeepseekTurns gives the conversation the shape V4.1 renders: a run of
// user messages and tool results becomes one user turn, its parts joined by a
// blank line and each result wrapped in <tool_result>; "developer" is a system
// message.
func mergeDeepseekTurns(msgs []Message) []dsMsg {
	var out []dsMsg
	var parts []string
	for _, m := range msgs {
		role := m.Role
		if role == "developer" {
			role = "system"
		}
		if role == "user" || role == "tool" {
			part := m.Content
			if role == "tool" {
				part = "<tool_result>" + m.Content + "</tool_result>"
			}
			if n := len(out); n > 0 && out[n-1].role == "user" {
				parts = append(parts, part)
				out[n-1].content = strings.Join(parts, "\n\n")
				out[n-1].toolOnly = out[n-1].toolOnly && role == "tool"
				continue
			}
			parts = []string{part}
			out = append(out, dsMsg{role: "user", content: part, toolOnly: role == "tool"})
			continue
		}
		out = append(out, dsMsg{role: role, content: m.Content, thinking: m.Thinking, calls: m.ToolCalls})
	}
	return out
}

func dsMarker(thinking bool) string {
	if thinking {
		return thinkOpen
	}
	return thinkClose
}

// dsEffort is the reasoning budget the prompt states. DeepSeek's encoder takes
// low, high, max or a number from 1 to 100; its server also takes minimal,
// medium, xhigh and ultra, folded into the same three. Anything else is the
// default, high.
func dsEffort(effort string) string {
	switch e := strings.ToLower(strings.TrimSpace(effort)); e {
	case "low", "minimal":
		return "50"
	case "max", "ultra":
		return "100"
	default:
		if n, err := strconv.Atoi(e); err == nil && n >= 1 && n <= 100 {
			return strconv.Itoa(n)
		}
		return "75"
	}
}
