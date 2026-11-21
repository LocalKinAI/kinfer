package chat

import (
	"regexp"
	"strconv"
	"strings"
)

// What a template does with reasoning, read from its own text — as
// noThinkSuffix and thinkOnSuffix read its switches, and for the same reason:
// llama_chat_apply_template takes {role, content} and nothing else, so an
// earlier turn's reasoning and a reasoning effort never reach the branches the
// template has for them. A client sent "thinking" or "reasoning_content" back
// and it was dropped; it asked for effort low, medium or xhigh and the prompt
// was the same to the token (Qwen3.8-27B, found by a client's own measurements,
// 2026-11-20). DeepSeek V4.1 was the only model that took either, having a
// renderer of its own.
//
// Two things are reproduced here, each exactly as the template writes it:
//
//   - an earlier assistant turn's reasoning, as the think block the turn
//     began with: '<think>\n' + reasoning + '\n</think>\n\n' + content
//   - the instruction a reasoning effort puts at the head of the system
//     prompt, in the template's own words for that level

// thinkBlockTmpl is how Qwen-lineage templates write an assistant turn's
// reasoning back into the conversation.
var thinkBlockTmpl = regexp.MustCompile(`'\\n<think>\\n'\s*\+\s*reasoning_content\s*\+\s*'\\n</think>\\n\\n'`)

// rendersReasoning reports that the template gives an assistant turn's
// reasoning_content back to the model.
func rendersReasoning(tmpl string) bool {
	return strings.Contains(tmpl, "reasoning_content") && thinkBlockTmpl.MatchString(tmpl)
}

// effortInstr finds each level's instruction:
//
//	{%- if resolved_reasoning_effort == 'xhigh' %}
//	    {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. …' %}
var effortInstr = regexp.MustCompile(`reasoning_effort\s*==\s*'([a-z]+)'\s*%\}\s*\{%-?\s*set\s+reasoning_instructions\s*=\s*'((?:[^'\\]|\\.)*)'`)

// effortLevels finds the levels the template accepts:
//
//	{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
var effortLevels = regexp.MustCompile(`reasoning_effort\s+not\s+in\s+\(([^)]*)\)`)

// effortAlias finds a level the template reads as another:
//
//	{%- if resolved_reasoning_effort == 'high' %}{%- set resolved_reasoning_effort = 'xhigh' %}
var effortAlias = regexp.MustCompile(`reasoning_effort\s*==\s*'([a-z]+)'\s*%\}\s*\{%-?\s*set\s+resolved_reasoning_effort\s*=\s*'([a-z]+)'`)

// efforts is what a template says about reasoning effort: the levels it
// takes, and the instruction each one writes ("" for a level that writes
// none, as Qwen3.8's medium).
type efforts struct {
	levels map[string]string
	alias  map[string]string
}

// readEfforts reads the template's reasoning-effort levels, or nil for a
// template that has none.
func readEfforts(tmpl string) *efforts {
	if !strings.Contains(tmpl, "reasoning_effort") {
		return nil
	}
	e := &efforts{levels: map[string]string{}, alias: map[string]string{}}
	if m := effortLevels.FindStringSubmatch(tmpl); m != nil {
		for _, lv := range strings.Split(m[1], ",") {
			if lv = strings.Trim(strings.TrimSpace(lv), `'"`); lv != "" {
				e.levels[lv] = ""
			}
		}
	}
	for _, m := range effortInstr.FindAllStringSubmatch(tmpl, -1) {
		e.levels[m[1]] = unquoteJinja(m[2])
	}
	for _, m := range effortAlias.FindAllStringSubmatch(tmpl, -1) {
		e.alias[m[1]] = m[2]
	}
	if len(e.levels) == 0 {
		return nil
	}
	return e
}

// unquoteJinja undoes the escapes of a single-quoted Jinja string.
func unquoteJinja(s string) string {
	return strings.NewReplacer(`\n`, "\n", `\'`, `'`, `\"`, `"`, `\\`, `\`).Replace(s)
}

// instruction is what the template writes for the effort a caller named, and
// whether the template knows the level at all. Callers name levels in several
// vocabularies — OpenAI's (minimal, low, medium, high), Ollama's think levels,
// kinfer's own max and 1 to 100 — and a template has its own: each is read as
// the nearest level the template takes.
func (e *efforts) instruction(effort string) (string, bool) {
	effort = strings.ToLower(strings.TrimSpace(effort))
	if to, ok := e.alias[effort]; ok {
		effort = to
	}
	if s, ok := e.levels[effort]; ok {
		return s, true
	}
	var want []string
	switch effort {
	case "minimal":
		want = []string{"low"}
	case "high", "max":
		want = []string{"xhigh", "high", "max"}
	case "xhigh":
		want = []string{"high", "max"}
	default:
		n, err := strconv.Atoi(effort)
		if err != nil || n < 1 || n > 100 {
			return "", false
		}
		switch {
		case n <= 33:
			want = []string{"low"}
		case n <= 66:
			want = []string{"medium"}
		default:
			want = []string{"xhigh", "high", "max"}
		}
	}
	for _, lv := range want {
		if s, ok := e.levels[lv]; ok {
			return s, true
		}
	}
	return "", false
}

// withReasoning is the conversation as the template's reasoning branches
// would have it: the effort's instruction at the head of the system prompt,
// and earlier reasoning back in the assistant turns it belongs to.
//
// Which turns get their reasoning back: those since the last question the
// user asked, the steps of the tool loop the reply continues, where the model
// reads its own working out before the next call (the template's
// preserve_thinking false, and what Qwen3 templates before 3.8 always did).
// With th.KeepAll, every turn — Qwen3.8's default, at a price: a reply's
// reasoning ran to 10–17 thousand tokens in the measurements that found this,
// and two such turns fill a 32k window.
//
// A caller that names no effort gets no instruction, as before; one that sends
// no reasoning back gets its turns as before.
func (t *Template) withReasoning(msgs []Message, th Thinking) []Message {
	var instr string
	if t.efforts != nil && !th.Off && th.Effort != "" {
		instr, _ = t.efforts.instruction(th.Effort)
	}
	back := false
	if t.rendersReasoning {
		for _, m := range msgs {
			if m.Role == "assistant" && strings.TrimSpace(m.Thinking) != "" {
				back = true
				break
			}
		}
	}
	if instr == "" && !back {
		return msgs
	}

	out := make([]Message, 0, len(msgs)+1)
	out = append(out, msgs...)
	if back {
		last := lastQuery(out)
		for i := range out {
			m := &out[i]
			if m.Role != "assistant" || (!th.KeepAll && i <= last) {
				continue
			}
			if r := strings.TrimSpace(m.Thinking); r != "" {
				m.Content = "<think>\n" + r + "\n</think>\n\n" + strings.TrimSpace(m.Content)
			}
		}
	}
	if instr != "" {
		if len(out) > 0 && out[0].Role == "system" {
			if c := strings.TrimSpace(out[0].Content); c != "" {
				out[0].Content = instr + "\n\n" + c
			} else {
				out[0].Content = instr
			}
		} else {
			out = append([]Message{{Role: "system", Content: instr}}, out...)
		}
	}
	return out
}

// lastQuery is the index of the last thing the user asked: the last user turn
// that is not a tool's result handed back. -1 when there is none.
func lastQuery(msgs []Message) int {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != "user" {
			continue
		}
		c := strings.TrimSpace(msgs[i].Content)
		if !(strings.HasPrefix(c, "<tool_response>") && strings.HasSuffix(c, "</tool_response>")) {
			return i
		}
	}
	return -1
}
