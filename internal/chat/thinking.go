package chat

import "strings"

// Reasoning models write their working out before they answer, wrapped in
// <think> tags:
//
//	<think>
//	The user asks 2+2. They want only the number.
//	</think>
//	4
//
// That text is not the reply. Left in place it reaches the caller as part of
// the answer, and a client that shows replies verbatim shows the model talking
// to itself. Worse for an agent fleet: a tool call decided on inside the
// thinking block is not a tool call, so a caller waiting for one sees prose.
//
// Ollama separates the two, returning the working out in message.thinking and
// the answer in message.content. kinfer does the same.

const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
)

// SplitThinking separates a reasoning model's working out from its answer.
//
// A reply with no thinking block is returned unchanged, so this is safe to run
// over every model's output. A block that opens and never closes means the
// model was still thinking when its budget ran out: all of it is thinking, and
// the answer is empty — which is the honest result, and what Ollama reports
// too.
func SplitThinking(reply string) (thinking, content string) {
	i := strings.Index(reply, thinkOpen)
	if i < 0 {
		return "", reply
	}

	rest := reply[i+len(thinkOpen):]
	j := strings.Index(rest, thinkClose)
	if j < 0 {
		return strings.TrimSpace(rest), ""
	}

	thinking = strings.TrimSpace(rest[:j])
	content = reply[:i] + rest[j+len(thinkClose):]
	return thinking, strings.TrimSpace(content)
}

// ThinkSplitter routes streamed fragments to the thinking or the answer as
// they arrive, since neither is complete until the tags have been seen.
//
// It holds back only what could still turn out to be part of a tag — never a
// whole reply — so streaming stays responsive.
type ThinkSplitter struct {
	inThink bool
	seen    bool // an opening tag has been found, so no later one can start
	pending string
}

// Next splits one fragment. Either return may be empty.
func (s *ThinkSplitter) Next(frag string) (thinking, content string) {
	s.pending += frag

	var th, ct strings.Builder
	for {
		if s.inThink {
			i := strings.Index(s.pending, thinkClose)
			if i < 0 {
				// Emit everything that cannot be the start of the closing tag.
				keep := partialSuffix(s.pending, thinkClose)
				th.WriteString(s.pending[:len(s.pending)-keep])
				s.pending = s.pending[len(s.pending)-keep:]
				break
			}
			th.WriteString(s.pending[:i])
			s.pending = s.pending[i+len(thinkClose):]
			s.inThink = false
			continue
		}

		// Only the first tag counts. A model quoting "<think>" in its answer
		// should not be read as starting to think again.
		if !s.seen {
			if i := strings.Index(s.pending, thinkOpen); i >= 0 {
				ct.WriteString(s.pending[:i])
				s.pending = s.pending[i+len(thinkOpen):]
				s.inThink, s.seen = true, true
				continue
			}
			keep := partialSuffix(s.pending, thinkOpen)
			ct.WriteString(s.pending[:len(s.pending)-keep])
			s.pending = s.pending[len(s.pending)-keep:]
			break
		}

		ct.WriteString(s.pending)
		s.pending = ""
		break
	}
	return th.String(), ct.String()
}

// Flush returns whatever was held back, once no more fragments are coming.
func (s *ThinkSplitter) Flush() (thinking, content string) {
	out := s.pending
	s.pending = ""
	if s.inThink {
		return out, ""
	}
	return "", out
}

// partialSuffix reports how many trailing bytes of s could still grow into tag.
func partialSuffix(s, tag string) int {
	max := len(tag) - 1
	if len(s) < max {
		max = len(s)
	}
	for n := max; n > 0; n-- {
		if strings.HasPrefix(tag, s[len(s)-n:]) {
			return n
		}
	}
	return 0
}
