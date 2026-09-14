package tools

import "strings"

// CallSplitter separates the prose of a streamed reply from its tool calls.
//
// A tool call is JSON split across many tokens, and handing a client the
// fragments would hand it broken syntax — which is why replies with functions
// on the table used to be buffered whole and delivered once. That kept the
// syntax intact and cost the caller every sign of life until the end: a fleet
// agent whose idle watchdog counts streamed chunks as progress saw none for the
// whole of a tool-bearing reply, and cut it off at sixty seconds as a hang.
// Ollama streams the prose and delivers the calls in the final message, so a
// client that reads Ollama already expects exactly that shape.
//
// So: prose is passed through as it arrives, a possible start of a tag is held
// back until it resolves, and everything from the first open tag on is kept
// out of the stream. The full text is parsed for calls at the end, as before;
// this only decides what is safe to show while it is still being written.
type CallSplitter struct {
	pending string
	inCall  bool
	held    strings.Builder
}

// openMarkers are the ways a call can begin. Both conventions Detect knows
// open with <tool_call>; the XML one is also recognisable by its first element,
// which some models emit without the outer tag.
var openMarkers = []string{openTag, "<function="}

// Next takes the next fragment and returns whatever prose can be shown now.
func (s *CallSplitter) Next(frag string) string {
	if s.inCall {
		s.held.WriteString(frag)
		return ""
	}
	s.pending += frag
	for _, m := range openMarkers {
		if i := strings.Index(s.pending, m); i >= 0 {
			prose := s.pending[:i]
			s.held.WriteString(s.pending[i:])
			s.pending = ""
			s.inCall = true
			return prose
		}
	}
	// Hold back only as much as could still turn out to be a marker.
	keep := 0
	for _, m := range openMarkers {
		if n := partialSuffix(s.pending, m); n > keep {
			keep = n
		}
	}
	prose := s.pending[:len(s.pending)-keep]
	s.pending = s.pending[len(s.pending)-keep:]
	return prose
}

// Flush returns what is still worth showing once the reply has ended: a held
// partial marker that never became one, and any prose after the last call.
// A call that was cut off unterminated is shown too — Parse keeps it visible
// for the same reason, and hiding it would silently lose the reply's tail.
func (s *CallSplitter) Flush() string {
	if !s.inCall {
		p := s.pending
		s.pending = ""
		return p
	}
	held := s.held.String()
	s.held.Reset()
	if j := strings.LastIndex(held, closeTag); j >= 0 {
		return held[j+len(closeTag):]
	}
	return held
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
