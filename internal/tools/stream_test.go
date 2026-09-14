package tools

import (
	"strings"
	"testing"
)

func feed(frags ...string) (streamed, tail string) {
	var s CallSplitter
	var b strings.Builder
	for _, f := range frags {
		b.WriteString(s.Next(f))
	}
	return b.String(), s.Flush()
}

// Prose before a call streams; the call itself never does. What the client
// sees live plus what Flush returns must be exactly the reply's prose.
func TestCallSplitterStreamsProseAndHoldsTheCall(t *testing.T) {
	streamed, tail := feed("Let me check.", " <tool_c", "all>\n{\"name\":\"x\"}\n</tool_call>", " Done.")
	if streamed != "Let me check. " {
		t.Errorf("streamed %q, want the prose before the call (space included) and nothing of the call", streamed)
	}
	if tail != " Done." {
		t.Errorf("flush %q, want the prose after the call", tail)
	}
}

// A fragment that ends in what might be the start of a tag is held back until
// it resolves, and released untouched when it turns out to be plain text.
func TestCallSplitterHoldsBackAPossibleTagThenReleasesIt(t *testing.T) {
	var s CallSplitter
	if got := s.Next("temperature <"); got != "temperature " {
		t.Errorf("first fragment streamed %q, want the '<' held back", got)
	}
	if got := s.Next("40 is"); got != "<40 is" {
		t.Errorf("second fragment streamed %q, want the held '<' released with it", got)
	}
	if got := s.Flush(); got != "" {
		t.Errorf("flush %q, want nothing left", got)
	}
}

// With no call at all, every byte streams and Flush releases whatever a
// trailing partial marker held.
func TestCallSplitterPassesPlainRepliesThrough(t *testing.T) {
	streamed, tail := feed("All ", "good", " here <")
	if streamed+tail != "All good here <" {
		t.Errorf("got %q + %q, want the whole reply", streamed, tail)
	}
}

// A call cut off mid-JSON is shown rather than swallowed — Parse keeps it
// visible, and a stream must not lose what a buffered reply would have shown.
func TestCallSplitterShowsAnUnterminatedCall(t *testing.T) {
	_, tail := feed("Sure. ", "<tool_call>{\"name\":")
	if !strings.Contains(tail, "<tool_call>") {
		t.Errorf("flush %q, want the unterminated call kept visible", tail)
	}
}

// The XML convention can begin with its first element and no outer tag.
func TestCallSplitterRecognisesTheXMLOpener(t *testing.T) {
	streamed, _ := feed("Checking. <function=get_weather><parameter=city>Berlin")
	if streamed != "Checking. " {
		t.Errorf("streamed %q, want only the prose before <function=", streamed)
	}
}
