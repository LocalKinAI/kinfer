package chat

import (
	"strings"
	"testing"
)

func TestSplitThinking(t *testing.T) {
	cases := []struct {
		name             string
		in               string
		thinking, answer string
	}{
		{"typical", "<think>\nwork it out\n</think>\n4", "work it out", "4"},
		{"no block", "just an answer", "", "just an answer"},
		{"prose before the block", "hm <think>w</think> 4", "w", "hm  4"},
		{"empty", "", "", ""},
		// Budget exhausted mid-thought: all thinking, no answer. Reporting an
		// answer here would invent one.
		{"unterminated", "<think>still working", "still working", ""},
	}
	for _, c := range cases {
		th, ct := SplitThinking(c.in)
		if th != c.thinking || ct != c.answer {
			t.Errorf("%s: SplitThinking(%q) = (%q, %q), want (%q, %q)",
				c.name, c.in, th, ct, c.thinking, c.answer)
		}
	}
}

// TestThinkSplitterMatchesTheWholeReplySplit: streaming a reply fragment by
// fragment must produce exactly what splitting it whole produces, whatever the
// tags are cut across.
func TestThinkSplitterMatchesTheWholeReplySplit(t *testing.T) {
	reply := "<think>\nreasoning here\n</think>\nThe answer is 4."
	wantTh, wantCt := SplitThinking(reply)

	for _, size := range []int{1, 2, 3, 5, 7, 13} {
		var s ThinkSplitter
		var th, ct strings.Builder
		for i := 0; i < len(reply); i += size {
			j := min(i+size, len(reply))
			a, b := s.Next(reply[i:j])
			th.WriteString(a)
			ct.WriteString(b)
		}
		a, b := s.Flush()
		th.WriteString(a)
		ct.WriteString(b)

		if strings.TrimSpace(th.String()) != wantTh {
			t.Errorf("chunks of %d: thinking = %q, want %q", size, th.String(), wantTh)
		}
		if strings.TrimSpace(ct.String()) != wantCt {
			t.Errorf("chunks of %d: content = %q, want %q", size, ct.String(), wantCt)
		}
	}
}

func TestThinkSplitterPassesPlainRepliesStraightThrough(t *testing.T) {
	var s ThinkSplitter
	_, ct := s.Next("Hello")
	if ct != "Hello" {
		t.Errorf("a reply with no tags was held back: %q", ct)
	}
}

// A model that quotes the opening tag inside its answer is not starting to
// think again.
func TestThinkSplitterOnlyHonoursTheFirstTag(t *testing.T) {
	var s ThinkSplitter
	var ct strings.Builder
	for _, frag := range []string{"<think>a</think>", "use <think>", " in XML"} {
		_, c := s.Next(frag)
		ct.WriteString(c)
	}
	_, c := s.Flush()
	ct.WriteString(c)
	if got := ct.String(); !strings.Contains(got, "<think>") {
		t.Errorf("a quoted tag was swallowed: %q", got)
	}
}
