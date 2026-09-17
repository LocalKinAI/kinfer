package engine

import (
	"errors"
	"strings"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// A stand-in template: every message costs its text plus four tokens of
// markup, and the prompt ends with two more for the reply's opening.
func fakeRender(msgs []chat.Message) ([]llama.Token, error) {
	n := 2
	for _, m := range msgs {
		n += 4 + fakeSize(m)
	}
	return make([]llama.Token, n), nil
}

func fakeSize(m chat.Message) int {
	n := len(m.Content)
	for _, c := range m.ToolCalls {
		n += len(c.Name) + len(c.Arguments)
	}
	return n
}

func text(role string, n int) chat.Message {
	return chat.Message{Role: role, Content: strings.Repeat("x", n)}
}

func call(name string) chat.Message {
	return chat.Message{Role: "assistant", ToolCalls: []tools.Call{{Name: name, Arguments: []byte(`{}`)}}}
}

func roles(msgs []chat.Message) string {
	var r []string
	for _, m := range msgs {
		r = append(r, m.Role)
	}
	return strings.Join(r, ",")
}

// Nothing is dropped from a conversation that fits.
func TestFitLeavesAConversationThatFitsAlone(t *testing.T) {
	msgs := []chat.Message{text("system", 100), text("user", 50)}
	kept, tokens, dropped, err := fitMessages(msgs, 1000, 2000, fakeRender, fakeSize)
	if err != nil || dropped != 0 || len(kept) != 2 || len(tokens) != 160 {
		t.Errorf("got %d kept, %d dropped, %d tokens, err %v; want both, untouched", len(kept), dropped, len(tokens), err)
	}
}

// An agent's session past its window: the system prompt, the task and the
// newest tool exchange stay, the oldest exchanges go — each call with its
// result, never a result on its own.
func TestFitKeepsSystemRequestAndNewestTurn(t *testing.T) {
	task := text("user", 50)
	task.Content = "fix the build"
	msgs := []chat.Message{
		text("system", 100), task,
		call("read"), text("tool", 1000),
		call("read"), text("tool", 1000),
		call("read"), text("tool", 1000),
		call("edit"), text("tool", 300),
	}
	kept, tokens, dropped, err := fitMessages(msgs, 1800, 4096, fakeRender, fakeSize)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) > 1800 || dropped == 0 {
		t.Fatalf("%d tokens after dropping %d; want it under the 1800 budget", len(tokens), dropped)
	}
	if kept[0].Role != "system" || kept[1].Content != "fix the build" {
		t.Errorf("kept %s; the system prompt and the task must lead", roles(kept))
	}
	if last := kept[len(kept)-1]; last.Role != "tool" || len(last.Content) != 300 {
		t.Errorf("the newest result was dropped: kept %s", roles(kept))
	}
	for i := 1; i < len(kept); i++ {
		if kept[i].Role == "tool" && kept[i-1].Role != "assistant" && kept[i-1].Role != "tool" {
			t.Errorf("a result lost its call: kept %s", roles(kept))
		}
	}
}

// Turns before the latest request go before the work that request started.
func TestFitDropsTurnsOlderThanTheRequestFirst(t *testing.T) {
	msgs := []chat.Message{
		text("system", 100),
		text("user", 400), text("assistant", 500), // an earlier question, answered
		text("user", 50), // the request
		call("grep"), text("tool", 500),
		call("read"), text("tool", 200),
	}
	kept, _, dropped, err := fitMessages(msgs, 1500, 4096, fakeRender, fakeSize)
	if err != nil {
		t.Fatal(err)
	}
	if dropped != 2 || roles(kept) != "system,user,assistant,tool,assistant,tool" {
		t.Errorf("dropped %d, kept %s; want the earlier question and answer gone, the work kept", dropped, roles(kept))
	}
}

// Claude Code sends reminders in the same message as a tool result. That text
// belongs to the result's turn: it is not a new request, and it goes with it.
func TestFitTreatsTextBesideAToolResultAsPartOfIt(t *testing.T) {
	msgs := []chat.Message{
		text("system", 100), text("user", 50),
		call("read"), text("tool", 1500), text("user", 30),
		call("read"), text("tool", 100), text("user", 30),
	}
	kept, _, dropped, err := fitMessages(msgs, 800, 4096, fakeRender, fakeSize)
	if err != nil {
		t.Fatal(err)
	}
	if dropped != 3 || roles(kept) != "system,user,assistant,tool,user" {
		t.Errorf("dropped %d, kept %s; want the middle call, its result and its reminder gone together", dropped, roles(kept))
	}
}

// The pool matches next turn what was kept this turn, so a conversation that
// grows by a turn must be cut where it was cut before, not one turn later —
// and when the cut does move, it must move far enough to stay put again.
func TestFitCutsInTheSamePlaceATurnLater(t *testing.T) {
	const budget = 4000
	msgs := []chat.Message{text("system", 100), text("user", 50)}
	for i := 0; i < 12; i++ {
		msgs = append(msgs, call("read"), text("tool", 400))
	}
	kept, _, _, err := fitMessages(msgs, budget, 8192, fakeRender, fakeSize)
	if err != nil {
		t.Fatal(err)
	}
	cut := len(msgs) - len(kept)
	if cut == 0 {
		t.Fatal("nothing was dropped; the test needs a conversation over budget")
	}
	for turn := 1; ; turn++ {
		msgs = append(msgs, call("read"), text("tool", 20))
		kept, tokens, _, err := fitMessages(msgs, budget, 8192, fakeRender, fakeSize)
		if err != nil {
			t.Fatal(err)
		}
		got := len(msgs) - len(kept)
		if got == cut {
			continue
		}
		if turn == 1 {
			t.Errorf("a turn of 34 tokens moved the cut from %d to %d messages", cut, got)
		}
		if len(tokens) > budget-budget/8 {
			t.Errorf("turn %d: the cut moved but left %d of %d tokens — it will move again next turn", turn, len(tokens), budget)
		}
		break
	}
}

// Refused only when what can never be dropped does not fit, and the error
// says how long that part is.
func TestFitRefusesOnlyWhatCannotFitAtAll(t *testing.T) {
	msgs := []chat.Message{text("system", 5000), text("user", 20), text("assistant", 300), text("user", 100)}
	_, _, _, err := fitMessages(msgs, 3584, 4096, fakeRender, fakeSize)
	var big *PromptTooLongError
	if !errors.As(err, &big) {
		t.Fatalf("err = %v; want a PromptTooLongError", err)
	}
	if big.Tokens != 2+4+5000+4+100 || big.Limit != 4096 {
		t.Errorf("reported %d of %d; want the system prompt and request alone, 5110 of 4096", big.Tokens, big.Limit)
	}
}

// A prompt over the reply budget but inside the slot, with nothing that could
// go, runs as it always did.
func TestFitRunsAFullPromptWhenNothingCanGo(t *testing.T) {
	msgs := []chat.Message{text("system", 3700), text("user", 50)}
	kept, _, dropped, err := fitMessages(msgs, 3584, 4096, fakeRender, fakeSize)
	if err != nil || dropped != 0 || len(kept) != 2 {
		t.Errorf("got %d kept, %d dropped, err %v; want it run whole", len(kept), dropped, err)
	}
}

func TestReplyRoom(t *testing.T) {
	for _, c := range []struct{ ctx, max, want int }{
		{32768, 32000, 4096}, // Claude Code's ask, capped at an eighth
		{32768, 512, 512},
		{4096, 0, 512},
	} {
		if got := replyRoom(c.ctx, c.max); got != c.want {
			t.Errorf("replyRoom(%d, %d) = %d, want %d", c.ctx, c.max, got, c.want)
		}
	}
}
