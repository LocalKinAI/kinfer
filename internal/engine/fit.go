package engine

import (
	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/llama"
)

// Fitting a conversation into a slot.
//
// Ollama never refuses a conversation for being long. It keeps the system
// prompt and the newest message and drops the oldest turns until the rest fits,
// which is why an agent session there carries on past its window — and why the
// same session here ended at "prompt is 7428 tokens but each slot holds 4096
// (… lower -slots or raise -ctx)", naming two flags an agent in a terminal
// cannot set. Every agent session outgrows its window sooner or later. One that
// loses its oldest turns carries on; one that is refused is over.
//
// So a prompt that does not fit is fitted the way Ollama fits it, with the
// differences an agent needs:
//
//   - The request that set the work going stays: the newest user turn that is
//     not riding along with a tool result. Dropped with the other old turns, it
//     leaves the model reading tool output with no idea what for.
//   - A tool call and its results go together. A result without the call that
//     asked for it is noise; a call without its result invites the model to
//     make it again.
//   - Room is left to answer in. A prompt that fills the context leaves a reply
//     of a few tokens, and a tool call cut off there is not a tool call.
//   - Turns go in chunks, at the same places every time. What the pool matches
//     next turn is what was kept this turn; dropping just enough each turn would
//     move the start of the conversation every turn, and a hybrid model — which
//     adopts pooled prompts only whole — would prefill all of it every turn.
//     Cut points that depend only on the turns before them stay put while the
//     conversation grows past them.
//
// Only when the system prompt, the request and the newest turn do not fit by
// themselves is the prompt refused.

// fitMessages returns msgs and its tokens, with the oldest turns dropped if
// that is what it takes to fit in budget tokens. When no cut fits the budget, a
// prompt shorter than limit runs whole, since dropping turns would then buy
// only a little room at the cost of all of them, and a longer one is cut to the
// least that fits under limit.
//
// render renders and tokenizes a conversation; size is roughly one message's
// tokens, used only to place the cut points. dropped counts messages left out.
func fitMessages(msgs []chat.Message, budget, limit int,
	render func([]chat.Message) ([]llama.Token, error), size func(chat.Message) int,
) (kept []chat.Message, tokens []llama.Token, dropped int, err error) {
	full, err := render(msgs)
	if err != nil {
		return nil, nil, 0, err
	}
	if len(full) <= budget {
		return msgs, full, 0, nil
	}

	order := dropOrder(msgs)
	cuts := cutPoints(msgs, order, max(budget/4, 1), size)
	without := func(groups int) []chat.Message {
		gone := make(map[int]bool)
		for _, g := range order[:groups] {
			for _, i := range g {
				gone[i] = true
			}
		}
		out := make([]chat.Message, 0, len(msgs)-len(gone))
		for i, m := range msgs {
			if !gone[i] {
				out = append(out, m)
			}
		}
		return out
	}

	rendered := map[int][]llama.Token{}
	try := func(c int) ([]llama.Token, error) {
		if t, ok := rendered[c]; ok {
			return t, nil
		}
		t, err := render(without(cuts[c]))
		rendered[c] = t
		return t, err
	}
	// smallest is the first cut whose prompt is at most bound tokens, or -1.
	smallest := func(bound int) (int, error) {
		if len(cuts) == 0 {
			return -1, nil
		}
		hi := len(cuts) - 1
		if t, err := try(hi); err != nil || len(t) > bound {
			return -1, err
		}
		lo := 0
		for lo < hi {
			mid := (lo + hi) / 2
			t, err := try(mid)
			if err != nil {
				return -1, err
			}
			if len(t) <= bound {
				hi = mid
			} else {
				lo = mid + 1
			}
		}
		return lo, nil
	}

	c, err := smallest(budget)
	if err != nil {
		return nil, nil, 0, err
	}
	if c < 0 {
		if len(full) < limit {
			return msgs, full, 0, nil
		}
		if c, err = smallest(limit - 1); err != nil {
			return nil, nil, 0, err
		}
		if c < 0 {
			least := len(full)
			if len(cuts) > 0 {
				least = len(rendered[len(cuts)-1])
			}
			return nil, nil, 0, &PromptTooLongError{Tokens: least, Limit: limit}
		}
	}
	kept = without(cuts[c])
	return kept, rendered[c], len(msgs) - len(kept), nil
}

// dropOrder groups the messages that may be dropped into turns, in the order
// they go: the turns before the request, oldest first, then the ones after it.
// System messages, the request's turn and the newest turn are never listed.
//
// A turn is a message and whatever rides with it: a tool call takes the results
// that follow it, a result takes the user text that came in the same breath —
// which is how Claude Code's reminders arrive — and a reply that calls nothing
// goes with what it answers, so no answer outlives its question.
func dropOrder(msgs []chat.Message) [][]int {
	var turns [][]int
	prev := "" // role of the previous non-system message
	for i, m := range msgs {
		if m.Role == "system" {
			continue
		}
		rides := len(turns) > 0 && (m.Role == "tool" ||
			(m.Role == "user" && prev == "tool") ||
			(m.Role == "assistant" && len(m.ToolCalls) == 0))
		if rides {
			turns[len(turns)-1] = append(turns[len(turns)-1], i)
		} else {
			turns = append(turns, []int{i})
		}
		prev = m.Role
	}
	if len(turns) == 0 {
		return nil
	}

	// The request is the newest turn a person started; failing that, the newest
	// with any user text in it.
	request := -1
	for t := len(turns) - 1; t >= 0 && request < 0; t-- {
		if msgs[turns[t][0]].Role == "user" {
			request = t
		}
	}
	for t := len(turns) - 1; t >= 0 && request < 0; t-- {
		for _, i := range turns[t] {
			if msgs[i].Role == "user" {
				request = t
				break
			}
		}
	}

	newest := len(turns) - 1
	var order [][]int
	for t := 0; t < len(turns); t++ {
		if t != request && t != newest && (request < 0 || t < request) {
			order = append(order, turns[t])
		}
	}
	for t := request + 1; request >= 0 && t < newest; t++ {
		order = append(order, turns[t])
	}
	return order
}

// cutPoints are the numbers of leading groups of order that may be dropped
// together. A cut is placed where the running size first reaches each multiple
// of chunk, so it depends only on the turns before it and lands in the same
// place when the same conversation comes back a turn longer. The last cut drops
// every group.
func cutPoints(msgs []chat.Message, order [][]int, chunk int, size func(chat.Message) int) []int {
	var cuts []int
	total, next := 0, chunk
	for g, turn := range order {
		for _, i := range turn {
			total += size(msgs[i])
		}
		if total >= next || g == len(order)-1 {
			cuts = append(cuts, g+1)
			for next <= total {
				next += chunk
			}
		}
	}
	return cuts
}
