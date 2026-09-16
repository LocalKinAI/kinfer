package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// reply is one finished generation taken apart: what the model thought, what
// it said, which functions it asked for, and the measurements.
type reply struct {
	Thinking string
	Content  string
	Calls    []tools.Call
	Stats    engine.Stats
}

// generate runs a conversation, handing thinking and prose to the callbacks as
// they are written — call syntax held back, exactly as the OpenAI dialect
// streams it — and parses the whole reply once it is complete.
//
// The Anthropic and Responses dialects are two more ways of writing down the
// same events. Sharing this is what keeps them from each re-deciding where a
// think block ends or a call begins, and from disagreeing about it.
func generate(ctx context.Context, eng generator, msgs []chat.Message, p engine.GenParams,
	onThinking, onContent func(string)) (reply, error) {
	withTools := len(p.Tools) > 0
	var split chat.ThinkSplitter
	var calls tools.CallSplitter
	emit := func(thinking, content string) {
		if thinking != "" && onThinking != nil {
			onThinking(thinking)
		}
		if withTools {
			content = calls.Next(content)
		}
		if content != "" && onContent != nil {
			onContent(content)
		}
	}

	text, st, err := eng.ChatFull(ctx, msgs, p, func(frag string) { emit(split.Next(frag)) })
	if err != nil {
		return reply{Stats: st}, err
	}
	emit(split.Flush())
	if withTools {
		if rest := calls.Flush(); rest != "" && onContent != nil {
			onContent(rest)
		}
	}

	thinking, answer := chat.SplitThinking(text)
	r := reply{Thinking: thinking, Content: answer, Stats: st}
	if withTools {
		r.Content, r.Calls = eng.ToolFormat().Parse(answer)
	}
	return r, nil
}

// jsonObject returns raw when it is a JSON object, and {} when it is not. Both
// agent dialects put tool input on the wire as an object; a model that wrote
// anything else wrote nothing a client can hand to a function.
func jsonObject(raw json.RawMessage) json.RawMessage {
	t := bytes.TrimSpace(raw)
	if len(t) > 0 && t[0] == '{' && json.Valid(t) {
		return t
	}
	return json.RawMessage(`{}`)
}

var idCounter atomic.Uint64

// newID makes an identifier in the prefix_… form both dialects use. It only has
// to be unique within this process: a client uses it to match a call to its
// result inside one conversation.
func newID(prefix string) string {
	return fmt.Sprintf("%s_%x%x", prefix, time.Now().UnixNano(), idCounter.Add(1))
}

// toolChoiceNone reports a caller that declared tools and then said not to use
// any: {"type":"none"} in the Anthropic dialect, "none" in the Responses one.
// Leaving the tools out of the prompt is the only way a local model reliably
// does not call them.
func toolChoiceNone(raw json.RawMessage) bool {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s == "none"
	}
	var o struct {
		Type string `json:"type"`
	}
	return json.Unmarshal(raw, &o) == nil && o.Type == "none"
}

// omittedAttachment is what an image or a document becomes in the prompt of a
// model that reads text only: said, not silently dropped, so the model does not
// answer as if nothing had been attached.
func omittedAttachment(kind string) string {
	return "[" + kind + " omitted: this model reads text only]"
}
