package engine

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/llama"
)

// ImageMarker is where a picture goes in a message's text: one for each of
// its Images, in order. A dialect whose pictures come apart from the text —
// Ollama's "images" — puts them before it.
const ImageMarker = llama.MediaMarker

// ErrNoPictures is a picture sent to a model with no projector to read it.
// It is refused, as Ollama refuses it, rather than said in words: told that
// a picture was there, Qwen's 4B described one it had not seen.
var ErrNoPictures = errors.New("this model reads no pictures: it has no projector (mmproj)")

// PictureError is a picture the model cannot read — not an image, one more or
// fewer than the places for them, or any for a model with no projector — the
// caller's to fix.
type PictureError struct{ Err error }

func (e PictureError) Error() string { return e.Err.Error() }
func (e PictureError) Unwrap() error { return e.Err }

// showsPictures says whether any message carries a picture.
func showsPictures(msgs []chat.Message) bool {
	for _, m := range msgs {
		if len(m.Images) > 0 {
			return true
		}
	}
	return false
}

// prepareMedia cuts a prompt with pictures into libmtmd's text and picture
// chunks as it reaches the head of the queue: the cells its pictures take
// decide the layout it runs whole in, as a text prompt's tokens do.
func (s *scheduler) prepareMedia(j *job) bool {
	text := s.renderer(j)(j.msgs, j.params.Tools)
	if r := j.params.Resume; r != "" {
		if s.tpl.OpensThink(j.params.thinking()) {
			r = strings.TrimPrefix(r, "<think>")
		}
		text += r
	}
	var pictures [][]byte
	for _, m := range j.msgs {
		pictures = append(pictures, m.Images...)
	}
	p, err := llama.TokenizeMedia(s.mtmd, text, pictures)
	if err != nil {
		j.finish(PictureError{err})
		return false
	}
	j.media = p
	j.want = len(s.slots)
	if s.flex != nil {
		j.want = s.flex.want(p.Tokens, j.params.MaxTokens)
	}
	return true
}

// startMedia reads a prompt with pictures into sl whole, at once: libmtmd
// encodes each picture and reads it with decodes of its own, so the other
// conversations wait the second or so that takes. The prefix pool, which knows
// a prompt by its tokens, neither lends to such a prompt nor keeps it, and an
// older turn is not dropped to make it fit.
func (s *scheduler) startMedia(sl *slot, j *job) {
	p := j.media
	j.media = nil
	defer p.Free()

	room := s.ctxPerSeq - p.Tokens
	if room < replyRoom(s.ctxPerSeq, j.params.MaxTokens) {
		slots := len(s.slots)
		if s.flex != nil {
			slots = 1
		}
		j.finish(&PromptTooLongError{Tokens: p.Tokens, Limit: s.ctxPerSeq, Slots: slots})
		return
	}
	// The reply is bounded by the cells left, not the positions: a picture
	// laid out in two dimensions moves the position less than it fills cells.
	maxGen := j.params.MaxTokens
	if maxGen <= 0 || maxGen > room {
		maxGen = room
	}

	s.forget(sl.seq)
	// libmtmd reads the prompt with decodes of its own, which the MTP head
	// does not follow, so this conversation is not drafted for.
	s.draft.untrack(sl.seq)
	start := time.Now()
	next, err := llama.EvalMedia(s.mtmd, s.lctx, p, sl.seq, s.batch.Cap())
	if err != nil {
		s.forget(sl.seq)
		var de *llama.DecodeError
		if errors.As(err, &de) && de.Fatal() {
			// The context is finished, as a failed step would leave it.
			s.fatal = err
			j.finish(fmt.Errorf("%w: %w", ErrInterrupted, err))
			return
		}
		j.finish(fmt.Errorf("read the pictures: %w", err))
		return
	}
	if s.debug {
		s.admitted++
		s.promptTok += p.Tokens
	}
	s.occupy(sl, j, nil, 0, nil, maxGen)
	sl.media = p.Tokens
	sl.nPast = next
	sl.admitted = start
	// The prompt's last logits wait at index -1: the first token comes from
	// them now, as harvest takes every later one from a step's.
	s.harvest(sl, time.Now())
}
