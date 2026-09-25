package engine

import (
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/llama"
)

// Slots that follow the prompt.
//
// llama.cpp splits a context evenly between its sequences, so the longest
// prompt a server takes is its context divided by its slot count, and the trade
// between many conversations and long ones was made once, at startup, by
// whoever wrote the flags. On the box that was -ctx 16384 -slots 4: four
// conversations at once, and a 44,555-token prompt refused with a 413 while the
// cache held 65,536 tokens in all. Giving every slot the whole 65,536 was tried
// and does not fit: 8.2 GiB of cache where the weights leave 4.4, and the first
// decode died with kIOGPUCommandBufferCallbackErrorOutOfMemory. Ollama answers
// the same question with one slot, sized once.
//
// So the cache keeps its total and changes its shape. The layout it was loaded
// with — four slots of 16384 there — is home. A prompt that will not run whole
// in a home slot waits for the slots to drain, and the context is rebuilt as the
// most slots that still hold it: 3 x 21760, 2 x 32768 or 1 x 65536. The weights
// stay where they are; only the cache is reallocated.
//
// The request at the head of the queue decides, and nothing behind it is
// admitted until it has its slot. A long prompt is not starved by a stream of
// short ones arriving after it, and the way back works the same: once nothing
// running needs the longer slots, the next request that wants more of them
// stops admission, the slots drain, and home is rebuilt. Until then a short
// request that finds a free longer slot takes it rather than wait.

// layout is one way of dividing the context between conversations.
type layout struct {
	slots, prefix, perSeq int
	shared                bool
}

func (l layout) cells() int {
	return Plan{Slots: l.slots, Prefix: l.prefix, SharedPool: l.shared}.Cells()
}

func (l layout) String() string { return fmt.Sprintf("%d x %d", l.slots, l.perSeq) }

// reshapeSlack is how much less headroom than home a rebuilt layout may leave.
// The layouts hold the same number of tokens and differ only in what scales
// with the sequence count, the recurrent state, and with the sequence's length,
// the compute buffer. Estimated from the box's own figures, the three layouts
// come within about 0.1 GiB of home. The slack allows for that and not much
// more.
const reshapeSlack = 256 << 20

// flex re-slices one context between conversations; see the top of this file.
// Only the scheduler goroutine uses it.
type flex struct {
	home  layout
	total int // tokens of conversation in home: slots x perSeq
	most  int // the longest one conversation may be: total, or the trained context

	// build creates the context for a layout, or explains why it would not. It
	// refuses a layout that leaves less memory than home did, rather than hand
	// back a context whose first batch fails for want of it, and reports what
	// the one it built left free (0 when there is no accelerator to ask). The
	// engine supplies it, because the engine holds the model and the budget.
	build func(layout) (llama.Context, uint64, error)

	// refused are the slot counts build turned down. They are not tried again:
	// each attempt frees the cache and builds it twice while every request
	// waits, and the memory that was short is still short.
	refused map[int]bool
}

// newFlex returns nil when no layout would give a conversation more context
// than home does: home has one slot, or the model was trained for no more.
func newFlex(home layout, trained int, build func(layout) (llama.Context, uint64, error)) *flex {
	f := &flex{home: home, total: home.slots * home.perSeq, build: build, refused: map[int]bool{}}
	f.most = f.total
	if trained > 0 {
		f.most = min(f.most, trained)
	}
	if home.slots <= 1 || f.per(1) <= home.perSeq {
		return nil
	}
	return f
}

// per is the context each of n slots gets: an equal share of the total, capped
// at what the model was trained for, rounded down to a multiple of 256, which
// is how llama.cpp pads a context.
func (f *flex) per(n int) int {
	if n == f.home.slots {
		return f.home.perSeq
	}
	return min(f.total/n, f.most) / 256 * 256
}

// layout is the shape with n slots. Only home keeps a prefix pool: its entries
// are cells beside the conversations, and a layout that gives the whole budget
// to fewer, longer conversations has none to spare.
func (f *flex) layout(n int) layout {
	if n == f.home.slots {
		return f.home
	}
	return layout{slots: n, perSeq: f.per(n)}
}

// want is the most slots in which a prompt of promptLen tokens runs whole,
// leaving the room for a reply that every slot keeps (replyRoom). A prompt no
// layout holds whole gets the longest conversations on offer, in as many slots
// as still give them, and loses the fewest of its oldest turns there.
func (f *flex) want(promptLen, maxTokens int) int {
	best := f.home.slots
	for n := f.home.slots; n >= 1; n-- {
		if n != f.home.slots && f.refused[n] {
			continue
		}
		per := f.per(n)
		if promptLen <= per-replyRoom(per, maxTokens) {
			return n
		}
		if per > f.per(best) {
			best = n
		}
	}
	return best
}

// offers lists the layouts beyond home, for the line printed at load.
func (f *flex) offers() string {
	var parts []string
	last := f.home.perSeq
	for n := f.home.slots - 1; n >= 1; n-- {
		if p := f.per(n); p > last {
			parts = append(parts, f.layout(n).String())
			last = p
		}
	}
	return strings.Join(parts, ", ")
}

// placement is what admission does with the request at the head of the queue.
type placement int

const (
	// placeWait leaves it at the head until a slot frees or the slots drain.
	placeWait placement = iota
	// placeNow starts it in a free slot.
	placeNow
	// placeReshape rebuilds the idle context for it first.
	placeReshape
)

// place decides where the request at the head of the queue goes.
func (s *scheduler) place(j *job) placement {
	free := s.freeSlot() != nil
	switch {
	case s.flex == nil || j.want == len(s.slots):
		if free {
			return placeNow
		}
		return placeWait
	case s.active() == 0:
		return placeReshape
	case j.want > len(s.slots) && free && s.layoutNeeded():
		// Shorter than these slots, and something running still needs them:
		// a free one is here now, and the layout is not changing yet anyway.
		return placeNow
	default:
		// Longer than these slots, or shorter with nothing left that needs
		// them: admit nothing more, so the slots drain and the layout changes.
		return placeWait
	}
}

// layoutNeeded reports whether a running conversation would not have run whole
// in more slots than there are now.
func (s *scheduler) layoutNeeded() bool {
	for _, sl := range s.slots {
		if sl.job != nil && sl.job.want <= len(s.slots) {
			return true
		}
	}
	return false
}

// reshape rebuilds the context for the request at the head of the queue, as
// the layout it wants. Every slot is idle, so nothing is lost but the prefix
// pool, which lives in the context.
//
// A layout that cannot be built is refused for good and home is rebuilt, and
// the request is placed again among what is left. false means not even home
// could be built: the scheduler has no context and must stop.
func (s *scheduler) reshape(j *job) bool {
	from := layout{slots: len(s.slots), perSeq: s.ctxPerSeq}
	target := s.flex.layout(j.want)
	began := time.Now()

	llama.FreeContext(s.lctx)
	s.lctx = 0
	lctx, free, err := s.flex.build(target)
	if err != nil && target != s.flex.home {
		log.Printf("slots: %s could not be built for a %d-token prompt (%v) — rebuilding %s, and %d slots are not tried again",
			target, len(j.full), err, s.flex.home, target.slots)
		s.flex.refused[target.slots] = true
		j.want = s.flex.want(len(j.full), j.params.MaxTokens)
		target = s.flex.home
		lctx, free, err = s.flex.build(target)
	}
	if err != nil {
		log.Printf("slots: no context could be built (%v) — retiring this model; the next request reloads it", err)
		return false
	}
	s.install(lctx, target)
	if target == from {
		return true // back where it was, after a refusal already reported
	}

	why := fmt.Sprintf("for a %d-token prompt", len(j.full))
	if target.slots > from.slots {
		why = "— nothing running needs the longer slots"
	}
	left := ""
	if free > 0 {
		left = ", leaving " + gib(free) + " free (this process only)"
	}
	log.Printf("slots: %s -> %s %s (it queued %s; the cache was rebuilt in %s%s)",
		from, target, why, time.Since(j.queued).Round(100*time.Millisecond), time.Since(began).Round(time.Millisecond), left)
	return true
}

// install makes lctx the scheduler's context, divided as l.
func (s *scheduler) install(lctx llama.Context, l layout) {
	s.lctx = lctx
	s.ctxPerSeq = l.perSeq
	s.slots = make([]*slot, l.slots)
	for i := range s.slots {
		s.slots[i] = &slot{seq: int32(i), logitIdx: -1}
	}
	// Pooled prefixes live in sequence ids above the slots.
	s.pool = newPrefixPool(lctx, int32(l.slots), l.prefix, minPrefixMatch, s.wholePrefixes)
	s.nSlots.Store(int32(l.slots))
}

func (s *scheduler) freeSlot() *slot {
	for _, sl := range s.slots {
		if sl.job == nil {
			return sl
		}
	}
	return nil
}
