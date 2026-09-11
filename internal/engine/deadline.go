package engine

import "time"

// Deadlines, on the two places a request can consume the server without
// finishing: a slot it will not give back, and a place in the queue it no
// longer wants.
//
// Neither was bounded. The token budget looks like it bounds the first, but
// `num_predict` is passed straight through and Ollama's convention for it
// includes "keep going" — so a caller could hold a slot until the model emitted
// an end token or filled the context. On the 125B that is 16384 tokens at 9
// tokens a second: half an hour of one slot, for one request, and eight such
// requests are the whole server.
//
// Which is the shape of the outage kinfer was written after. `kimi-k2.5:cloud`
// did not return an error on 2026-07-23; it stopped answering, and a fleet of
// about 150 agents waited on it for half a day. A runtime that can be held that
// way by an ordinary request has reproduced the thing it was built to prevent —
// and reproduced it as the same silence, because a slow server and a stuck one
// look identical from outside.
//
// So: a wall clock on generation, and an age limit on waiting.

const (
	// DefaultMaxGenerate is how long one reply may take.
	//
	// Five minutes is far past any answer a fleet agent is still waiting for
	// and far short of the half hour a single request could previously hold a
	// slot. It is a backstop, not a policy: a caller's own context deadline is
	// the authority whenever it sets one, and this only covers the callers that
	// do not.
	DefaultMaxGenerate = 5 * time.Minute

	// DefaultMaxWait is how long a request may sit in the queue before it is
	// refused rather than given a slot.
	//
	// Tied to DefaultMaxGenerate rather than chosen separately, and the link is
	// the argument: a request that has already waited longer than a whole reply
	// is allowed to take has almost certainly been abandoned, and running it
	// would spend a slot on an answer nobody is left to read.
	DefaultMaxWait = DefaultMaxGenerate
)

// TimeoutError is a reply cut short by the wall clock.
//
// It is an error rather than a quiet truncation at the engine's boundary, but
// the HTTP layer reports it as a finish reason with whatever text was produced,
// because that is more use to a caller than a failure with nothing attached.
// Reporting it as an ordinary stop would be the bug this project keeps finding
// in itself: a request that did not finish, described as one that did.
type TimeoutError struct {
	After time.Duration
	Stage string
}

// The two stages are not interchangeable, and conflating them produces exactly
// the bug this type exists to prevent. A reply cut off while generating has
// text to hand back and a reason to attach. A request that expired in the queue
// produced nothing at all, and dressing that up as a finished reply with an
// empty body would tell a caller the model had nothing to say.
const (
	StageGenerating = "generating"
	StageQueued     = "waiting for a slot"
)

// BeforeStarting reports that nothing of the request ever ran, so there is no
// partial answer and the caller should be refused rather than given an empty
// one. It is the same situation as a full queue, arrived at by time instead of
// by depth.
func (e *TimeoutError) BeforeStarting() bool { return e.Stage == StageQueued }

func (e *TimeoutError) Error() string {
	return "gave up after " + e.After.Round(time.Second).String() + " " + e.Stage
}

// deadlines holds the limits one scheduler enforces. Zero in either field
// disables that limit, which is a choice an operator can make and not one
// kinfer makes for them.
type deadlines struct {
	generate time.Duration
	wait     time.Duration
}

// expiredGenerating reports that a slot has held its job too long. since is when
// the slot admitted it, which is where the clock starts: queueing is the other
// limit's business.
func (d deadlines) expiredGenerating(since time.Time) (time.Duration, bool) {
	if d.generate <= 0 || since.IsZero() {
		return 0, false
	}
	if el := time.Since(since); el > d.generate {
		return el, true
	}
	return 0, false
}

// expiredWaiting reports that a queued job has been waiting too long to be
// worth starting.
func (d deadlines) expiredWaiting(queued time.Time) (time.Duration, bool) {
	if d.wait <= 0 || queued.IsZero() {
		return 0, false
	}
	if el := time.Since(queued); el > d.wait {
		return el, true
	}
	return 0, false
}
