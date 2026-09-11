package engine

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// Admission control.
//
// Before this, submit blocked when the queue was full. For a fleet that is the
// wrong shape twice over: a caller has no way to tell "you are behind 200
// requests" from "the model is slow", and an agent that would happily have gone
// away and come back instead sits on a connection until something times it out.
// The failure kinfer was built after was a hang, and a request that waits
// forever without saying so is the same thing wearing a different hat.
//
// So the queue is bounded and says no. Saying no is a service: 503 is a fact a
// client can act on, where an open connection is not.

// ErrBusy is returned when too many requests are already waiting. It carries
// the queue's state so a caller can be told how far behind it would have been.
var ErrBusy = errors.New("too many requests are already waiting")

// BusyError describes a refused admission. Errors.Is(err, ErrBusy) matches it.
type BusyError struct {
	// Waiting and Capacity are the queue at the moment of refusal.
	Waiting, Capacity int

	// RetryAfter is how long the queue would take to clear at the rate this
	// engine has actually been completing requests, or zero when that rate is
	// not yet known — which it is not until some requests have finished.
	//
	// Zero means "no estimate", not "retry immediately". A made-up number here
	// would be worse than none: every refused client would come back at the
	// same invented moment.
	RetryAfter time.Duration
}

func (e *BusyError) Error() string {
	if e.RetryAfter > 0 {
		return fmt.Sprintf("%v: %d of %d slots in the queue are taken, about %s of work ahead",
			ErrBusy, e.Waiting, e.Capacity, e.RetryAfter.Round(time.Second))
	}
	return fmt.Sprintf("%v: %d of %d slots in the queue are taken", ErrBusy, e.Waiting, e.Capacity)
}

func (e *BusyError) Unwrap() error { return ErrBusy }

// completionRate is a decaying average of how often requests finish.
//
// It exists to answer one question honestly: how long is the work already in
// the queue likely to take? Anything else — a fixed retry interval, a guess
// from the slot count — would be a number with nothing behind it, and this
// project has been bitten by those.
type completionRate struct {
	mu sync.Mutex

	last  time.Time
	mean  time.Duration // average gap between completions
	count int
}

// smoothing weights the newest interval against the running average. Low enough
// that one slow request does not move the estimate much, high enough that the
// estimate follows a change in load within a handful of requests.
const smoothing = 0.25

// done records that a request finished.
func (c *completionRate) done() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	if c.last.IsZero() {
		c.last, c.count = now, 1
		return
	}
	gap := now.Sub(c.last)
	c.last = now
	c.count++
	if c.mean == 0 {
		c.mean = gap
		return
	}
	c.mean = time.Duration(float64(c.mean)*(1-smoothing) + float64(gap)*smoothing)
}

// wait estimates how long n queued requests will take to clear, or zero when
// too little has finished to say.
//
// Two completions is the minimum that can produce an interval at all. Reporting
// from one would be reporting from nothing.
func (c *completionRate) wait(n int) time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.count < 2 || c.mean <= 0 || n <= 0 {
		return 0
	}
	return time.Duration(n) * c.mean
}
