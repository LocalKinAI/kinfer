package engine

import (
	"errors"
	"testing"
	"time"
)

// A refusal has to be recognisable as one. A caller that cannot tell "too busy"
// from "something broke" will either hammer a healthy server or abandon work
// that would have succeeded.
func TestBusyErrorIsIdentifiable(t *testing.T) {
	var err error = &BusyError{Waiting: 128, Capacity: 128}
	if !errors.Is(err, ErrBusy) {
		t.Fatal("a BusyError does not match ErrBusy, so no handler can act on it")
	}
	var be *BusyError
	if !errors.As(err, &be) || be.Waiting != 128 {
		t.Error("the queue state did not survive being wrapped as an error")
	}
}

// The message must say how deep the queue was; "busy" alone tells an operator
// nothing about whether to add capacity or to look for a stuck request.
func TestBusyErrorSaysHowFarBehind(t *testing.T) {
	e := &BusyError{Waiting: 96, Capacity: 128}
	if got := e.Error(); got == "" || !contains(got, "96") || !contains(got, "128") {
		t.Errorf("Error() = %q, want the queue depth in it", got)
	}
	withWait := &BusyError{Waiting: 96, Capacity: 128, RetryAfter: 42 * time.Second}
	if got := withWait.Error(); !contains(got, "42s") {
		t.Errorf("Error() = %q, want the estimated wait when one is known", got)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// No estimate until there is something to estimate from. A Retry-After that is
// a guess sends every refused caller back at the same invented moment.
func TestCompletionRateStaysSilentUntilItHasData(t *testing.T) {
	var c completionRate
	if got := c.wait(10); got != 0 {
		t.Errorf("wait = %v before anything finished, want 0 (no estimate)", got)
	}
	c.done()
	if got := c.wait(10); got != 0 {
		t.Errorf("wait = %v after one completion, want 0 — one point is not an interval", got)
	}
	c.done()
	if got := c.wait(10); got <= 0 {
		t.Error("wait = 0 after two completions, so no estimate is ever produced")
	}
}

// The estimate scales with the queue: ten waiting must be reported as longer
// than one.
func TestCompletionRateScalesWithDepth(t *testing.T) {
	var c completionRate
	c.done()
	time.Sleep(2 * time.Millisecond)
	c.done()

	one, ten := c.wait(1), c.wait(10)
	if one <= 0 {
		t.Fatal("no estimate after two completions")
	}
	if ten <= one {
		t.Errorf("wait(10) = %v is not longer than wait(1) = %v", ten, one)
	}
	if c.wait(0) != 0 || c.wait(-1) != 0 {
		t.Error("an empty queue was given a non-zero wait")
	}
}
