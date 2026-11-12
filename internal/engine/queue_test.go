package engine

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
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
	time.Sleep(20 * time.Millisecond) // Windows' clock can read the same twice
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

// testQueue is a scheduler's queue and nothing else: no loop reads it.
func testQueue(capacity int) *scheduler {
	return &scheduler{incoming: make(chan *job, capacity*givenUpRoom), queueCap: capacity,
		stop: make(chan struct{}), stopped: make(chan struct{})}
}

func newJob(ctx context.Context) *job {
	return &job{ctx: ctx, frags: make(chan string, 1), done: make(chan struct{})}
}

// Requests whose callers gave up stop counting at once: a burst of callers
// timing out and asking again does not leave the ones still wanted refused.
func TestGivenUpRequestsDoNotFillTheQueue(t *testing.T) {
	s := testQueue(2)
	var cancels []context.CancelFunc
	for i := 0; i < 2; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		cancels = append(cancels, cancel)
		if err := s.submit(newJob(ctx)); err != nil {
			t.Fatalf("request %d of 2: %v", i+1, err)
		}
	}
	if err := s.submit(newJob(context.Background())); !errors.Is(err, ErrBusy) {
		t.Fatalf("a third with two waiting: %v, want busy", err)
	}
	for _, c := range cancels {
		c()
	}
	deadline := time.Now().Add(5 * time.Second)
	for s.Waiting() != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := s.Waiting(); got != 0 {
		t.Fatalf("Waiting = %d after both callers gave up, want 0", got)
	}
	// Both are still in the queue, given up; two wanted ones fit beside them.
	for i := 0; i < 2; i++ {
		if err := s.submit(newJob(context.Background())); err != nil {
			t.Fatalf("a wanted request beside two given up: %v", err)
		}
	}
	if err := s.submit(newJob(context.Background())); !errors.Is(err, ErrBusy) {
		t.Errorf("a third wanted one: %v, want busy", err)
	}
	// A given-up one taken off the queue is not counted out twice.
	for len(s.incoming) > 0 {
		s.taken(<-s.incoming)
	}
	if got := s.Waiting(); got != 0 {
		t.Errorf("Waiting = %d with the queue emptied, want 0", got)
	}
}

// Once the loop has ended — a fatal backend failure — nothing is left waiting
// for it: what was queued is answered, and what comes after is refused.
func TestNothingWaitsForALoopThatEnded(t *testing.T) {
	s := testQueue(4)
	j := newJob(context.Background())
	if err := s.submit(j); err != nil {
		t.Fatal(err)
	}
	s.ended()
	select {
	case <-j.done:
	case <-time.After(5 * time.Second):
		t.Fatal("a request queued before the loop ended was never answered")
	}
	if !errors.Is(j.err, ErrNotStarted) {
		t.Errorf("queued before the end: %v, want ErrNotStarted", j.err)
	}
	if err := s.submit(newJob(context.Background())); !errors.Is(err, ErrNotStarted) {
		t.Errorf("after the end: %v, want ErrNotStarted", err)
	}
	if got := s.Waiting(); got != 0 {
		t.Errorf("Waiting = %d after the end, want 0", got)
	}
}

// A reply longer than its buffer, read slowly, keeps its end: what the buffer
// had no room for goes as the job's tail, read after the rest.
func TestAReplyReadSlowlyKeepsItsEnd(t *testing.T) {
	s := &scheduler{tpl: &chat.Template{}}
	j := &job{ctx: context.Background(), frags: make(chan string, 1), done: make(chan struct{})}
	j.frags <- "the start, " // the buffer is full, and nobody has read it yet
	sl := &slot{job: j}
	sl.out.WriteString("the start, and the end.")
	sl.emitted = len("the start, ")
	s.finishSlot(sl)
	var got strings.Builder
	for f := range j.frags {
		got.WriteString(f)
	}
	got.WriteString(j.tail)
	if got.String() != "the start, and the end." {
		t.Errorf("read slowly, the reply came as %q", got.String())
	}
}
