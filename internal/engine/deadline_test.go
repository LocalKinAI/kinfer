package engine

import (
	"testing"
	"time"
)

// A limit an operator turned off must stay off. Negative is how they say so,
// and silently re-imposing a default would be worse than having no flag.
func TestDisabledDeadlinesNeverFire(t *testing.T) {
	d := deadlines{generate: 0, wait: 0}
	long := time.Now().Add(-24 * time.Hour)
	if _, fired := d.expiredGenerating(long); fired {
		t.Error("a disabled generation limit fired after a day")
	}
	if _, fired := d.expiredWaiting(long); fired {
		t.Error("a disabled queue limit fired after a day")
	}
}

// A slot that has not started, or a job never queued, has no clock running.
// Treating a zero time as "infinitely old" would kill every request at birth.
func TestZeroTimeIsNotAnExpiredDeadline(t *testing.T) {
	d := deadlines{generate: time.Millisecond, wait: time.Millisecond}
	if _, fired := d.expiredGenerating(time.Time{}); fired {
		t.Error("an unstarted slot was reported as having run out of time")
	}
	if _, fired := d.expiredWaiting(time.Time{}); fired {
		t.Error("an unqueued job was reported as having waited too long")
	}
}

func TestDeadlinesFireOnceExceeded(t *testing.T) {
	d := deadlines{generate: 50 * time.Millisecond, wait: 50 * time.Millisecond}

	recent := time.Now().Add(-10 * time.Millisecond)
	if _, fired := d.expiredGenerating(recent); fired {
		t.Error("fired at 10ms against a 50ms limit")
	}
	old := time.Now().Add(-200 * time.Millisecond)
	el, fired := d.expiredGenerating(old)
	if !fired {
		t.Fatal("did not fire at 200ms against a 50ms limit")
	}
	if el < 150*time.Millisecond {
		t.Errorf("reported %v elapsed, want roughly 200ms — the caller is told how long it waited", el)
	}
}

// The message has to say which limit was hit. "Timed out" alone leaves an
// operator unable to tell a slow model from a saturated server.
func TestTimeoutErrorNamesTheStage(t *testing.T) {
	for _, stage := range []string{"generating", "waiting for a slot"} {
		e := &TimeoutError{After: 90 * time.Second, Stage: stage}
		got := e.Error()
		if !contains(got, stage) || !contains(got, "1m30s") {
			t.Errorf("Error() = %q, want the stage and the elapsed time", got)
		}
	}
}

// The two defaults are linked on purpose: a request that waited longer than a
// whole reply may take has been abandoned. If they drift apart, the reasoning
// in deadline.go stops describing the code.
func TestQueueLimitMatchesGenerationLimit(t *testing.T) {
	if DefaultMaxWait != DefaultMaxGenerate {
		t.Errorf("DefaultMaxWait %v != DefaultMaxGenerate %v; deadline.go explains them as the same number",
			DefaultMaxWait, DefaultMaxGenerate)
	}
}

// pick is what turns an operator's intent into a duration, and its three cases
// are all load-bearing.
func TestPickResolvesTheThreeCases(t *testing.T) {
	if got := pick(0, time.Minute); got != time.Minute {
		t.Errorf("unset = %v, want the default", got)
	}
	if got := pick(9*time.Second, time.Minute); got != 9*time.Second {
		t.Errorf("explicit = %v, want what was asked for", got)
	}
	if got := pick(-1, time.Minute); got != 0 {
		t.Errorf("negative = %v, want 0 — the operator turned the limit off", got)
	}
}
