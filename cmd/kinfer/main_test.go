package main

import (
	"testing"
	"time"
)

// The command line and the Go API say "no limit" differently, and the
// translation between them is the only place that can get it wrong.
//
// It exists because `-max-gen -1` cannot be typed: Go parses a duration and
// "-1" has no unit, so the flag package errors and the process exits. That
// happened, into a log nobody was reading, and looked exactly like a server
// that would not start.
func TestNoLimitTranslatesZero(t *testing.T) {
	if got := noLimit(0); got >= 0 {
		t.Errorf("noLimit(0) = %v, want a negative — 0 on the command line means no limit, "+
			"and 0 in engine.Options means take the default", got)
	}
	for _, d := range []time.Duration{time.Second, 5 * time.Minute, time.Hour} {
		if got := noLimit(d); got != d {
			t.Errorf("noLimit(%v) = %v, want it passed through", d, got)
		}
	}
	// A negative typed with a unit is already the API's own way of saying off.
	if got := noLimit(-time.Second); got != -time.Second {
		t.Errorf("noLimit(-1s) = %v, want it left alone", got)
	}
}
