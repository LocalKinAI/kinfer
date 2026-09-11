package server

import (
	"errors"
	"math"
	"net/http"
	"strconv"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// writeGenError turns a generation failure into a response, distinguishing a
// server that is overloaded from one that is broken.
//
// 503 rather than 500 for a full queue, because they mean different things to
// the thing on the other end: 500 says this request was bad or this server is
// sick, and 503 says come back. A fleet that retries a 500 is hammering a
// broken server; a fleet that gives up on a 503 is abandoning work that would
// have succeeded. Getting the code right is most of what backpressure is.
func writeGenError(w http.ResponseWriter, err error) {
	// A request that aged out of the queue is the same answer as a full one —
	// the server has more work than it can take — reached by time rather than
	// by depth. It gets the same status, and no Retry-After, because nothing
	// was measured about when this one would have run.
	var stale *engine.TimeoutError
	if errors.As(err, &stale) && stale.BeforeStarting() {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": stale.Error()})
		return
	}

	var busy *engine.BusyError
	if errors.As(err, &busy) {
		if busy.RetryAfter > 0 {
			// Whole seconds, rounded up: the header has no finer unit, and
			// rounding down would invite the caller back before the work it is
			// waiting on can have finished.
			w.Header().Set("Retry-After",
				strconv.Itoa(int(math.Ceil(busy.RetryAfter.Seconds()))))
		}
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": busy.Error()})
		return
	}
	writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
}

// writeOpenAIGenError is writeGenError in the OpenAI dialect's error shape.
func writeOpenAIGenError(w http.ResponseWriter, err error) {
	var stale *engine.TimeoutError
	if errors.As(err, &stale) && stale.BeforeStarting() {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"error": map[string]string{"message": stale.Error(), "type": "server_busy"},
		})
		return
	}

	var busy *engine.BusyError
	if errors.As(err, &busy) {
		if busy.RetryAfter > 0 {
			w.Header().Set("Retry-After",
				strconv.Itoa(int(math.Ceil(busy.RetryAfter.Seconds()))))
		}
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"error": map[string]string{"message": busy.Error(), "type": "server_busy"},
		})
		return
	}
	writeJSON(w, http.StatusInternalServerError, map[string]any{
		"error": map[string]string{"message": err.Error()},
	})
}

// refusedBeforeStreaming reports whether err is a refusal that reached the
// handler before anything was written, so a status code is still possible.
//
// A stream that has begun cannot be given one — the header is long gone — and
// that is why errors are carried in the body of the terminal object. A refusal
// is the one failure that always happens first, at admission, so it is the one
// that can still be a 503. Handlers defer their header write for exactly this.
func refusedBeforeStreaming(err error, streamed bool) bool {
	if streamed {
		return false
	}
	var stale *engine.TimeoutError
	return errors.Is(err, engine.ErrBusy) ||
		(errors.As(err, &stale) && stale.BeforeStarting())
}
