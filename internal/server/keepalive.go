package server

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"strconv"
	"time"
)

// Loading and unloading a model on request, the way Ollama's clients ask.
//
// kinfer could only let go of a model on its own idle clock, or by stopping the
// whole service. On the box that meant `launchctl bootout` to free 73 GiB for a
// second model on the same machine, and bootstrap again afterwards — the whole
// runtime down so that one model could leave memory. Ollama does it with one
// request, and so does every script that drives it:
//
//	POST /api/generate {"model": "m", "keep_alive": 0}       unload (`ollama stop`)
//	POST /api/generate {"model": "m"}                        load ahead of use
//	POST /api/chat     {"model": "m", "messages": [], ...}   the same two, as chat
//
// and on an ordinary request, keep_alive is how long the model stays loaded
// after it. Answers are Ollama's shape exactly, done_reason "load" or "unload",
// measured against 0.34.3.

// parseKeepAlive reads a keep_alive field as Ollama does: a number of seconds,
// or a duration string ("10s", "5m", "0"); negative means keep the model
// forever. nil when the field is absent or null, meaning the server's own
// -keepalive applies.
//
// A negative result is the "forever" sentinel, not a duration.
func parseKeepAlive(raw json.RawMessage) (*time.Duration, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var d time.Duration
	var num float64
	var str string
	switch {
	case json.Unmarshal(raw, &num) == nil:
		d = secondsToDuration(num)
	case json.Unmarshal(raw, &str) == nil:
		if parsed, err := time.ParseDuration(str); err == nil {
			d = parsed
		} else if n, nerr := strconv.ParseFloat(str, 64); nerr == nil {
			// "300" — a number in a string. Ollama refuses it; accepting it
			// costs nothing and a script that quoted its seconds is not wrong
			// about what it meant.
			d = secondsToDuration(n)
		} else {
			return nil, fmt.Errorf("keep_alive %q: want seconds or a duration like \"5m\"", str)
		}
	default:
		return nil, fmt.Errorf("keep_alive %s: want seconds or a duration like \"5m\"", raw)
	}
	if d < 0 {
		d = -1
	}
	return &d, nil
}

func secondsToDuration(s float64) time.Duration {
	if s < 0 {
		return -1
	}
	if s > math.MaxInt64/float64(time.Second) {
		return -1 // longer than a Duration holds: forever, which is what it means
	}
	return time.Duration(s * float64(time.Second))
}

// noteKeepAlive records what a request asked for on the resident copy of
// model, for the idle clock that starts when the last request lets go of it —
// see idleForLocked. A request that named nothing clears an earlier one's
// wish: in Ollama each request's keep_alive, or the default, governs what
// follows that request, not every request after it.
func (s *Server) noteKeepAlive(model string, keepAlive *time.Duration) {
	path, err := s.store.Resolve(model)
	if err != nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cur != nil && s.cur.path == path {
		s.cur.keepAlive = keepAlive
	}
}

// loadOrUnload answers a request that carries no prompt: unload the model
// when keepAlive is zero, otherwise load it and let it sit for keepAlive.
//
// An unload returns once the memory is free, so a caller can load something
// else the moment it hears back. A request still generating on the model
// finishes first — unloading under it would crash the process (see lease) —
// and until it does, nothing else loads: this is a swap to nothing, and a
// second copy loading beside the first is exactly what the caller is freeing
// memory to avoid.
func (s *Server) loadOrUnload(model string, keepAlive *time.Duration) (reason string, status int, err error) {
	path, err := s.store.Resolve(model)
	if err != nil {
		return "", http.StatusNotFound, err
	}

	if keepAlive != nil && *keepAlive == 0 {
		s.mu.Lock()
		for s.swapping {
			s.cond.Wait()
		}
		if s.cur != nil && s.cur.path == path {
			s.swapping = true
			s.retireLocked()
			s.swapping = false
			s.cond.Broadcast()
			log.Printf("unloaded %s as asked (keep_alive 0)", path)
		}
		s.mu.Unlock()
		return "unload", http.StatusOK, nil
	}

	_, release, err := s.acquire(model)
	if err != nil {
		return "", http.StatusInternalServerError, err
	}
	s.noteKeepAlive(model, keepAlive)
	release()
	return "load", http.StatusOK, nil
}

type ollamaGenerateRequest struct {
	Model     string          `json:"model"`
	Prompt    string          `json:"prompt"`
	KeepAlive json.RawMessage `json:"keep_alive"`
}

type ollamaGenerateResponse struct {
	Model      string    `json:"model"`
	CreatedAt  time.Time `json:"created_at"`
	Response   string    `json:"response"`
	Done       bool      `json:"done"`
	DoneReason string    `json:"done_reason,omitempty"`
}

// handleOllamaGenerate serves /api/generate for loading and unloading only —
// the endpoint `ollama stop` and preloading scripts use. Generating from a raw
// prompt is /api/chat's job here; a prompt gets a plain refusal naming it,
// rather than a reply that pretends /api/generate's raw-prompt semantics were
// honoured.
func (s *Server) handleOllamaGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	log.Printf("→ /api/generate from %s", r.RemoteAddr)

	var req ollamaGenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if req.Prompt != "" {
		writeJSON(w, http.StatusNotImplemented, map[string]string{
			"error": "kinfer answers /api/generate only to load or unload a model (no prompt); send prompts to /api/chat",
		})
		return
	}
	keepAlive, err := parseKeepAlive(req.KeepAlive)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	reason, status, err := s.loadOrUnload(req.Model, keepAlive)
	if err != nil {
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, ollamaGenerateResponse{
		Model: req.Model, CreatedAt: time.Now(), Done: true, DoneReason: reason,
	})
}
