package server

import (
	"context"
	"errors"
	"log"
	"sync"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// retrying runs one request, and runs it again on a replacement model when the
// engine went down before the request had started.
//
// The blast radius of a fatal backend failure is what this narrows. llama.cpp
// reports such a failure for the whole context, not for the batch that hit it,
// so kinfer retires the model — and in doing so fails every request the engine
// was holding. Watched once on a load test: 26 at a stroke, of which the great
// majority were still in the queue and had produced nothing.
//
// Those are recoverable and the in-flight ones are not, which is the whole
// distinction. A slot that was mid-batch has already streamed part of an answer
// to its client; running it again would repeat that text. A queued job holds
// only its messages and parameters, so re-running it is invisible — the client
// waits slightly longer and gets one complete reply.
//
// The retry lives here rather than in engine because reloading a model is the
// server's job: an engine whose backend has died cannot replace itself.
type retrying struct {
	s    *Server
	name string

	mu       sync.Mutex
	eng      generator
	rel      func()
	released bool
}

// acquire returns an engine for the named model plus the release function the
// caller must call when it is done generating.
//
// The engine it hands back survives its own model being retired mid-request;
// see retrying.
func (s *Server) acquire(name string) (generator, func(), error) {
	eng, rel, err := s.acquireOnce(name)
	if err != nil {
		return nil, nil, err
	}
	rt := &retrying{s: s, name: name, eng: eng, rel: rel}
	return rt, rt.release, nil
}

func (rt *retrying) current() generator {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return rt.eng
}

// release drops this request's hold on whichever model it ended up using. Safe
// to call twice, because swap already called it once.
func (rt *retrying) release() {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.released {
		return
	}
	rt.released = true
	if rt.rel != nil {
		rt.rel()
	}
}

// swap exchanges the dead model for a freshly loaded one.
//
// Order matters and is the reason this is not a one-liner: retiring a model
// waits for every outstanding reference to it to go, and this request holds one.
// Re-acquiring before letting go of it would wait for itself.
func (rt *retrying) swap() error {
	rt.mu.Lock()
	if rt.rel != nil && !rt.released {
		rt.rel()
	}
	rt.eng, rt.rel, rt.released = nil, nil, true
	rt.mu.Unlock()

	eng, rel, err := rt.s.acquireOnce(rt.name)
	if err != nil {
		return err
	}

	rt.mu.Lock()
	rt.eng, rt.rel, rt.released = eng, rel, false
	rt.mu.Unlock()
	return nil
}

func (rt *retrying) ChatFull(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, engine.Stats, error) {
	text, st, err := rt.current().ChatFull(ctx, msgs, p, onToken)
	if !errors.Is(err, engine.ErrNotStarted) {
		return text, st, err
	}

	// Nothing of this request reached llama.cpp or the client, so running it
	// again is not a repeat — it is the first attempt that gets to happen.
	log.Printf("%s went down before a queued request started; retrying it on a fresh model", rt.name)
	if err := rt.swap(); err != nil {
		return "", engine.Stats{}, err
	}
	return rt.current().ChatFull(ctx, msgs, p, onToken)
}

func (rt *retrying) Chat(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, error) {
	text, _, err := rt.ChatFull(ctx, msgs, p, onToken)
	return text, err
}

func (rt *retrying) Broken() bool             { return rt.current().Broken() }
func (rt *retrying) ToolFormat() tools.Format { return rt.current().ToolFormat() }

// Close is the model's lifecycle, not one request's. The server closes engines
// when it retires them; a handler holding this wrapper must not.
func (rt *retrying) Close() {}
