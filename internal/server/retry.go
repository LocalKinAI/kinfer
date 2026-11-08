package server

import (
	"context"
	"errors"
	"log"
	"sync"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// retrying runs one request, and when the engine went down under it, reloads
// the model and runs it again — or, if it was in the middle of its reply, goes
// on with the reply on the replacement.
//
// The blast radius of a fatal backend failure is what this narrows. llama.cpp
// reports such a failure for the whole context, not for the batch that hit it,
// so kinfer retires the model — and in doing so fails every request the engine
// was holding. Watched once on a load test: 26 at a stroke, of which the great
// majority were still in the queue and had produced nothing. On a Mac the usual
// cause is the GPU running out of memory, because another app took some after
// the load; the replacement is placed to fit what is free when it loads.
//
// The two kinds are recovered differently. A queued job holds only its messages
// and parameters, so re-running it is invisible — the client waits slightly
// longer and gets one complete reply. A slot that was mid-batch has already
// streamed part of an answer to its client, and running it again would repeat
// that text; the replacement prefills it after the prompt and goes on from its
// end instead (engine.GenParams.Resume), so the client sees a pause.
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

// maxReloads is how many replacement models one request loads before its
// failure is reported. A replacement fits what is free when it loads, so one is
// the usual; the second is for memory taken again in between, and the bound is
// for a model that fails however it is placed.
const maxReloads = 2

func (rt *retrying) ChatFull(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, engine.Stats, error) {
	text, st, err := rt.current().ChatFull(ctx, msgs, p, onToken)
	for reloads := 0; reloads < maxReloads && ctx.Err() == nil; reloads++ {
		switch {
		case errors.Is(err, engine.ErrNotStarted):
			// Nothing of this attempt reached llama.cpp or the client, so
			// running it again is not a repeat — it is the attempt that gets
			// to happen.
			log.Printf("%s went down before a queued request started; retrying it on a fresh model", rt.name)
		case errors.Is(err, engine.ErrInterrupted):
			log.Printf("%s went down %d tokens into a reply; reloading it to go on from there", rt.name, st.EvalTokens)
		default:
			return text, st, err
		}

		reloadStart := time.Now()
		if err := rt.swap(); err != nil {
			return text, st, err
		}
		reload := time.Since(reloadStart)

		q := p
		q.Resume = p.Resume + text
		if p.MaxTokens > 0 {
			q.MaxTokens = max(p.MaxTokens-st.EvalTokens, 1)
		}
		var more string
		var next engine.Stats
		more, next, err = rt.current().ChatFull(ctx, msgs, q, onToken)
		text += more
		st = carriedOver(st, next)
		// The caller measured its load before generation started, so the
		// reload this request just paid for is invisible to it. Hand it back.
		st.ReloadDuration += reload
	}
	return text, st, err
}

// carriedOver is the stats of a request that ran as st and then, on a
// replacement model, as next. A first attempt that never ran leaves next's
// prompt as the request's; one that did makes next's prefill — the prompt
// again, and the reply so far — part of the recovery, with the reload.
func carriedOver(st, next engine.Stats) engine.Stats {
	if st.PromptTokens == 0 {
		st.PromptTokens, st.PromptEvalDuration = next.PromptTokens, next.PromptEvalDuration
	} else {
		st.ReloadDuration += next.PromptEvalDuration
	}
	st.EvalTokens += next.EvalTokens
	st.EvalDuration += next.EvalDuration
	st.ReloadDuration += next.ReloadDuration
	st.Truncated, st.Timeout = next.Truncated, next.Timeout
	return st
}

func (rt *retrying) Chat(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, error) {
	text, _, err := rt.ChatFull(ctx, msgs, p, onToken)
	return text, err
}

func (rt *retrying) Broken() bool                     { return rt.current().Broken() }
func (rt *retrying) Load() (waiting, busy, slots int) { return rt.current().Load() }
func (rt *retrying) PromptTokens() int64              { return rt.current().PromptTokens() }
func (rt *retrying) EvalTokens() int64                { return rt.current().EvalTokens() }
func (rt *retrying) Drafts() (int64, int64)           { return rt.current().Drafts() }
func (rt *retrying) Reuse() (int64, int)              { return rt.current().Reuse() }
func (rt *retrying) ToolGrammar() (int64, int64)      { return rt.current().ToolGrammar() }
func (rt *retrying) Context() (int, int)              { return rt.current().Context() }
func (rt *retrying) ToolFormat() tools.Format         { return rt.current().ToolFormat() }

func (rt *retrying) CountTokens(msgs []chat.Message, p engine.GenParams) (int, error) {
	return rt.current().CountTokens(msgs, p)
}

// Close is the model's lifecycle, not one request's. The server closes engines
// when it retires them; a handler holding this wrapper must not.
func (rt *retrying) Close() {}
