// Package server exposes loaded models over HTTP.
//
// It speaks two dialects on purpose:
//
//   - Ollama's /api/chat and /api/tags
//   - OpenAI's /v1/chat/completions and /v1/models
//
// The Ollama dialect is what makes kinfer a drop-in second leg for LocalKin.
// Its ~300 souls already declare `provider: "ollama"`, so switching a fleet from
// Ollama to kinfer is one endpoint change — and switching back, when something
// goes wrong at 2am, is the same one change.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/store"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// Server holds the model store and whatever is currently loaded.
type Server struct {
	store *store.Store
	opts  engine.Options

	// stats are what /metrics reports.
	stats counters

	// Exactly one model stays resident. A 7B model is several GB, so keeping a
	// map of them would exhaust memory on the laptops kinfer targets; loading is
	// fast enough (0.5s for a 0.5B, a few seconds for a 7B) that swapping on
	// demand is the better trade.
	//
	// The resident model is reference counted. A request that is mid-generation
	// still holds a llama.cpp context, so freeing it because another request
	// asked for a different model is a use-after-free: the first request's next
	// llama_decode lands on a NULL context and takes the whole process down.
	// cond wakes the swapper when the last in-flight request lets go.
	mu       sync.Mutex
	cond     *sync.Cond
	cur      *lease
	swapping bool

	// keepAlive is how long an idle model stays resident. The CLI relies on
	// this: `kinfer run` is only instant the second time if the daemon still
	// has the weights. Zero keeps a model forever.
	keepAlive time.Duration
	idle      *time.Timer
	expiry    time.Time

	// open loads a model. Tests replace it; nothing else should.
	open func(path string, opts engine.Options) (generator, error)
}

// Version is reported by /api/version.
//
// It has to parse as a version: Ollama clients compare this string to decide
// which features to use, and a value like "kinfer" fails that comparison
// silently, disabling capabilities rather than reporting anything.
const Version = "0.2.0"

// lease is one loaded model plus the number of users holding it: every
// in-flight request, plus one for the server itself while the model is
// resident. The engine is closed when the count reaches zero, which may be
// after the server has already stopped calling it resident.
type lease struct {
	eng  generator
	path string
	refs int

	// keepAlive is what the last request to finish with this model asked for,
	// in Ollama's terms (see parseKeepAlive); nil when it named nothing, and the
	// server's own -keepalive applies.
	keepAlive *time.Duration
}

// generator is the part of engine.Engine this package uses.
//
// It exists as an interface for one reason: the rules this package has to get
// right — who may free a loaded model, and when — are rules about concurrency,
// and a test for them must not need a multi-gigabyte GGUF and a GPU. The bug
// this seam was introduced for freed a model while another request was still
// generating, which segfaulted the process.
type generator interface {
	Chat(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, error)
	ChatFull(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, engine.Stats, error)
	Broken() bool
	ToolFormat() tools.Format
	Load() (waiting, busy, slots int)
	PromptTokens() int64
	EvalTokens() int64
	Close()
}

func openEngine(path string, opts engine.Options) (generator, error) {
	return engine.Open(path, opts)
}

// New creates a server over the given store.
func New(s *store.Store, opts engine.Options) *Server {
	srv := &Server{store: s, opts: opts, open: openEngine}
	srv.cond = sync.NewCond(&srv.mu)
	return srv
}

// SetKeepAlive sets how long an idle model stays loaded. Call before serving.
func (s *Server) SetKeepAlive(d time.Duration) { s.keepAlive = d }

// Handler returns the HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// Ollama dialect.
	mux.HandleFunc("/api/chat", s.handleOllamaChat)
	mux.HandleFunc("/api/generate", s.handleOllamaGenerate)
	mux.HandleFunc("/api/tags", s.handleOllamaTags)
	mux.HandleFunc("/api/ps", s.handlePS)
	mux.HandleFunc("/api/show", s.handleShow)
	mux.HandleFunc("/api/version", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"version": Version})
	})

	// OpenAI dialect.
	mux.HandleFunc("/v1/chat/completions", s.handleOpenAIChat)
	mux.HandleFunc("/v1/models", s.handleOpenAIModels)

	// The dialects coding agents speak: Anthropic's Messages API for Claude
	// Code, OpenAI's Responses API for Codex. Same engine, same split of thinking
	// from prose from calls — two more ways of writing it down.
	mux.HandleFunc("/v1/messages", s.handleAnthropicMessages)
	mux.HandleFunc("/v1/responses", s.handleResponses)

	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	// Counting wraps everything, so an endpoint added later cannot forget to
	// report itself.
	return s.count(mux)
}

// Close releases the resident model.
func (s *Server) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.retireLocked()
}

// retireLocked stops treating the resident model as resident and waits for
// in-flight requests to finish with it. Caller holds s.mu; the wait releases
// it, so other requests are not blocked meanwhile.
func (s *Server) retireLocked() {
	s.stopIdleLocked()
	old := s.cur
	if old == nil {
		return
	}
	s.cur = nil
	s.dropLocked(old)
	for old.refs > 0 {
		s.cond.Wait()
	}
}

func (s *Server) release(l *lease) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.dropLocked(l)
}

// dropLocked removes one reference, closing the engine when the last one goes.
func (s *Server) dropLocked(l *lease) {
	l.refs--
	if l.refs == 0 {
		l.eng.Close()
	}
	s.armIdleLocked()
	s.cond.Broadcast()
}

// armIdleLocked starts the unload countdown once nothing is generating.
//
// refs == 1 means the server itself is the only holder: the model is resident
// but nobody is using it. That is the moment the clock should start.
func (s *Server) armIdleLocked() {
	s.stopIdleLocked()
	if s.cur == nil || s.cur.refs != 1 {
		return
	}
	l := s.cur
	d, forever := s.idleForLocked(l)
	if forever {
		return
	}
	s.expiry = time.Now().Add(d)
	s.idle = time.AfterFunc(d, func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		// Re-check under the lock: a request may have arrived while the timer
		// was firing, in which case the model is in use again.
		if s.cur == l && l.refs == 1 {
			log.Printf("unloading %s after %s idle", l.path, d)
			s.retireLocked()
		}
	})
}

// idleForLocked is how long l may sit idle before it is unloaded, or forever.
//
// The -keepalive flag and a request's keep_alive disagree about zero, and each
// keeps its own meaning. The flag's zero is "forever": `kinfer install` records
// -keepalive 0 for a machine that serves a fleet. The request field is Ollama's
// wire format, where zero is "unload now" — the idiom `ollama stop` and every
// script that frees memory for another model send — so there it means that.
func (s *Server) idleForLocked(l *lease) (d time.Duration, forever bool) {
	if l.keepAlive != nil {
		if *l.keepAlive < 0 {
			return 0, true
		}
		return *l.keepAlive, false
	}
	if s.keepAlive <= 0 {
		return 0, true
	}
	return s.keepAlive, false
}

func (s *Server) stopIdleLocked() {
	if s.idle != nil {
		s.idle.Stop()
		s.idle = nil
	}
	s.expiry = time.Time{}
}

// acquireOnce returns an engine for the named model, loading it if necessary,
// plus a release function the caller MUST call when it is done generating.
//
// The engine stays alive for as long as the reference is held. Swapping to a
// different model retires the current one and waits for its in-flight requests
// to finish before loading the replacement — only one model fits in memory, and
// freeing one that is still generating crashes the process.
//
// Handlers call acquire, not this: a model can be retired while a request is
// still queued against it, and acquire wraps that case. Callers of this one
// hold a reference to a specific engine and must let go of it before asking
// for another, or retireLocked will wait for them.
func (s *Server) acquireOnce(name string) (generator, func(), error) {
	path, err := s.store.Resolve(name)
	if err != nil {
		return nil, nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for {
		// A model whose backend failed fatally cannot generate again. Handing
		// it out would answer every request with the same error, which for a
		// fallback runtime means being down without saying so. Retire it and
		// load a fresh one.
		if s.cur != nil && s.cur.eng.Broken() {
			log.Printf("retiring %s: its llama.cpp backend failed fatally", s.cur.path)
			s.retireLocked()
		}

		if s.cur != nil && s.cur.path == path {
			l := s.cur
			l.refs++
			s.stopIdleLocked()
			return l.eng, func() { s.release(l) }, nil
		}
		// Someone else is already loading. Wait rather than start a second
		// load of the same multi-gigabyte file.
		if s.swapping {
			s.cond.Wait()
			continue
		}
		break
	}

	s.swapping = true
	defer func() {
		s.swapping = false
		s.cond.Broadcast()
	}()

	s.retireLocked()

	// Loading takes seconds for a 7B; hold no lock across it so requests for
	// the model being loaded can queue on cond instead of on the mutex.
	s.mu.Unlock()
	log.Printf("loading %s", path)
	e, err := s.open(path, s.opts)
	s.mu.Lock()

	if err != nil {
		return nil, nil, err
	}
	l := &lease{eng: e, path: path, refs: 2} // one for the server, one for the caller
	s.cur = l
	s.stopIdleLocked()
	return e, func() { s.release(l) }, nil
}

// ─── Ollama dialect ──────────────────────────────────────────────────────────

type ollamaChatRequest struct {
	Model    string         `json:"model"`
	Messages []ollamaMsg    `json:"messages"`
	Stream   *bool          `json:"stream"`
	Options  *ollamaOptions `json:"options"`
	Tools    []tools.Tool   `json:"tools"`

	// Think is Ollama's switch for a reasoning model: false keeps it out of its
	// think block (see thinkOff); true, a level string, or nothing at all lets
	// it think and returns the thinking in message.thinking (see wantThinking).
	Think json.RawMessage `json:"think"`

	// KeepAlive is how long the model stays loaded after this request, in
	// Ollama's terms: see parseKeepAlive. With no messages, the request only
	// loads the model — or, at zero, unloads it.
	KeepAlive json.RawMessage `json:"keep_alive"`
}

type ollamaMsg struct {
	Role      string       `json:"role"`
	Content   string       `json:"content"`
	Thinking  string       `json:"thinking,omitempty"`
	ToolCalls []ollamaCall `json:"tool_calls,omitempty"`
}

// ollamaCall is a tool call on the wire. Both dialects nest the call under a
// "function" key; kinfer's own tools.Call is the flat form.
type ollamaCall struct {
	Function tools.Call `json:"function"`
}

// thinkOff reports that the caller explicitly asked for no thinking — a JSON
// false, or the string "false" Ollama also accepts.
//
// Absent is not off. A reasoning model left to itself thinks, which is what
// Ollama does when the field is missing, and matching that keeps a caller that
// never sends the field getting the same model behaviour from both.
func thinkOff(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var b bool
	if json.Unmarshal(raw, &b) == nil {
		return !b
	}
	var s string
	return json.Unmarshal(raw, &s) == nil && s == "false"
}

// wantThinking reports whether a reasoning model's working out goes back to the
// caller, in its own field — message.thinking here, reasoning and
// reasoning_content in the OpenAI dialect — never mixed into the reply.
//
// Absent means yes, as it does in Ollama (measured against 0.34.3 on the box:
// no "think" field, and the reply carries the model's thinking beside a clean
// content). It used to mean no: the model still thought, since absent is not
// off (see thinkOff), but the thinking was split off and dropped. A caller then
// got an empty reply with nothing to show why — measured on Flash-Next behind a
// wall-clock limit, 3,143 tokens generated, 0 characters of thinking and 0 of
// content returned. Dropping it also never protected anyone: a client that
// prints content verbatim does not print a field it does not know about.
func wantThinking(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return true
	}
	var b bool
	if json.Unmarshal(raw, &b) == nil {
		return b
	}
	// Ollama also accepts a level ("low", "high"); any string asks for it.
	var s string
	return json.Unmarshal(raw, &s) == nil && s != "" && s != "false"
}

func wireCalls(calls []tools.Call) []ollamaCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]ollamaCall, len(calls))
	for i, c := range calls {
		out[i] = ollamaCall{Function: c}
	}
	return out
}

// toChatMessages converts wire messages, keeping tool calls and tool results
// so a conversation can be replayed with its function round trips intact.
func toChatMessages(in []ollamaMsg) []chat.Message {
	out := make([]chat.Message, len(in))
	for i, m := range in {
		out[i] = chat.Message{Role: m.Role, Content: m.Content}
		for _, c := range m.ToolCalls {
			out[i].ToolCalls = append(out[i].ToolCalls, c.Function)
		}
	}
	return out
}

type ollamaOptions struct {
	Temperature   *float64 `json:"temperature"`
	TopP          *float64 `json:"top_p"`
	TopK          *int     `json:"top_k"`
	RepeatPenalty *float64 `json:"repeat_penalty"`
	NumPredict    *int     `json:"num_predict"`
	Seed          *int64   `json:"seed"`
}

type ollamaChatResponse struct {
	Model      string    `json:"model"`
	CreatedAt  time.Time `json:"created_at"`
	Message    ollamaMsg `json:"message"`
	Done       bool      `json:"done"`
	DoneReason string    `json:"done_reason,omitempty"`

	// Timings, in integer nanoseconds, on the final object only — Ollama's
	// shape exactly, because everything that measures a local model reads these
	// and none of it asks kinfer how it spells them. `ollama run --verbose`
	// divides eval_count by eval_duration for its tokens-per-second line;
	// benchmark scripts and dashboards do the same. Absent, they read zero, and
	// a runtime that reports zero tokens in zero nanoseconds can only be timed
	// from outside with a stopwatch.
	//
	// They are omitempty so the streaming path's per-token objects stay as
	// slim as Ollama's, which carry no counts at all.
	TotalDuration      int64 `json:"total_duration,omitempty"`
	LoadDuration       int64 `json:"load_duration,omitempty"`
	PromptEvalCount    int   `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64 `json:"prompt_eval_duration,omitempty"`
	EvalCount          int   `json:"eval_count,omitempty"`
	EvalDuration       int64 `json:"eval_duration,omitempty"`

	// Error travels in-band because the response headers are long gone by the
	// time generation can fail. Without it a failure is indistinguishable from
	// an empty reply — which is precisely how a context overflow reached
	// LocalKin as a silent empty answer over a 200 OK.
	Error string `json:"error,omitempty"`
}

// withTimings fills in the measurements a caller reads to work out how fast
// this was. load is how long acquiring the model took — seconds when the
// request paid for a load, microseconds when it found one resident — and total
// is the whole request, prompt to last token.
//
// A model that died mid-request and was reloaded adds to load rather than
// vanishing: the caller took its reading before generation, the reload happened
// during it, and the two are the same cost. Without this the components of a
// retried request sum to well under its total.
func (r ollamaChatResponse) withTimings(st engine.Stats, load, total time.Duration) ollamaChatResponse {
	r.TotalDuration = total.Nanoseconds()
	r.LoadDuration = (load + st.ReloadDuration).Nanoseconds()
	r.PromptEvalCount = st.PromptTokens
	r.PromptEvalDuration = st.PromptEvalDuration.Nanoseconds()
	r.EvalCount = st.EvalTokens
	r.EvalDuration = st.EvalDuration.Nanoseconds()
	return r
}

func (s *Server) handleOllamaChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	log.Printf("→ /api/chat from %s", r.RemoteAddr)

	start := time.Now()

	var req ollamaChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("  decode failed: %v", err)
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	keepAlive, err := parseKeepAlive(req.KeepAlive)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	// No messages: Ollama's way of loading a model ahead of use, and with
	// keep_alive 0 of unloading it.
	if len(req.Messages) == 0 {
		reason, status, err := s.loadOrUnload(req.Model, keepAlive)
		if err != nil {
			writeJSON(w, status, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, ollamaChatResponse{Model: req.Model, CreatedAt: time.Now(),
			Message: ollamaMsg{Role: "assistant"}, Done: true, DoneReason: reason})
		return
	}

	loadStart := time.Now()
	eng, release, err := s.acquire(req.Model)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
		return
	}
	defer release()
	// Runs before release: the model's idle clock starts when the last
	// request lets go, and it runs for what that request asked.
	defer s.noteKeepAlive(req.Model, keepAlive)
	// Whatever acquire took is load time: reading a GGUF off disk, or waiting
	// behind another request's swap. It is near zero when the model was already
	// resident, which is the number a caller wants to see.
	load := time.Since(loadStart)

	msgs := toChatMessages(req.Messages)
	params := applyOllamaOptions(engine.DefaultGenParams(), req.Options)
	params.Tools = req.Tools
	params.NoThink = thinkOff(req.Think)

	if len(req.Tools) > 0 && eng.ToolFormat() == tools.None {
		// Declaring functions to a model that was never trained on this
		// convention produces prose where the caller is waiting for a call.
		// Saying so beats answering something useless.
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "this model's chat template does not use the <tool_call> convention, so it cannot answer with tool calls",
		})
		return
	}

	// Ollama streams unless told otherwise — the opposite of OpenAI.
	stream := req.Stream == nil || *req.Stream

	// A caller that did not ask to stream gets the reply whole, with its calls
	// parsed out. A caller that did gets the prose as it is written and the
	// calls in the final object — see tools.CallSplitter for why the calls
	// cannot stream and why the prose must.
	if len(req.Tools) > 0 && !stream {
		text, st, err := eng.ChatFull(r.Context(), msgs, params, nil)
		if err != nil {
			writeGenError(w, err)
			return
		}
		// Thinking first: a reasoning model decides on its tool call inside the
		// block, and a <tool_call> it merely considered there is not one it
		// made.
		thinking, answer := chat.SplitThinking(text)
		content, calls := eng.ToolFormat().Parse(answer)
		reason := doneReason(st)
		if len(calls) > 0 {
			reason = "tool_calls"
		}
		msg := ollamaMsg{Role: "assistant", Content: content, ToolCalls: wireCalls(calls)}
		if wantThinking(req.Think) {
			msg.Thinking = thinking
		}
		writeJSON(w, http.StatusOK, ollamaChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now(),
			Message:    msg,
			Done:       true,
			DoneReason: reason,
		}.withTimings(st, load, time.Since(start)))
		return
	}

	if !stream {
		text, st, err := eng.ChatFull(r.Context(), msgs, params, nil)
		if err != nil {
			writeGenError(w, err)
			return
		}
		thinking, content := chat.SplitThinking(text)
		msg := ollamaMsg{Role: "assistant", Content: content}
		if wantThinking(req.Think) {
			msg.Thinking = thinking
		}
		writeJSON(w, http.StatusOK, ollamaChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now(),
			Message:    msg,
			Done:       true,
			DoneReason: doneReason(st),
		}.withTimings(st, load, time.Since(start)))
		return
	}

	// Ollama's stream format is newline-delimited JSON, not SSE.
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "streaming unsupported"})
		return
	}
	// The header is written on the first thing actually sent, not here. A full
	// queue is refused at admission, before any token exists, and deferring
	// leaves that one case able to answer 503 instead of a 200 whose body
	// quietly says the server was busy. Go would write an implicit 200 on the
	// first Write in any case; this only makes the moment explicit.
	streamed := false
	begin := func() {
		if streamed {
			return
		}
		streamed = true
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
	}

	enc := json.NewEncoder(w)
	showThinking := wantThinking(req.Think)
	var split chat.ThinkSplitter
	var calls tools.CallSplitter // only consulted when functions are on the table

	emit := func(thinking, content string) {
		if len(req.Tools) > 0 {
			content = calls.Next(content)
		}
		if content == "" && (thinking == "" || !showThinking) {
			return
		}
		begin()
		msg := ollamaMsg{Role: "assistant", Content: content}
		if showThinking {
			msg.Thinking = thinking
		}
		_ = enc.Encode(ollamaChatResponse{Model: req.Model, CreatedAt: time.Now(), Message: msg})
		flusher.Flush()
	}

	text, st, genErr := eng.ChatFull(r.Context(), msgs, params, func(frag string) {
		emit(split.Next(frag))
	})
	emit(split.Flush())
	if len(req.Tools) > 0 {
		if rest := calls.Flush(); rest != "" {
			begin()
			_ = enc.Encode(ollamaChatResponse{Model: req.Model, CreatedAt: time.Now(),
				Message: ollamaMsg{Role: "assistant", Content: rest}})
			flusher.Flush()
		}
	}
	if refusedBeforeStreaming(genErr, streamed) {
		writeGenError(w, genErr)
		return
	}
	begin()
	// The terminal object is the only one that carries counts, and it is the
	// one a streaming client reads them from — `ollama run` streams, so this is
	// the path its verbose line is computed from. It is also where the tool
	// calls go, parsed from the whole reply once it is complete: Ollama's
	// shape, and the one a client reading Ollama already handles.
	final := ollamaChatResponse{
		Model:      req.Model,
		CreatedAt:  time.Now(),
		Message:    ollamaMsg{Role: "assistant"},
		Done:       true,
		DoneReason: doneReason(st),
	}.withTimings(st, load, time.Since(start))
	if len(req.Tools) > 0 && genErr == nil {
		_, answer := chat.SplitThinking(text)
		if _, cs := eng.ToolFormat().Parse(answer); len(cs) > 0 {
			final.Message.ToolCalls = wireCalls(cs)
			final.DoneReason = "tool_calls"
		}
	}
	if genErr != nil {
		// Headers are already out, so the error can only travel as a final
		// frame. It MUST travel: a 200 that ends in an empty message is
		// indistinguishable from the model choosing to say nothing.
		log.Printf("generation failed: %v", genErr)
		final.DoneReason = "error"
		final.Error = genErr.Error()
	}
	_ = enc.Encode(final)
	flusher.Flush()
}

// handleOllamaTags lists everything this server can answer for.
//
// All rather than List: Resolve accepts a model borrowed from Ollama's store,
// so the list a caller checks before choosing has to offer the same names, or
// it concludes a model is unavailable that a request for it would have served.
func (s *Server) handleOllamaTags(w http.ResponseWriter, r *http.Request) {
	models, err := s.store.All()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	type tag struct {
		Name       string    `json:"name"`
		Model      string    `json:"model"`
		Size       int64     `json:"size"`
		ModifiedAt time.Time `json:"modified_at"`
	}
	tags := make([]tag, len(models))
	for i, m := range models {
		tags[i] = tag{Name: m.Name, Model: m.Name, Size: m.Size, ModifiedAt: m.Modified}
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": tags})
}

// handlePS reports what is loaded right now — the question you ask when a
// request was slower than expected and you want to know whether it paid for a
// model load.
func (s *Server) handlePS(w http.ResponseWriter, r *http.Request) {
	type entry struct {
		Name      string     `json:"name"`
		Model     string     `json:"model"`
		Size      int64      `json:"size"`
		ExpiresAt *time.Time `json:"expires_at,omitempty"`
	}

	s.mu.Lock()
	var loaded []entry
	if s.cur != nil {
		e := entry{Name: modelName(s.cur.path), Model: modelName(s.cur.path)}
		if fi, err := os.Stat(s.cur.path); err == nil {
			e.Size = fi.Size()
		}
		if !s.expiry.IsZero() {
			t := s.expiry
			e.ExpiresAt = &t
		}
		loaded = append(loaded, e)
	}
	s.mu.Unlock()

	if loaded == nil {
		loaded = []entry{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": loaded})
}

// handleShow answers the capability probe many Ollama clients make before their
// first chat. Returning 404 here leaves a client to guess, and some of them
// guess by disabling features.
func (s *Server) handleShow(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
		Name  string `json:"name"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	name := req.Model
	if name == "" {
		name = req.Name
	}

	path, err := s.store.Resolve(name)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"details": map[string]any{
			"family":             chat.Detect(path).Name,
			"format":             "gguf",
			"parameter_size":     "",
			"quantization_level": "",
		},
		"model_info":   map[string]any{},
		"capabilities": []string{"completion"},
	})
}

func modelName(path string) string {
	return strings.TrimSuffix(filepath.Base(path), ".gguf")
}

// ─── OpenAI dialect ──────────────────────────────────────────────────────────

type openAIChatRequest struct {
	Model       string          `json:"model"`
	Messages    []ollamaMsg     `json:"messages"`
	Stream      bool            `json:"stream"`
	Temperature *float64        `json:"temperature"`
	TopP        *float64        `json:"top_p"`
	MaxTokens   *int            `json:"max_tokens"`
	Seed        *int64          `json:"seed"`
	Tools       []tools.Tool    `json:"tools"`
	Think       json.RawMessage `json:"think"`

	// StreamOptions is where a streaming caller asks for token counts. OpenAI
	// sends usage on a stream only when include_usage is set, in one extra
	// chunk after the last content — so a client that did not ask must not get
	// it, and one that did must not have to guess.
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

// usage is the OpenAI dialect's token accounting. Clients bill, budget and
// trim context against it; an object of zeros tells a caller its whole
// conversation cost nothing, which it then believes.
func usage(st engine.Stats) map[string]any {
	return map[string]any{
		"prompt_tokens":     st.PromptTokens,
		"completion_tokens": st.EvalTokens,
		"total_tokens":      st.PromptTokens + st.EvalTokens,
	}
}

func (s *Server) handleOpenAIChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req openAIChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{"message": err.Error()}})
		return
	}

	eng, release, err := s.acquire(req.Model)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]any{"error": map[string]string{"message": err.Error()}})
		return
	}
	defer release()

	msgs := toChatMessages(req.Messages)

	params := engine.DefaultGenParams()
	params.Tools = req.Tools
	params.NoThink = thinkOff(req.Think)
	if req.Temperature != nil {
		params.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = float32(*req.TopP)
	}
	if req.MaxTokens != nil {
		params.MaxTokens = *req.MaxTokens
	}
	if req.Seed != nil {
		params.Seed = *req.Seed
	}

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	if len(req.Tools) > 0 && eng.ToolFormat() == tools.None {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{
			"message": "this model's chat template does not use the <tool_call> convention, so it cannot answer with tool calls",
		}})
		return
	}

	// As in the Ollama dialect: a caller that did not ask to stream gets the
	// reply whole with its calls parsed; one that did gets prose as it is
	// written and the calls in the closing chunk.
	if len(req.Tools) > 0 && !req.Stream {
		text, st, err := eng.ChatFull(r.Context(), msgs, params, nil)
		if err != nil {
			writeOpenAIGenError(w, err)
			return
		}
		thinking, answer := chat.SplitThinking(text)
		content, calls := eng.ToolFormat().Parse(answer)

		msg := map[string]any{"role": "assistant", "content": content}
		if wantThinking(req.Think) && thinking != "" {
			withReasoning(msg, thinking)
		}
		finish := openAIFinishReason(st)
		if len(calls) > 0 {
			finish = "tool_calls"
			wire := make([]any, len(calls))
			for i, c := range calls {
				wire[i] = map[string]any{
					"id": fmt.Sprintf("call_%d_%d", created, i), "type": "function",
					"function": map[string]any{"name": c.Name, "arguments": string(c.Arguments)},
				}
			}
			msg["tool_calls"] = wire
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": req.Model,
			"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": finish}},
			"usage":   usage(st),
		})
		return
	}

	if !req.Stream {
		text, st, err := eng.ChatFull(r.Context(), msgs, params, nil)
		if err != nil {
			writeOpenAIGenError(w, err)
			return
		}
		thinking, content := chat.SplitThinking(text)
		msg := map[string]any{"role": "assistant", "content": content}
		if wantThinking(req.Think) && thinking != "" {
			withReasoning(msg, thinking)
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": req.Model,
			"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": openAIFinishReason(st)}},
			"usage":   usage(st),
		})
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]string{"message": "streaming unsupported"}})
		return
	}
	send := func(delta map[string]any, finish any) {
		payload, _ := json.Marshal(map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": req.Model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
		})
		fmt.Fprintf(w, "data: %s\n\n", payload)
		flusher.Flush()
	}

	// Nothing is written until there is something to write, including OpenAI's
	// opening role chunk. A refused request has produced no tokens, so leaving
	// the header unwritten is what lets it answer 503 rather than a 200 whose
	// body reports the server was busy.
	streamed := false
	begin := func() {
		if streamed {
			return
		}
		streamed = true
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		send(map[string]any{"role": "assistant"}, nil)
	}

	chunk := func(delta map[string]any, finish any) {
		begin()
		send(delta, finish)
	}

	showThinking := wantThinking(req.Think)
	var split chat.ThinkSplitter
	var calls tools.CallSplitter
	emit := func(thinking, content string) {
		if len(req.Tools) > 0 {
			content = calls.Next(content)
		}
		if content != "" {
			chunk(map[string]any{"content": content}, nil)
		}
		if showThinking && thinking != "" {
			chunk(withReasoning(map[string]any{}, thinking), nil)
		}
	}

	text, st, genErr := eng.ChatFull(r.Context(), msgs, params, func(frag string) {
		emit(split.Next(frag))
	})
	if err := genErr; err != nil {
		if refusedBeforeStreaming(err, streamed) {
			writeOpenAIGenError(w, err)
			return
		}
		// Reporting finish_reason "stop" here would claim the model finished
		// normally. Emit an error event first — the shape OpenAI clients
		// already understand — then close the stream.
		log.Printf("generation failed: %v", err)
		begin()
		payload, _ := json.Marshal(map[string]any{
			"error": map[string]string{"message": err.Error(), "type": "generation_error"},
		})
		fmt.Fprintf(w, "data: %s\n\n", payload)
		chunk(map[string]any{}, "error")
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}
	emit(split.Flush())
	if len(req.Tools) > 0 {
		if rest := calls.Flush(); rest != "" {
			chunk(map[string]any{"content": rest}, nil)
		}
		_, answer := chat.SplitThinking(text)
		if _, cs := eng.ToolFormat().Parse(answer); len(cs) > 0 {
			// OpenAI streams calls as deltas carrying an index; all of them fit
			// in one delta here because they are only known once complete.
			wire := make([]any, len(cs))
			for i, c := range cs {
				wire[i] = map[string]any{
					"index": i, "id": fmt.Sprintf("call_%d_%d", created, i), "type": "function",
					"function": map[string]any{"name": c.Name, "arguments": string(c.Arguments)},
				}
			}
			chunk(map[string]any{"tool_calls": wire}, nil)
			chunk(map[string]any{}, "tool_calls")
			goto usage
		}
	}
	chunk(map[string]any{}, openAIFinishReason(st))
usage:
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		// OpenAI's shape for this: one last chunk carrying no choices at all,
		// only the counts.
		payload, _ := json.Marshal(map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": req.Model,
			"choices": []any{}, "usage": usage(st),
		})
		fmt.Fprintf(w, "data: %s\n\n", payload)
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// withReasoning puts a reasoning model's thinking where OpenAI-dialect clients
// look for it, and returns m for use inline. OpenAI has no standard field.
// reasoning_content is DeepSeek's, and most clients built against that API read
// it; reasoning is what Ollama's own /v1 endpoint writes (0.34.3, measured), so
// a client that was pointed at Ollama before kinfer finds it where it looked.
// Both, because a caller that finds neither concludes the model said nothing.
func withReasoning(m map[string]any, thinking string) map[string]any {
	m["reasoning_content"] = thinking
	m["reasoning"] = thinking
	return m
}

func (s *Server) handleOpenAIModels(w http.ResponseWriter, r *http.Request) {
	models, err := s.store.List()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]string{"message": err.Error()}})
		return
	}

	data := make([]any, len(models))
	for i, m := range models {
		data[i] = map[string]any{
			"id": m.Name, "object": "model", "created": m.Modified.Unix(), "owned_by": "kinfer",
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// ─── helpers ─────────────────────────────────────────────────────────────────

func applyOllamaOptions(p engine.GenParams, o *ollamaOptions) engine.GenParams {
	if o == nil {
		return p
	}
	if o.Temperature != nil {
		p.Temperature = float32(*o.Temperature)
	}
	if o.TopP != nil {
		p.TopP = float32(*o.TopP)
	}
	if o.TopK != nil {
		p.TopK = *o.TopK
	}
	if o.RepeatPenalty != nil {
		p.RepeatPenalty = float32(*o.RepeatPenalty)
	}
	if o.NumPredict != nil {
		p.MaxTokens = *o.NumPredict
	}
	if o.Seed != nil {
		p.Seed = *o.Seed
	}
	return p
}

// doneReason names why generation ended. "length" is what both dialects use for
// a reply cut short by the token budget, and it is the only signal a caller has
// that a reasoning model returned nothing because it spent the whole budget
// thinking rather than because it had nothing to say.
// doneReason is why a reply ended, in the Ollama dialect's vocabulary.
//
// "timeout" is kinfer's own: Ollama has no word for a reply the server cut off
// on the clock, and the alternatives both lie. "stop" would say the model
// finished, and "length" would say it hit the token budget it was given. The
// whole reason this runtime exists is an incomplete answer that looked complete.
func doneReason(st engine.Stats) string {
	switch {
	case st.Timeout:
		return "timeout"
	case st.Truncated:
		return "length"
	default:
		return "stop"
	}
}

// openAIFinishReason is the same fact in OpenAI's vocabulary, which is a closed
// set its clients switch on — "stop", "length", "tool_calls", "content_filter".
// A timeout is reported as "length": not the whole truth, but true in the part
// that matters to a caller, which is that the reply is incomplete and was cut
// off by a limit rather than by the model.
func openAIFinishReason(st engine.Stats) string {
	if st.Timeout {
		return "length"
	}
	return doneReason(st)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
