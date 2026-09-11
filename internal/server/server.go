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
	ChatFull(ctx context.Context, msgs []chat.Message, p engine.GenParams, onToken func(string)) (string, bool, error)
	ToolFormat() tools.Format
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
	mux.HandleFunc("/api/tags", s.handleOllamaTags)
	mux.HandleFunc("/api/ps", s.handlePS)
	mux.HandleFunc("/api/show", s.handleShow)
	mux.HandleFunc("/api/version", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"version": Version})
	})

	// OpenAI dialect.
	mux.HandleFunc("/v1/chat/completions", s.handleOpenAIChat)
	mux.HandleFunc("/v1/models", s.handleOpenAIModels)

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	return mux
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
	if s.keepAlive <= 0 || s.cur == nil || s.cur.refs != 1 {
		return
	}
	l := s.cur
	s.expiry = time.Now().Add(s.keepAlive)
	s.idle = time.AfterFunc(s.keepAlive, func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		// Re-check under the lock: a request may have arrived while the timer
		// was firing, in which case the model is in use again.
		if s.cur == l && l.refs == 1 {
			log.Printf("unloading %s after %s idle", l.path, s.keepAlive)
			s.retireLocked()
		}
	})
}

func (s *Server) stopIdleLocked() {
	if s.idle != nil {
		s.idle.Stop()
		s.idle = nil
	}
	s.expiry = time.Time{}
}

// acquire returns an engine for the named model, loading it if necessary, plus
// a release function the caller MUST call when it is done generating.
//
// The engine stays alive for as long as the reference is held. Swapping to a
// different model retires the current one and waits for its in-flight requests
// to finish before loading the replacement — only one model fits in memory, and
// freeing one that is still generating crashes the process.
func (s *Server) acquire(name string) (generator, func(), error) {
	path, err := s.store.Resolve(name)
	if err != nil {
		return nil, nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for {
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

	// Think asks for a reasoning model's working out to be returned separately
	// rather than dropped. Ollama takes true/false or a level string; only
	// presence matters here, since kinfer does not instruct the model either
	// way — it splits whatever the model produced.
	Think json.RawMessage `json:"think"`
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

// wantThinking reports whether the caller asked to see a reasoning model's
// working out. Absent or false, it is split off and dropped: it is not the
// reply, and a client that prints replies verbatim would show the model talking
// to itself.
func wantThinking(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
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

	// Error travels in-band because the response headers are long gone by the
	// time generation can fail. Without it a failure is indistinguishable from
	// an empty reply — which is precisely how a context overflow reached
	// LocalKin as a silent empty answer over a 200 OK.
	Error string `json:"error,omitempty"`
}

func (s *Server) handleOllamaChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	log.Printf("→ /api/chat from %s", r.RemoteAddr)

	var req ollamaChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("  decode failed: %v", err)
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	eng, release, err := s.acquire(req.Model)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
		return
	}
	defer release()

	msgs := toChatMessages(req.Messages)
	params := applyOllamaOptions(engine.DefaultGenParams(), req.Options)
	params.Tools = req.Tools

	if len(req.Tools) > 0 && eng.ToolFormat() == tools.None {
		// Declaring functions to a model that was never trained on this
		// convention produces prose where the caller is waiting for a call.
		// Saying so beats answering something useless.
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "this model's chat template does not use the <tool_call> convention, so it cannot answer with tool calls",
		})
		return
	}

	// A tool call is JSON split across many tokens; streaming it would hand the
	// client fragments of syntax. With functions on the table the reply is
	// buffered and delivered once, parsed.
	if len(req.Tools) > 0 {
		text, truncated, err := eng.ChatFull(r.Context(), msgs, params, nil)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
			return
		}
		// Thinking first: a reasoning model decides on its tool call inside the
		// block, and a <tool_call> it merely considered there is not one it
		// made.
		thinking, answer := chat.SplitThinking(text)
		content, calls := eng.ToolFormat().Parse(answer)
		reason := doneReason(truncated)
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
		})
		return
	}

	// Ollama streams unless told otherwise — the opposite of OpenAI.
	stream := req.Stream == nil || *req.Stream

	if !stream {
		text, truncated, err := eng.ChatFull(r.Context(), msgs, params, nil)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
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
			DoneReason: doneReason(truncated),
		})
		return
	}

	// Ollama's stream format is newline-delimited JSON, not SSE.
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "streaming unsupported"})
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)

	enc := json.NewEncoder(w)
	showThinking := wantThinking(req.Think)
	var split chat.ThinkSplitter

	emit := func(thinking, content string) {
		if content == "" && (thinking == "" || !showThinking) {
			return
		}
		msg := ollamaMsg{Role: "assistant", Content: content}
		if showThinking {
			msg.Thinking = thinking
		}
		_ = enc.Encode(ollamaChatResponse{Model: req.Model, CreatedAt: time.Now(), Message: msg})
		flusher.Flush()
	}

	_, truncated, genErr := eng.ChatFull(r.Context(), msgs, params, func(frag string) {
		emit(split.Next(frag))
	})
	emit(split.Flush())
	final := ollamaChatResponse{
		Model:      req.Model,
		CreatedAt:  time.Now(),
		Message:    ollamaMsg{Role: "assistant"},
		Done:       true,
		DoneReason: doneReason(truncated),
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

func (s *Server) handleOllamaTags(w http.ResponseWriter, r *http.Request) {
	models, err := s.store.List()
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

	// As in the Ollama dialect: a tool call arrives as JSON across many tokens,
	// so with functions on the table the reply is buffered and parsed.
	if len(req.Tools) > 0 {
		text, truncated, err := eng.ChatFull(r.Context(), msgs, params, nil)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]string{"message": err.Error()}})
			return
		}
		thinking, answer := chat.SplitThinking(text)
		content, calls := eng.ToolFormat().Parse(answer)

		msg := map[string]any{"role": "assistant", "content": content}
		if wantThinking(req.Think) && thinking != "" {
			msg["reasoning_content"] = thinking
		}
		finish := doneReason(truncated)
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
		})
		return
	}

	if !req.Stream {
		text, truncated, err := eng.ChatFull(r.Context(), msgs, params, nil)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]string{"message": err.Error()}})
			return
		}
		thinking, content := chat.SplitThinking(text)
		msg := map[string]any{"role": "assistant", "content": content}
		if wantThinking(req.Think) && thinking != "" {
			// OpenAI has no standard field for this; reasoning_content is what
			// DeepSeek introduced and what most clients now look for.
			msg["reasoning_content"] = thinking
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": req.Model,
			"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": doneReason(truncated)}},
		})
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]string{"message": "streaming unsupported"}})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	chunk := func(delta map[string]any, finish any) {
		payload, _ := json.Marshal(map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": req.Model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
		})
		fmt.Fprintf(w, "data: %s\n\n", payload)
		flusher.Flush()
	}

	chunk(map[string]any{"role": "assistant"}, nil)

	showThinking := wantThinking(req.Think)
	var split chat.ThinkSplitter
	emit := func(thinking, content string) {
		if content != "" {
			chunk(map[string]any{"content": content}, nil)
		}
		if showThinking && thinking != "" {
			chunk(map[string]any{"reasoning_content": thinking}, nil)
		}
	}

	_, truncated, genErr := eng.ChatFull(r.Context(), msgs, params, func(frag string) {
		emit(split.Next(frag))
	})
	if err := genErr; err != nil {
		// Reporting finish_reason "stop" here would claim the model finished
		// normally. Emit an error event first — the shape OpenAI clients
		// already understand — then close the stream.
		log.Printf("generation failed: %v", err)
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
	chunk(map[string]any{}, doneReason(truncated))
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
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
func doneReason(truncated bool) string {
	if truncated {
		return "length"
	}
	return "stop"
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
