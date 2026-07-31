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
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// Server holds the model store and whatever is currently loaded.
type Server struct {
	store *store.Store
	opts  engine.Options

	// Exactly one model stays resident. A 7B model is several GB, so keeping a
	// map of them would exhaust memory on the laptops kinfer targets; loading is
	// fast enough (0.5s for a 0.5B, a few seconds for a 7B) that swapping on
	// demand is the better trade.
	mu      sync.Mutex
	current *engine.Engine
}

// New creates a server over the given store.
func New(s *store.Store, opts engine.Options) *Server {
	return &Server{store: s, opts: opts}
}

// Handler returns the HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// Ollama dialect.
	mux.HandleFunc("/api/chat", s.handleOllamaChat)
	mux.HandleFunc("/api/tags", s.handleOllamaTags)
	mux.HandleFunc("/api/version", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"version": "kinfer"})
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
	if s.current != nil {
		s.current.Close()
		s.current = nil
	}
}

// acquire returns an engine for the named model, loading it if necessary.
//
// The returned engine stays valid because Engine.Chat serialises internally and
// swaps only happen here, under the same lock.
func (s *Server) acquire(name string) (*engine.Engine, error) {
	path, err := s.store.Resolve(name)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.current != nil && s.current.Path() == path {
		return s.current, nil
	}
	if s.current != nil {
		s.current.Close()
		s.current = nil
	}

	log.Printf("loading %s", path)
	e, err := engine.Open(path, s.opts)
	if err != nil {
		return nil, err
	}
	s.current = e
	return e, nil
}

// ─── Ollama dialect ──────────────────────────────────────────────────────────

type ollamaChatRequest struct {
	Model    string         `json:"model"`
	Messages []ollamaMsg    `json:"messages"`
	Stream   *bool          `json:"stream"`
	Options  *ollamaOptions `json:"options"`
}

type ollamaMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Message   ollamaMsg `json:"message"`
	Done      bool      `json:"done"`
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

	eng, err := s.acquire(req.Model)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
		return
	}

	msgs := make([]chat.Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = chat.Message{Role: m.Role, Content: m.Content}
	}
	params := applyOllamaOptions(engine.DefaultGenParams(), req.Options)

	// Ollama streams unless told otherwise — the opposite of OpenAI.
	stream := req.Stream == nil || *req.Stream

	if !stream {
		text, err := eng.Chat(r.Context(), msgs, params, nil)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, ollamaChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now(),
			Message:   ollamaMsg{Role: "assistant", Content: text},
			Done:      true,
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
	_, genErr := eng.Chat(r.Context(), msgs, params, func(frag string) {
		_ = enc.Encode(ollamaChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now(),
			Message:   ollamaMsg{Role: "assistant", Content: frag},
		})
		flusher.Flush()
	})
	if genErr != nil {
		// Headers are already out, so the error can only travel as a final
		// frame. Clients that check `done` will still terminate cleanly.
		log.Printf("generation failed: %v", genErr)
	}
	_ = enc.Encode(ollamaChatResponse{
		Model:     req.Model,
		CreatedAt: time.Now(),
		Message:   ollamaMsg{Role: "assistant"},
		Done:      true,
	})
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

// ─── OpenAI dialect ──────────────────────────────────────────────────────────

type openAIChatRequest struct {
	Model       string      `json:"model"`
	Messages    []ollamaMsg `json:"messages"`
	Stream      bool        `json:"stream"`
	Temperature *float64    `json:"temperature"`
	TopP        *float64    `json:"top_p"`
	MaxTokens   *int        `json:"max_tokens"`
	Seed        *int64      `json:"seed"`
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

	eng, err := s.acquire(req.Model)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]any{"error": map[string]string{"message": err.Error()}})
		return
	}

	msgs := make([]chat.Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = chat.Message{Role: m.Role, Content: m.Content}
	}

	params := engine.DefaultGenParams()
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

	if !req.Stream {
		text, err := eng.Chat(r.Context(), msgs, params, nil)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]string{"message": err.Error()}})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": req.Model,
			"choices": []any{map[string]any{
				"index":         0,
				"message":       ollamaMsg{Role: "assistant", Content: text},
				"finish_reason": "stop",
			}},
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
	if _, err := eng.Chat(r.Context(), msgs, params, func(frag string) {
		chunk(map[string]any{"content": frag}, nil)
	}); err != nil {
		log.Printf("generation failed: %v", err)
	}
	chunk(map[string]any{}, "stop")
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

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
