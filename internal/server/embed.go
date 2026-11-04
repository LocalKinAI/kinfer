package server

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"time"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// ─── embeddings ──────────────────────────────────────────────────────────────
//
//	POST /api/embed       Ollama's: {"model", "input": a text or texts, "truncate", "dimensions", "keep_alive"}
//	POST /api/embeddings  Ollama's older one: {"model", "prompt"} → {"embedding"}, as the model pools it
//	POST /v1/embeddings   OpenAI's: "input" a text, texts, or tokens; "encoding_format" float or base64
//
// An embedding model is small — nomic-embed-text 262 MB, EmbeddingGemma 1.3 GB
// — and asked for between a chat model's turns by the same agent, which looks
// up what to put in the prompt it is about to send. So it stays loaded beside
// the chat model in a place of its own, with its own idle clock, instead of
// swapping the chat model out: Flash-Next takes half a minute to load again.
// One embedding model is loaded at a time.

// embedder is the part of engine.Embedder this package uses; tests replace it.
type embedder interface {
	Embed(ctx context.Context, inputs []engine.EmbedInput, truncate bool) ([][]float32, int, error)
	Close()
}

func openEmbedder(path string, opts engine.Options) (embedder, error) {
	return engine.OpenEmbedder(path, opts)
}

// embedLease is the loaded embedding model and its holders: each request
// using it, and the server while it stays loaded. See lease.
type embedLease struct {
	emb       embedder
	path      string
	refs      int
	keepAlive *time.Duration
}

// modelMissing is a name the store resolves to nothing.
type modelMissing struct{ error }

func (m modelMissing) Unwrap() error { return m.error }

// acquireEmbedder returns the embedding model name names, loading it when
// another is loaded or none is, and how long that took. The caller must hand
// the lease back with releaseEmbedder.
func (s *Server) acquireEmbedder(name string) (*embedLease, time.Duration, error) {
	path, err := s.store.Resolve(name)
	if err != nil {
		return nil, 0, modelMissing{err}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for {
		if l := s.emb; l != nil && l.path == path {
			l.refs++
			s.stopEmbedIdleLocked()
			return l, 0, nil
		}
		if !s.embLoading {
			break
		}
		s.cond.Wait()
	}
	s.embLoading = true
	defer func() {
		s.embLoading = false
		s.cond.Broadcast()
	}()
	s.retireEmbedderLocked()

	s.mu.Unlock()
	log.Printf("loading %s to embed with", path)
	start := time.Now()
	e, err := s.openEmbed(path, s.opts)
	load := time.Since(start)
	s.mu.Lock()
	if err != nil {
		return nil, 0, err
	}
	l := &embedLease{emb: e, path: path, refs: 2} // one for the server, one for the caller
	s.emb = l
	return l, load, nil
}

// releaseEmbedder hands back a lease, recording what the request asked for its
// idle clock (see noteKeepAlive).
func (s *Server) releaseEmbedder(l *embedLease, keepAlive *time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.emb == l {
		l.keepAlive = keepAlive
	}
	s.dropEmbedderLocked(l)
}

func (s *Server) dropEmbedderLocked(l *embedLease) {
	l.refs--
	if l.refs == 0 {
		l.emb.Close()
	}
	s.armEmbedIdleLocked()
	s.cond.Broadcast()
}

// retireEmbedderLocked lets go of the loaded embedding model, and waits for
// the requests still using it. Caller holds s.mu; the wait releases it.
func (s *Server) retireEmbedderLocked() {
	s.stopEmbedIdleLocked()
	old := s.emb
	if old == nil {
		return
	}
	s.emb = nil
	s.dropEmbedderLocked(old)
	for old.refs > 0 {
		s.cond.Wait()
	}
}

// armEmbedIdleLocked starts the embedding model's unload countdown once only
// the server holds it, as armIdleLocked does for the chat model.
func (s *Server) armEmbedIdleLocked() {
	s.stopEmbedIdleLocked()
	l := s.emb
	if l == nil || l.refs != 1 {
		return
	}
	d, forever := s.idleFor(l.keepAlive)
	if forever {
		return
	}
	s.embExpiry = time.Now().Add(d)
	s.embIdle = time.AfterFunc(d, func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.emb == l && l.refs == 1 {
			log.Printf("unloading %s after %s idle", l.path, d)
			s.retireEmbedderLocked()
		}
	})
}

func (s *Server) stopEmbedIdleLocked() {
	if s.embIdle != nil {
		s.embIdle.Stop()
		s.embIdle = nil
	}
	s.embExpiry = time.Time{}
}

// embedInputs reads an "input" field: a text, a list of texts, the tokens of
// a text (OpenAI's list of numbers), or a list of those. Absent, or an empty
// text, is no input: the request only loads the model, as in Ollama.
func embedInputs(raw json.RawMessage) ([]engine.EmbedInput, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		if text == "" {
			return nil, nil
		}
		return []engine.EmbedInput{{Text: text}}, nil
	}
	var items []json.RawMessage
	if json.Unmarshal(raw, &items) != nil {
		return nil, errors.New("input: want a text, a list of texts, or tokens")
	}
	var tokens []int32
	if len(items) > 0 && json.Unmarshal(raw, &tokens) == nil {
		return []engine.EmbedInput{{Tokens: tokens}}, nil
	}
	out := make([]engine.EmbedInput, len(items))
	for i, item := range items {
		var media map[string]json.RawMessage
		switch {
		case json.Unmarshal(item, &out[i].Text) == nil:
		case json.Unmarshal(item, &out[i].Tokens) == nil && len(out[i].Tokens) > 0:
		case json.Unmarshal(item, &media) == nil && (media["image"] != nil || media["audio"] != nil):
			return nil, fmt.Errorf("input %d: kinfer embeds text; pictures and sound are not supported yet", i)
		default:
			return nil, fmt.Errorf("input %d: want a text or a list of tokens", i)
		}
	}
	return out, nil
}

// embedStatus is the HTTP status an embedding request failed with.
func embedStatus(err error) int {
	var missing modelMissing
	var bad engine.InputError
	switch {
	case errors.As(err, &missing):
		return http.StatusNotFound
	case errors.Is(err, engine.ErrNotEmbedding), errors.As(err, &bad):
		return http.StatusBadRequest
	default:
		return http.StatusInternalServerError
	}
}

// embed is what the three dialects share: the model's vectors for inputs,
// the tokens read, and how long loading the model took.
func (s *Server) embed(ctx context.Context, model string, inputs []engine.EmbedInput, truncate bool, keepAlive *time.Duration) ([][]float32, int, time.Duration, error) {
	l, load, err := s.acquireEmbedder(model)
	if err != nil {
		return nil, 0, 0, err
	}
	defer s.releaseEmbedder(l, keepAlive)
	if len(inputs) == 0 {
		return [][]float32{}, 0, load, nil
	}
	vecs, n, err := l.emb.Embed(ctx, inputs, truncate)
	return vecs, n, load, err
}

type ollamaEmbedRequest struct {
	Model      string          `json:"model"`
	Input      json.RawMessage `json:"input"`
	Truncate   *bool           `json:"truncate"`
	Dimensions int             `json:"dimensions"`
	KeepAlive  json.RawMessage `json:"keep_alive"`
}

func (s *Server) handleOllamaEmbed(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req ollamaEmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON: " + err.Error()})
		return
	}
	keepAlive, err := parseKeepAlive(req.KeepAlive)
	if err == nil && req.Dimensions < 0 {
		err = errors.New("dimensions must not be negative")
	}
	var inputs []engine.EmbedInput
	if err == nil {
		inputs, err = embedInputs(req.Input)
	}
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	truncate := req.Truncate == nil || *req.Truncate
	vecs, n, load, err := s.embed(r.Context(), req.Model, inputs, truncate, keepAlive)
	if err != nil {
		writeJSON(w, embedStatus(err), map[string]string{"error": err.Error()})
		return
	}
	for i := range vecs {
		vecs[i] = engine.Shorten(engine.Normalize(vecs[i]), req.Dimensions)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"model":             req.Model,
		"embeddings":        vecs,
		"total_duration":    time.Since(start).Nanoseconds(),
		"load_duration":     load.Nanoseconds(),
		"prompt_eval_count": n,
	})
}

// handleOllamaEmbeddings is Ollama's first embedding route, still what older
// clients send. Its vector is the model's own, not normalized, as Ollama's is.
func (s *Server) handleOllamaEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model     string          `json:"model"`
		Prompt    string          `json:"prompt"`
		KeepAlive json.RawMessage `json:"keep_alive"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON: " + err.Error()})
		return
	}
	keepAlive, err := parseKeepAlive(req.KeepAlive)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	var inputs []engine.EmbedInput
	if req.Prompt != "" {
		inputs = []engine.EmbedInput{{Text: req.Prompt}}
	}
	vecs, _, _, err := s.embed(r.Context(), req.Model, inputs, true, keepAlive)
	if err != nil {
		writeJSON(w, embedStatus(err), map[string]string{"error": err.Error()})
		return
	}
	vec := []float32{}
	if len(vecs) > 0 {
		vec = vecs[0]
	}
	writeJSON(w, http.StatusOK, map[string]any{"embedding": vec})
}

type openAIEmbedRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format"`
	Dimensions     int             `json:"dimensions"`
}

func (s *Server) handleOpenAIEmbeddings(w http.ResponseWriter, r *http.Request) {
	fail := func(status int, msg string) {
		writeJSON(w, status, map[string]any{"error": map[string]string{"message": msg, "type": "invalid_request_error"}})
	}
	var req openAIEmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	inputs, err := embedInputs(req.Input)
	switch {
	case err != nil:
		fail(http.StatusBadRequest, err.Error())
		return
	case len(inputs) == 0:
		fail(http.StatusBadRequest, "input is required")
		return
	case req.EncodingFormat != "" && req.EncodingFormat != "float" && req.EncodingFormat != "base64":
		fail(http.StatusBadRequest, fmt.Sprintf("encoding_format is float or base64, not %q", req.EncodingFormat))
		return
	case req.Dimensions < 0:
		fail(http.StatusBadRequest, "dimensions must not be negative")
		return
	}
	vecs, n, _, err := s.embed(r.Context(), req.Model, inputs, true, nil)
	if err != nil {
		fail(embedStatus(err), err.Error())
		return
	}
	data := make([]map[string]any, len(vecs))
	for i, v := range vecs {
		v = engine.Shorten(engine.Normalize(v), req.Dimensions)
		var out any = v
		if req.EncodingFormat == "base64" {
			out = float32Base64(v)
		}
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": out}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   data,
		"model":  req.Model,
		"usage":  map[string]int{"prompt_tokens": n, "total_tokens": n},
	})
}

// float32Base64 is OpenAI's base64 encoding of a vector: its float32s, little
// endian, which is what the OpenAI SDKs ask for unless told otherwise.
func float32Base64(v []float32) string {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return base64.StdEncoding.EncodeToString(b)
}
