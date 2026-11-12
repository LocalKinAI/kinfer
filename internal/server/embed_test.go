package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// fakeEmbedder pools every input to (3L, 4L, 0, 0), L its length: (0.6, 0.8,
// 0, 0) normalized, whatever the input, and its length when not.
type fakeEmbedder struct {
	path string

	mu     sync.Mutex
	closed bool
	broken bool
	seen   []engine.EmbedInput
}

func (f *fakeEmbedder) Embed(_ context.Context, inputs []engine.EmbedInput, truncate bool) ([][]float32, int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed {
		return nil, 0, fmt.Errorf("embed on a closed model")
	}
	f.seen = append(f.seen, inputs...)
	out := make([][]float32, len(inputs))
	n := 0
	for i, in := range inputs {
		if len(in.Media) > 0 { // as engine.Embedder, a text model, refuses them
			return nil, 0, engine.InputError{Err: fmt.Errorf("input %d: this model reads text, not pictures or sound", i)}
		}
		l := len(in.Text) + len(in.Tokens)
		if l > 100 && !truncate {
			return nil, 0, engine.InputError{Err: fmt.Errorf("input %d is too long", i)}
		}
		out[i] = []float32{3 * float32(l), 4 * float32(l), 0, 0}
		n += l
	}
	return out, n, nil
}

func (f *fakeEmbedder) Close() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closed = true
}

func (f *fakeEmbedder) Broken() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.broken
}

func (f *fakeEmbedder) isClosed() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.closed
}

// embedServer is newTestServer with "nomic" an embedding model; embedders
// holds each one it loaded.
func embedServer(t *testing.T) (*Server, *registry, func() []*fakeEmbedder) {
	t.Helper()
	srv, reg := newTestServer(t, "alpha", "nomic", "chatty")
	var mu sync.Mutex
	var loaded []*fakeEmbedder
	srv.openEmbed = func(path string, _ engine.Options) (embedder, error) {
		if modelName(path) == "chatty" {
			return nil, fmt.Errorf("chatty.gguf is %w", engine.ErrNotEmbedding)
		}
		f := &fakeEmbedder{path: path}
		mu.Lock()
		loaded = append(loaded, f)
		mu.Unlock()
		return f, nil
	}
	t.Cleanup(srv.Close)
	return srv, reg, func() []*fakeEmbedder {
		mu.Lock()
		defer mu.Unlock()
		return append([]*fakeEmbedder(nil), loaded...)
	}
}

func postJSON(t *testing.T, srv *Server, path, body string) (int, map[string]any) {
	t.Helper()
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", path, strings.NewReader(body)))
	var out map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatalf("%s answered %d with %q", path, w.Code, w.Body.String())
	}
	return w.Code, out
}

func vectors(t *testing.T, v any) [][]float64 {
	t.Helper()
	var out [][]float64
	b, _ := json.Marshal(v)
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("not a list of vectors: %v", v)
	}
	return out
}

func near(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-6 {
			return false
		}
	}
	return true
}

func TestOllamaEmbed(t *testing.T) {
	srv, _, embedders := embedServer(t)

	code, out := postJSON(t, srv, "/api/embed", `{"model":"nomic","input":"hello"}`)
	if code != http.StatusOK {
		t.Fatalf("one text: %d %v", code, out)
	}
	if v := vectors(t, out["embeddings"]); len(v) != 1 || !near(v[0], []float64{0.6, 0.8, 0, 0}) {
		t.Errorf("one text: %v, want one vector of unit length", v)
	}
	if out["prompt_eval_count"] != float64(5) || out["model"] != "nomic" {
		t.Errorf("one text: %v", out)
	}

	_, out = postJSON(t, srv, "/api/embed", `{"model":"nomic","input":["a","bb","ccc"],"dimensions":1}`)
	if v := vectors(t, out["embeddings"]); len(v) != 3 || !near(v[2], []float64{1}) {
		t.Errorf("three texts at 1 dimension: %v", v)
	}

	// No input loads the model and embeds nothing, as in Ollama.
	code, out = postJSON(t, srv, "/api/embed", `{"model":"nomic"}`)
	if code != http.StatusOK || len(vectors(t, out["embeddings"])) != 0 {
		t.Errorf("no input: %d %v", code, out)
	}
	if n := len(embedders()); n != 1 {
		t.Errorf("loaded the embedding model %d times; it stays loaded", n)
	}

	code, out = postJSON(t, srv, "/api/embed", `{"model":"nomic","input":[{"image":"aGk="}]}`)
	if code != http.StatusBadRequest || !strings.Contains(fmt.Sprint(out["error"]), "pictures") {
		t.Errorf("a picture: %d %v", code, out)
	}
	code, out = postJSON(t, srv, "/api/embed", `{"model":"nomic","input":"`+strings.Repeat("x", 101)+`","truncate":false}`)
	if code != http.StatusBadRequest {
		t.Errorf("too long, not to be cut: %d %v", code, out)
	}
}

// Ollama's first route answers with the vector as the model pools it.
func TestOllamaEmbeddingsIsNotNormalized(t *testing.T) {
	srv, _, _ := embedServer(t)
	code, out := postJSON(t, srv, "/api/embeddings", `{"model":"nomic","prompt":"abc"}`)
	if code != http.StatusOK {
		t.Fatalf("%d %v", code, out)
	}
	if v := vectors(t, []any{out["embedding"]}); !near(v[0], []float64{9, 12, 0, 0}) {
		t.Errorf("embedding %v, want the model's own (9, 12, 0, 0)", v[0])
	}
}

func TestOpenAIEmbeddings(t *testing.T) {
	srv, _, embedders := embedServer(t)

	code, out := postJSON(t, srv, "/v1/embeddings", `{"model":"nomic","input":["a","bb"]}`)
	if code != http.StatusOK || out["object"] != "list" {
		t.Fatalf("%d %v", code, out)
	}
	data := out["data"].([]any)
	if len(data) != 2 || data[1].(map[string]any)["index"] != float64(1) {
		t.Errorf("data %v", data)
	}
	if u := out["usage"].(map[string]any); u["prompt_tokens"] != float64(3) || u["total_tokens"] != float64(3) {
		t.Errorf("usage %v", u)
	}

	// The OpenAI SDKs ask for base64: the float32s, little endian.
	_, out = postJSON(t, srv, "/v1/embeddings", `{"model":"nomic","input":"hi","encoding_format":"base64"}`)
	raw, err := base64.StdEncoding.DecodeString(out["data"].([]any)[0].(map[string]any)["embedding"].(string))
	if err != nil || len(raw) != 16 {
		t.Fatalf("base64 embedding: %v, %d bytes", err, len(raw))
	}
	got := make([]float64, 4)
	for i := range got {
		got[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
	}
	if !near(got, []float64{0.6, 0.8, 0, 0}) {
		t.Errorf("base64 decodes to %v", got)
	}

	// Tokens, as OpenAI takes them: one text's, or several texts'.
	postJSON(t, srv, "/v1/embeddings", `{"model":"nomic","input":[101,2023,102]}`)
	postJSON(t, srv, "/v1/embeddings", `{"model":"nomic","input":[[101,102],[7]]}`)
	seen := embedders()[0].seen
	if len(seen) != 6 || len(seen[3].Tokens) != 3 || seen[3].Tokens[1] != 2023 || len(seen[5].Tokens) != 1 {
		t.Errorf("token inputs reached the model as %+v", seen[3:])
	}

	if code, _ := postJSON(t, srv, "/v1/embeddings", `{"model":"nomic"}`); code != http.StatusBadRequest {
		t.Errorf("no input: %d, want 400", code)
	}
}

func TestEmbedWithAModelThatDoesNotEmbed(t *testing.T) {
	srv, _, _ := embedServer(t)
	if code, out := postJSON(t, srv, "/api/embed", `{"model":"chatty","input":"hi"}`); code != http.StatusBadRequest {
		t.Errorf("a model that writes text: %d %v, want 400", code, out)
	}
	if code, _ := postJSON(t, srv, "/api/embed", `{"model":"nobody","input":"hi"}`); code != http.StatusNotFound {
		t.Errorf("a model that is not here: %d, want 404", code)
	}
}

// RAG asks for vectors between a chat model's turns. The chat model stays
// loaded: Flash-Next takes half a minute to come back.
func TestEmbeddingLeavesTheChatModelLoaded(t *testing.T) {
	srv, reg, _ := embedServer(t)
	if code, out := postJSON(t, srv, "/api/chat", `{"model":"alpha","stream":false,"messages":[{"role":"user","content":"hi"}]}`); code != http.StatusOK {
		t.Fatalf("chat: %d %v", code, out)
	}
	if code, out := postJSON(t, srv, "/api/embed", `{"model":"nomic","input":"hi"}`); code != http.StatusOK {
		t.Fatalf("embed: %d %v", code, out)
	}
	if reg.get("alpha").isClosed() || reg.loadCount() != 1 {
		t.Errorf("embedding unloaded the chat model (loads %d)", reg.loadCount())
	}
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("GET", "/api/ps", nil))
	if !strings.Contains(w.Body.String(), `"alpha"`) || !strings.Contains(w.Body.String(), `"nomic"`) {
		t.Errorf("ps does not list both: %s", w.Body.String())
	}
}

// writeEncoderHeader makes name.gguf a GGUF header of a BERT-style model.
func writeEncoderHeader(t *testing.T, dir, name string) {
	t.Helper()
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(0)) // tensors
	w(uint64(1)) // keys
	str("general.architecture")
	w(uint32(8))
	str("nomic-bert")
	if err := os.WriteFile(filepath.Join(dir, name+".gguf"), b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

// Asked to chat, an embedding model is refused before the chat model is let
// go — not after loading it in the chat model's place.
func TestChatRefusesAnEmbeddingModel(t *testing.T) {
	srv, reg, _ := embedServer(t)
	writeEncoderHeader(t, srv.store.Root(), "nomic")
	postJSON(t, srv, "/api/chat", `{"model":"alpha","stream":false,"messages":[{"role":"user","content":"hi"}]}`)

	code, out := postJSON(t, srv, "/api/chat", `{"model":"nomic","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	if code != http.StatusBadRequest || !strings.Contains(fmt.Sprint(out["error"]), "embedding model") {
		t.Errorf("chat with an embedding model: %d %v", code, out)
	}
	if reg.get("alpha").isClosed() || reg.loadCount() != 1 {
		t.Errorf("the refusal unloaded the chat model (loads %d)", reg.loadCount())
	}
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", "/api/show", strings.NewReader(`{"model":"nomic"}`)))
	if !strings.Contains(w.Body.String(), `"embedding"`) {
		t.Errorf("show does not say it embeds: %s", w.Body.String())
	}
}

func TestEmbeddingModelUnloadsAsAsked(t *testing.T) {
	srv, _, embedders := embedServer(t)

	// keep_alive 0 on the request itself: gone once it is answered.
	postJSON(t, srv, "/api/embed", `{"model":"nomic","input":"hi","keep_alive":0}`)
	waitFor(t, "the embedding model to unload after keep_alive 0", embedders()[0].isClosed)

	// ollama stop: the generate idiom, which must find it in its own place.
	postJSON(t, srv, "/api/embed", `{"model":"nomic","input":"hi"}`)
	if code, out := postJSON(t, srv, "/api/generate", `{"model":"nomic","keep_alive":0}`); code != http.StatusOK || out["done_reason"] != "unload" {
		t.Errorf("unload: %d %v", code, out)
	}
	if !embedders()[1].isClosed() {
		t.Error("ollama stop left the embedding model loaded")
	}
}

// Pictures and sounds come as Ollama's MLX embedding models take them: an
// object per input, base64 or a data URL, with optional text before them.
func TestEmbedInputsReadsPicturesAndSound(t *testing.T) {
	png := base64.StdEncoding.EncodeToString([]byte("\x89PNG"))
	in, err := embedInputs(json.RawMessage(`["a cat", {"image": "` + png + `"}, {"text": "hi", "audio": ["data:audio/wav;base64,` + png + `"]}]`))
	if err != nil {
		t.Fatal(err)
	}
	if len(in) != 3 || in[0].Text != "a cat" || len(in[1].Media) != 1 || in[1].Media[0].Kind != "image" ||
		string(in[1].Media[0].Data) != "\x89PNG" || in[2].Text != "hi" || in[2].Media[0].Kind != "audio" {
		t.Fatalf("%+v", in)
	}
	if _, err := embedInputs(json.RawMessage(`[{"image": "not base64!"}]`)); err == nil {
		t.Error("bad base64 was read")
	}
}

// An embedder whose backend failed fatally is loaded again rather than handed
// out: every call to it would fail the same way until it was unloaded.
func TestABrokenEmbedderIsLoadedAgain(t *testing.T) {
	srv, _, loaded := embedServer(t)
	if code, _ := postJSON(t, srv, "/api/embed", `{"model":"nomic","input":"one"}`); code != 200 {
		t.Fatalf("first embedding: %d", code)
	}
	first := loaded()[0]
	first.mu.Lock()
	first.broken = true
	first.mu.Unlock()
	if code, _ := postJSON(t, srv, "/api/embed", `{"model":"nomic","input":"two"}`); code != 200 {
		t.Fatalf("after it broke: %d", code)
	}
	if n := len(loaded()); n != 2 {
		t.Fatalf("%d embedders loaded, want the broken one loaded again", n)
	}
	deadline := time.Now().Add(5 * time.Second)
	for !first.isClosed() && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if !first.isClosed() {
		t.Error("the broken embedder was never closed")
	}
}
