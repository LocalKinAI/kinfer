package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A thinking level reaches the engine as an effort, in either dialect; OpenAI's
// "none" is thinking off.
func TestReasoningEffortReachesTheEngine(t *testing.T) {
	const msgs = `"messages":[{"role":"user","content":"hi"}]`
	for _, c := range []struct {
		path, body string
		effort     string
		noThink    bool
	}{
		{"/api/chat", `{"model":"alpha","stream":false,"think":"low",` + msgs + `}`, "low", false},
		{"/api/chat", `{"model":"alpha","stream":false,"think":true,` + msgs + `}`, "", false},
		{"/api/chat", `{"model":"alpha","stream":false,"think":false,` + msgs + `}`, "", true},
		{"/v1/chat/completions", `{"model":"alpha","reasoning_effort":"max",` + msgs + `}`, "max", false},
		{"/v1/chat/completions", `{"model":"alpha","reasoning_effort":"none",` + msgs + `}`, "", true},
		{"/v1/chat/completions", `{"model":"alpha",` + msgs + `}`, "", false},
	} {
		srv, reg := newTestServer(t, "alpha")
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", c.path, strings.NewReader(c.body)))
		if w.Code != http.StatusOK {
			t.Fatalf("%s %s: %d %s", c.path, c.body, w.Code, w.Body)
		}
		p := reg.get("alpha").lastParams()
		if p.Effort != c.effort || p.NoThink != c.noThink {
			t.Errorf("%s %s: effort %q, noThink %v; want %q, %v", c.path, c.body, p.Effort, p.NoThink, c.effort, c.noThink)
		}
	}
}

// An earlier turn's thinking comes back in "thinking" (Ollama) or, from
// clients built against DeepSeek's API, "reasoning_content".
func TestEarlierThinkingIsKept(t *testing.T) {
	var in []ollamaMsg
	if err := json.Unmarshal([]byte(`[
		{"role":"assistant","content":"a","thinking":"ollama"},
		{"role":"assistant","content":"b","reasoning_content":"deepseek"},
		{"role":"assistant","content":"c","thinking":"both","reasoning_content":"ignored"}]`), &in); err != nil {
		t.Fatal(err)
	}
	got := toChatMessages(in)
	for i, want := range []string{"ollama", "deepseek", "both"} {
		if got[i].Thinking != want {
			t.Errorf("message %d: thinking %q, want %q", i, got[i].Thinking, want)
		}
	}
}
