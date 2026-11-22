package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Each dialect's way of asking for JSON reaches the model as one format, and
// a schema a reply cannot be held to is refused with what is wrong with it.
func TestReplyFormatInEveryDialect(t *testing.T) {
	const schema = `{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`
	for _, c := range []struct{ path, field, want string }{
		{"/api/chat", `"format":"json"`, `"json"`},
		{"/api/chat", `"format":` + schema, schema},
		{"/api/chat", `"format":""`, ``},
		{"/v1/chat/completions", `"response_format":{"type":"json_object"}`, `"json"`},
		{"/v1/chat/completions", `"response_format":{"type":"json_schema","json_schema":{"name":"place","strict":true,"schema":` + schema + `}}`, schema},
		{"/v1/chat/completions", `"response_format":{"type":"text"}`, ``},
		{"/v1/responses", `"text":{"format":{"type":"json_schema","name":"place","schema":` + schema + `}}`, schema},
		{"/v1/responses", `"text":{"format":{"type":"json_object"}}`, `"json"`},
		{"/v1/responses", `"text":{"verbosity":"low"}`, ``},
		{"/v1/messages", `"max_tokens":50,"output_config":{"format":{"type":"json_schema","schema":` + schema + `}}`, schema},
	} {
		srv, reg := newTestServer(t, "m")
		body := `{"model":"m","stream":false,"messages":[{"role":"user","content":"where"}],` + c.field + `}`
		if c.path == "/v1/responses" {
			body = `{"model":"m","input":"where",` + c.field + `}`
		}
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", c.path, strings.NewReader(body)))
		if w.Code != http.StatusOK {
			t.Errorf("%s %s: %d %s", c.path, c.field, w.Code, w.Body.String())
			srv.Close()
			continue
		}
		f := reg.get("m")
		f.mu.Lock()
		got := string(f.params.Format)
		f.mu.Unlock()
		if got != c.want {
			t.Errorf("%s %s: the model was given format %q, want %q", c.path, c.field, got, c.want)
		}
		srv.Close()
	}

	srv, _ := newTestServer(t, "m")
	defer srv.Close()
	for _, c := range []struct{ path, field, says string }{
		{"/api/chat", `"format":"yaml"`, "neither"},
		{"/api/chat", `"format":{"type":"object","properties":{"a":{"$ref":"#/$defs/none"}}}`, "nothing there"},
		{"/v1/chat/completions", `"response_format":{"type":"json_schema"}`, "without its schema"},
		{"/v1/chat/completions", `"response_format":{"type":"xml"}`, "none of"},
		{"/v1/responses", `"text":{"format":{"type":"json_schema","schema":{"type":"tuple"}}}`, "tuple"},
	} {
		body := `{"model":"m","stream":false,"messages":[{"role":"user","content":"where"}],` + c.field + `}`
		if c.path == "/v1/responses" {
			body = `{"model":"m","input":"where",` + c.field + `}`
		}
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, httptest.NewRequest("POST", c.path, strings.NewReader(body)))
		var out map[string]any
		json.Unmarshal(w.Body.Bytes(), &out)
		if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), c.says) {
			t.Errorf("%s %s: %d %s, want a 400 saying %q", c.path, c.field, w.Code, w.Body.String(), c.says)
		}
	}
}
