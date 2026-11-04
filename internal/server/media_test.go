package server

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
)

var (
	pic    = []byte("\x89PNG not really, but bytes")
	pic64  = base64.StdEncoding.EncodeToString(pic)
	marker = engine.ImageMarker
)

func TestMessagesCarryPictures(t *testing.T) {
	cases := []struct {
		name, raw, content string
		pictures           int
	}{
		{"a plain message", `{"role":"user","content":"hi"}`, "hi", 0},
		{"Ollama's images go before the text", `{"role":"user","content":"look","images":["` + pic64 + `"]}`, marker + "\nlook", 1},
		{"OpenAI's parts, in place",
			`{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + pic64 + `"}},{"type":"text","text":"b"}]}`,
			"a\n" + marker + "\nb", 1},
		{"an image_url as a bare string", `{"role":"user","content":[{"type":"image_url","image_url":"data:image/png;base64,` + pic64 + `"}]}`, marker, 1},
		{"text parts only", `{"role":"user","content":[{"type":"text","text":"x"},{"type":"text","text":"y"}]}`, "x\ny", 0},
		{"no content", `{"role":"assistant","content":null}`, "", 0},
	}
	for _, c := range cases {
		var m ollamaMsg
		if err := json.Unmarshal([]byte(c.raw), &m); err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if m.Content != c.content || len(m.pictures) != c.pictures {
			t.Errorf("%s: content %q with %d pictures, want %q with %d", c.name, m.Content, len(m.pictures), c.content, c.pictures)
		}
		if c.pictures > 0 && string(m.pictures[0]) != string(pic) {
			t.Errorf("%s: the picture decoded to %q", c.name, m.pictures[0])
		}
	}

	for name, c := range map[string]struct{ raw, says string }{
		"a link":             {`{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}`, "does not fetch"},
		"not base64":         {`{"role":"user","content":"x","images":["%%%"]}`, "not base64"},
		"a data URL in text": {`{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png,abc"}}]}`, "must be base64"},
	} {
		var m ollamaMsg
		if err := json.Unmarshal([]byte(c.raw), &m); err == nil || !strings.Contains(err.Error(), c.says) {
			t.Errorf("%s: read as %q with %d pictures (%v); want an error that says %q", name, m.Content, len(m.pictures), err, c.says)
		}
	}
}

// The picture reaches the model with the marker where it goes, in every
// dialect that carries one.
func TestEachDialectSendsThePicture(t *testing.T) {
	srv, reg := newTestServer(t, "alpha")
	defer srv.Close()
	for _, c := range []struct{ path, body string }{
		{"/api/chat", `{"model":"alpha","stream":false,"messages":[{"role":"user","content":"what is it?","images":["` + pic64 + `"]}]}`},
		{"/v1/chat/completions", `{"model":"alpha","messages":[{"role":"user","content":[{"type":"text","text":"what is it?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + pic64 + `"}}]}]}`},
		{"/v1/messages", `{"model":"alpha","max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"what is it?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + pic64 + `"}}]}]}`},
		{"/v1/responses", `{"model":"alpha","input":[{"role":"user","content":[{"type":"input_text","text":"what is it?"},{"type":"input_image","image_url":"data:image/png;base64,` + pic64 + `"}]}]}`},
	} {
		if code, out := postJSON(t, srv, c.path, c.body); code != http.StatusOK {
			t.Errorf("%s: %d %v", c.path, code, out)
			continue
		}
		f := reg.get("alpha")
		f.mu.Lock()
		msgs := f.msgs
		f.mu.Unlock()
		var got *chat.Message
		for i := range msgs {
			if len(msgs[i].Images) > 0 {
				got = &msgs[i]
			}
		}
		if got == nil || len(got.Images) != 1 || string(got.Images[0]) != string(pic) || strings.Count(got.Content, marker) != 1 {
			t.Errorf("%s: the model was sent %+v", c.path, msgs)
		}
	}

	if code, _ := postJSON(t, srv, "/v1/messages", `{"model":"alpha","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}]}`); code != http.StatusBadRequest {
		t.Errorf("an Anthropic image by link: %d, want 400", code)
	}
}

// A picture the model cannot read is the caller's to fix: 400, not 500.
func TestUnreadablePictureIsTheCallers(t *testing.T) {
	srv, reg := newTestServer(t, "alpha")
	defer srv.Close()
	postJSON(t, srv, "/api/chat", `{"model":"alpha","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	f := reg.get("alpha")
	f.mu.Lock()
	f.err = engine.PictureError{Err: errors.New("picture 0 is not an image llama.cpp reads")}
	f.mu.Unlock()
	if code, out := postJSON(t, srv, "/api/chat", `{"model":"alpha","stream":false,"messages":[{"role":"user","content":"x","images":["`+pic64+`"]}]}`); code != http.StatusBadRequest {
		t.Errorf("an unreadable picture: %d %v, want 400", code, out)
	}
	if code, _ := postJSON(t, srv, "/v1/chat/completions", `{"model":"alpha","messages":[{"role":"user","content":"x"}]}`); code != http.StatusBadRequest {
		t.Errorf("OpenAI dialect: %d, want 400", code)
	}
}
