package server

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// ─── pictures ────────────────────────────────────────────────────────────────
//
// Every dialect can carry a picture, each its own way:
//
//	Ollama     "images": ["<base64>", …] beside a message's text
//	OpenAI     content parts {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}
//	Anthropic  content blocks {"type": "image", "source": {"type": "base64", "data": "…"}}
//	Responses  content parts {"type": "input_image", "image_url": "data:…"}
//
// They become chat.Message.Images, with an engine.ImageMarker in the text where
// each was — before the text, for Ollama's. A model with a projector reads
// them; one without is told in words that a picture was there.

// picture reads one picture as a request sends it: a data URL, or plain
// base64. kinfer does not fetch a picture from the web on a caller's behalf.
func picture(s string) ([]byte, error) {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://") {
		return nil, errors.New("a picture is read from the request itself (a data: URL or base64); kinfer does not fetch it from a link")
	}
	if rest, ok := strings.CutPrefix(s, "data:"); ok {
		meta, data, found := strings.Cut(rest, ",")
		if !found || !strings.HasSuffix(meta, ";base64") {
			return nil, errors.New("a data: URL picture must be base64")
		}
		s = data
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		if b, err2 := base64.RawStdEncoding.DecodeString(strings.TrimRight(s, "=")); err2 == nil {
			return b, nil
		}
		return nil, fmt.Errorf("a picture is not base64: %w", err)
	}
	return b, nil
}

// pictureFault is a picture the caller sent that the model cannot read.
func pictureFault(err error) bool {
	var bad engine.PictureError
	return errors.As(err, &bad)
}

// UnmarshalJSON reads a message in either dialect that shares this shape:
// content as a string, or as OpenAI's array of parts; and Ollama's "images".
func (m *ollamaMsg) UnmarshalJSON(b []byte) error {
	type plain ollamaMsg
	var raw struct {
		plain
		Content json.RawMessage `json:"content"`
		Images  []string        `json:"images"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	*m = ollamaMsg(raw.plain)

	var text string
	if len(raw.Content) > 0 && string(raw.Content) != "null" && json.Unmarshal(raw.Content, &text) != nil {
		var parts []struct {
			Type     string          `json:"type"`
			Text     string          `json:"text"`
			ImageURL json.RawMessage `json:"image_url"`
		}
		if err := json.Unmarshal(raw.Content, &parts); err != nil {
			return fmt.Errorf("content: want a text or a list of parts: %w", err)
		}
		var out []string
		for i, p := range parts {
			switch p.Type {
			case "text", "input_text":
				out = append(out, p.Text)
			case "image_url", "input_image":
				// {"url": "…"} in Chat Completions, the bare string elsewhere.
				var url string
				if json.Unmarshal(p.ImageURL, &url) != nil {
					var obj struct {
						URL string `json:"url"`
					}
					_ = json.Unmarshal(p.ImageURL, &obj)
					url = obj.URL
				}
				pic, err := picture(url)
				if err != nil {
					return fmt.Errorf("content[%d]: %w", i, err)
				}
				out = append(out, engine.ImageMarker)
				m.pictures = append(m.pictures, pic)
			default:
				out = append(out, omittedAttachment(p.Type))
			}
		}
		text = strings.Join(out, "\n")
	}

	// Ollama's pictures come apart from the text, and go before it.
	var lead []string
	var pics [][]byte
	for i, s := range raw.Images {
		pic, err := picture(s)
		if err != nil {
			return fmt.Errorf("images[%d]: %w", i, err)
		}
		lead = append(lead, engine.ImageMarker)
		pics = append(pics, pic)
	}
	if len(lead) > 0 {
		text = strings.Join(append(lead, text), "\n")
		m.pictures = append(pics, m.pictures...)
	}
	m.Content = text
	return nil
}
