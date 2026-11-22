package server

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/LocalKinAI/kinfer/internal/tools"
)

// A reply held to JSON, as each dialect asks for one. Each of these reads its
// dialect's field into engine.GenParams.Format — nothing, the string "json",
// or a JSON Schema — and refuses a schema a reply cannot be held to, with what
// is wrong with it, where the reply would otherwise come back unchecked and
// the caller would not know.

// checkedFormat is format once tools.JSONGrammar has taken it.
func checkedFormat(format json.RawMessage, field string) (json.RawMessage, error) {
	if _, err := tools.JSONGrammar(format); err != nil {
		return nil, fmt.Errorf("%s: %w", field, err)
	}
	return format, nil
}

// ollamaFormat reads /api/chat's format: "json", or a JSON Schema.
func ollamaFormat(raw json.RawMessage) (json.RawMessage, error) {
	raw = bytes.TrimSpace(raw)
	switch string(raw) {
	case "", "null", `""`:
		return nil, nil
	case `"json"`:
		return raw, nil
	}
	if raw[0] != '{' {
		return nil, fmt.Errorf(`format: %s is neither "json" nor a JSON Schema`, raw)
	}
	return checkedFormat(raw, "format")
}

// openAIFormat reads /v1/chat/completions' response_format:
// {"type":"text"}, {"type":"json_object"}, or
// {"type":"json_schema","json_schema":{"name":…,"schema":{…}}}.
func openAIFormat(raw json.RawMessage) (json.RawMessage, error) {
	if len(bytes.TrimSpace(raw)) == 0 || string(bytes.TrimSpace(raw)) == "null" {
		return nil, nil
	}
	var f struct {
		Type       string `json:"type"`
		JSONSchema *struct {
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("response_format: %w", err)
	}
	return typedFormat(f.Type, func() json.RawMessage {
		if f.JSONSchema == nil {
			return nil
		}
		return f.JSONSchema.Schema
	}, "response_format")
}

// responsesFormat reads /v1/responses' text.format: {"type":"text"},
// {"type":"json_object"}, or {"type":"json_schema","name":…,"schema":{…}}.
func responsesFormat(text json.RawMessage) (json.RawMessage, error) {
	if len(bytes.TrimSpace(text)) == 0 || string(bytes.TrimSpace(text)) == "null" {
		return nil, nil
	}
	var t struct {
		Format *struct {
			Type   string          `json:"type"`
			Schema json.RawMessage `json:"schema"`
		} `json:"format"`
	}
	if err := json.Unmarshal(text, &t); err != nil {
		return nil, fmt.Errorf("text: %w", err)
	}
	if t.Format == nil {
		return nil, nil
	}
	return typedFormat(t.Format.Type, func() json.RawMessage { return t.Format.Schema }, "text.format")
}

func typedFormat(typ string, schema func() json.RawMessage, field string) (json.RawMessage, error) {
	switch typ {
	case "", "text":
		return nil, nil
	case "json_object":
		return json.RawMessage(`"json"`), nil
	case "json_schema":
		s := bytes.TrimSpace(schema())
		if len(s) == 0 || s[0] != '{' {
			return nil, fmt.Errorf("%s: a json_schema without its schema", field)
		}
		return checkedFormat(s, field)
	}
	return nil, fmt.Errorf("%s: type %q is none of text, json_object and json_schema", field, typ)
}
