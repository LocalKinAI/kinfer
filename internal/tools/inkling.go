package tools

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
)

// Thinking Machines' Inkling 2 declares its tools and writes its calls as
// JSON without spaces, the way its chat template's tojson calls print them:
// a declaration with every object's keys sorted, a call as
// {"name":…,"args":{…}} with the arguments' names sorted and their values as
// they came. The functions here print and read that form.

// writePyCompact is writePy with json.dumps's separators=(",", ":"), and the
// keys of every object sorted when sorted.
func writePyCompact(b *strings.Builder, v any, sorted bool) {
	switch v := v.(type) {
	case *pyObject:
		keys := v.keys
		if sorted {
			keys = append([]string(nil), keys...)
			sort.Strings(keys)
		}
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}
			writePyString(b, k)
			b.WriteByte(':')
			writePyCompact(b, v.vals[k], sorted)
		}
		b.WriteByte('}')
	case []any:
		b.WriteByte('[')
		for i, e := range v {
			if i > 0 {
				b.WriteByte(',')
			}
			writePyCompact(b, e, sorted)
		}
		b.WriteByte(']')
	default:
		writePy(b, v)
	}
}

func decodeAll(raw []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := decodePy(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing data after JSON value")
	}
	return v, nil
}

// InklingDeclare is the tools as Inkling 2's prompt declares them: a list of
// {"description", "name", "parameters", "type"}, compact, keys sorted
// throughout.
func InklingDeclare(ts []Tool) (string, error) {
	var b strings.Builder
	b.WriteByte('[')
	for i, t := range ts {
		if i > 0 {
			b.WriteByte(',')
		}
		params := any(&pyObject{vals: map[string]any{}})
		if p := bytes.TrimSpace(t.Function.Parameters); len(p) > 0 && string(p) != "null" {
			v, err := decodeAll(p)
			if err != nil {
				return "", fmt.Errorf("tool %s: %w", t.Function.Name, err)
			}
			// The template takes an empty schema for none.
			if o, ok := v.(*pyObject); !ok || len(o.keys) > 0 {
				params = v
			}
		}
		kind := t.Type
		if kind == "" {
			kind = "function"
		}
		b.WriteString(`{"description":`)
		writePyString(&b, t.Function.Description)
		b.WriteString(`,"name":`)
		writePyString(&b, t.Function.Name)
		b.WriteString(`,"parameters":`)
		writePyCompact(&b, params, true)
		b.WriteString(`,"type":`)
		writePyString(&b, kind)
		b.WriteByte('}')
	}
	b.WriteByte(']')
	return b.String(), nil
}

// InklingInvoke is a call as an Inkling 2 reply writes it and its prompt
// repeats it: {"name":…,"args":{…}}, the arguments' names sorted.
func InklingInvoke(c Call) (string, error) {
	var b strings.Builder
	b.WriteString(`{"name":`)
	writePyString(&b, c.Name)
	b.WriteString(`,"args":{`)
	args := callArguments(c.Arguments)
	v, err := decodeAll(args)
	if err != nil {
		return "", fmt.Errorf("call of %s: %w", c.Name, err)
	}
	o, ok := v.(*pyObject)
	if !ok {
		return "", fmt.Errorf("call of %s: its arguments are not an object", c.Name)
	}
	keys := append([]string(nil), o.keys...)
	sort.Strings(keys)
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		writePyString(&b, k)
		b.WriteByte(':')
		writePyCompact(&b, o.vals[k], false)
	}
	b.WriteString("}}")
	return b.String(), nil
}

// InklingCall reads a call an Inkling 2 reply wrote, {"name":…,"args":{…}},
// its arguments kept as written. header is the name the reply gave before
// the call, which must be the call's when it gave one.
func InklingCall(body, header string) (Call, error) {
	var inv struct {
		Name *string         `json:"name"`
		Args json.RawMessage `json:"args"`
	}
	if err := json.Unmarshal([]byte(body), &inv); err != nil {
		return Call{}, fmt.Errorf("a tool call that is not JSON: %w", err)
	}
	if inv.Name == nil || *inv.Name == "" {
		return Call{}, fmt.Errorf("a tool call without a name")
	}
	if h := strings.TrimSpace(header); h != "" && h != *inv.Name {
		return Call{}, fmt.Errorf("a tool call of %q under the name %q", *inv.Name, h)
	}
	args := bytes.TrimSpace(inv.Args)
	if len(args) == 0 || string(args) == "null" {
		args = []byte("{}")
	}
	return Call{Name: *inv.Name, Arguments: json.RawMessage(args)}, nil
}
