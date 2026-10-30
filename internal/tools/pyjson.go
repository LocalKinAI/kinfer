package tools

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// PythonJSON re-prints JSON the way Python's json.dumps(v, ensure_ascii=False)
// prints it: ", " and ": " between items, keys in the order they arrived,
// non-ASCII left as it is, and floats as Python's repr writes them.
//
// DeepSeek's prompt encoder is Python, and a model reads its tool schemas and
// non-string tool arguments in exactly that form. Go's encoder differs in every
// one of those respects — compact separators, sorted map keys, <, > and &
// escaped, 1e-05 written as 1e-5 — and a schema the model never saw spelled
// that way is a schema it reads less reliably.
func PythonJSON(raw []byte) (string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := decodePy(dec)
	if err != nil {
		return "", err
	}
	if _, err := dec.Token(); err != io.EOF {
		return "", fmt.Errorf("trailing data after JSON value")
	}
	var b strings.Builder
	writePy(&b, v)
	return b.String(), nil
}

// pyObject keeps a JSON object's keys in the order they were written, as a
// Python dict does. A repeated key keeps its first position and its last value,
// which is also what json.loads does.
type pyObject struct {
	keys []string
	vals map[string]any
}

func decodePy(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	d, ok := tok.(json.Delim)
	if !ok {
		return tok, nil
	}
	switch d {
	case '{':
		obj := &pyObject{vals: map[string]any{}}
		for dec.More() {
			kt, err := dec.Token()
			if err != nil {
				return nil, err
			}
			k, ok := kt.(string)
			if !ok {
				return nil, fmt.Errorf("object key is %T, not a string", kt)
			}
			v, err := decodePy(dec)
			if err != nil {
				return nil, err
			}
			if _, seen := obj.vals[k]; !seen {
				obj.keys = append(obj.keys, k)
			}
			obj.vals[k] = v
		}
		_, err := dec.Token() // }
		return obj, err
	case '[':
		arr := []any{}
		for dec.More() {
			v, err := decodePy(dec)
			if err != nil {
				return nil, err
			}
			arr = append(arr, v)
		}
		_, err := dec.Token() // ]
		return arr, err
	}
	return nil, fmt.Errorf("unexpected %v", d)
}

func writePy(b *strings.Builder, v any) {
	switch v := v.(type) {
	case *pyObject:
		b.WriteByte('{')
		for i, k := range v.keys {
			if i > 0 {
				b.WriteString(", ")
			}
			writePyString(b, k)
			b.WriteString(": ")
			writePy(b, v.vals[k])
		}
		b.WriteByte('}')
	case []any:
		b.WriteByte('[')
		for i, e := range v {
			if i > 0 {
				b.WriteString(", ")
			}
			writePy(b, e)
		}
		b.WriteByte(']')
	case string:
		writePyString(b, v)
	case json.Number:
		b.WriteString(pyNumber(string(v)))
	case bool:
		if v {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case nil:
		b.WriteString("null")
	}
}

// writePyString quotes s as json.dumps does with ensure_ascii=False: only the
// quote, the backslash and control characters are escaped, the control
// characters without a short form as \u00xx in lowercase hex.
func writePyString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// pyNumber writes a JSON number the way Python reads and then re-prints it: an
// integer as an integer, anything with a fraction or an exponent as a float.
func pyNumber(text string) string {
	if !strings.ContainsAny(text, ".eE") {
		if text == "-0" {
			return "0" // an int has no negative zero
		}
		return text
	}
	f, err := strconv.ParseFloat(text, 64)
	if err != nil && !errors.Is(err, strconv.ErrRange) {
		return text
	}
	// Out of range, ParseFloat gives what Python's float() gives: ±inf or 0.
	return pyFloat(f)
}

// pyFloat is Python's float repr: the shortest digits that read back exactly,
// in fixed notation from 1e-4 up to 1e16 and with an exponent of at least two
// digits outside it.
func pyFloat(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	case f == 0:
		if math.Signbit(f) {
			return "-0.0"
		}
		return "0.0"
	}
	if a := math.Abs(f); a >= 1e-4 && a < 1e16 {
		s := strconv.FormatFloat(f, 'f', -1, 64)
		if !strings.Contains(s, ".") {
			s += ".0"
		}
		return s
	}
	// Go's 'e' form already writes a signed, two-digit exponent: 1e-05, 1e+16.
	return strconv.FormatFloat(f, 'e', -1, 64)
}
