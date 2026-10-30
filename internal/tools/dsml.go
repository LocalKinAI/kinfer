package tools

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// DeepSeek V4.1 calls functions in DSML, a markup of its own, written in one
// block after the reply's prose:
//
//	<｜DSML｜ calls>
//	<｜DSML｜ invoke name="get_weather">
//	<｜DSML｜ parameter name="city" string="true">Berlin</｜DSML｜ parameter>
//	<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
//	</｜DSML｜ invoke>
//	</｜DSML｜ calls>
//
// ｜DSML｜ is a single token. A string argument is written as it is, with no
// quoting; anything else as JSON, flagged string="false".
//
// V4 spelled the tags without the space — <｜DSML｜tool_calls>,
// <｜DSML｜invoke> — and V4.1 is prompted with the spaced form only. DeepSeek's
// reference parser refuses the old spelling; its server parser accepts both,
// and so does Parse here: a call the model did make is worth more than a
// lesson in spelling.
//
// Every string below is DeepSeek's own, from the encoder that ships with the
// model (encoding/encoding.py in deepseek-ai/DeepSeek-V4.1-Flash-1017). The
// model was trained on that text, not on a paraphrase of it.

const (
	dsmlToken = "｜DSML｜"

	// dsmlOpen is how every DSML tag begins, and so what a stream holds back on.
	dsmlOpen = "<" + dsmlToken
)

// dsmlToolPrompt is V4.1's "## Tools" section, which DeepSeek's encoder adds to
// the first system message. Its schemas are each function's name, description
// and parameters, one per line in json.dumps form; a function without
// parameters gets {}, as DeepSeek's server gives it.
func dsmlToolPrompt(ts []Tool) string {
	var schemas []string
	for _, t := range ts {
		params := t.Function.Parameters
		if len(bytes.TrimSpace(params)) == 0 {
			params = json.RawMessage("{}")
		}
		name, _ := json.Marshal(t.Function.Name)
		desc, _ := json.Marshal(t.Function.Description)
		obj := fmt.Sprintf(`{"name":%s,"description":%s,"parameters":%s}`, name, desc, params)
		s, err := PythonJSON([]byte(obj))
		if err != nil {
			continue // a tool that will not print cannot be offered
		}
		schemas = append(schemas, s)
	}

	const d = dsmlToken
	return `## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<` + d + ` calls>" block like the following:

<` + d + ` calls>
<` + d + ` invoke name="$TOOL_NAME">
<` + d + ` parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</` + d + ` parameter>
...
</` + d + ` invoke>
<` + d + ` invoke name="$TOOL_NAME2">
...
</` + d + ` invoke>
</` + d + ` calls>

String parameters should be specified as is and set ` + "`string=\"true\"`" + `. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set ` + "`string=\"false\"`" + `.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

` + strings.Join(schemas, "\n") + `

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
`
}

// renderDSMLCalls writes an assistant turn's calls back as the block the model
// produced them in, so a later turn sees its own output verbatim.
func renderDSMLCalls(calls []Call) string {
	var b strings.Builder
	b.WriteString("\n\n<" + dsmlToken + " calls>\n")
	for i, c := range calls {
		if i > 0 {
			b.WriteString("\n")
		}
		fmt.Fprintf(&b, "<%s invoke name=\"%s\">\n", dsmlToken, c.Name)
		for j, a := range dsmlArguments(c.Arguments) {
			if j > 0 {
				b.WriteString("\n")
			}
			fmt.Fprintf(&b, "<%s parameter name=\"%s\" string=\"%t\">%s</%s parameter>",
				dsmlToken, a.key, a.isString, a.value, dsmlToken)
		}
		b.WriteString("\n</" + dsmlToken + " invoke>")
	}
	b.WriteString("\n</" + dsmlToken + " calls>")
	return b.String()
}

type dsmlArg struct {
	key, value string
	isString   bool
}

// dsmlArguments splits a call's arguments into DSML parameters, in their
// order. Arguments arrive as an object (Ollama) or as a string holding one
// (OpenAI); like DeepSeek's encoder, up to two layers of string are taken off
// before giving up, and arguments that are no object at all become a single
// parameter named "arguments".
func dsmlArguments(raw json.RawMessage) []dsmlArg {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil
	}
	v, err := decodePyBytes(raw)
	if err != nil {
		return []dsmlArg{{key: "arguments", value: string(raw), isString: true}}
	}
	orig := v
	for range 2 {
		s, ok := v.(string)
		if !ok {
			break
		}
		inner, err := decodePyBytes([]byte(s))
		if err != nil {
			break
		}
		v = inner
	}
	obj, ok := v.(*pyObject)
	if !ok {
		obj = &pyObject{keys: []string{"arguments"}, vals: map[string]any{"arguments": orig}}
	}
	args := make([]dsmlArg, 0, len(obj.keys))
	for _, k := range obj.keys {
		if s, ok := obj.vals[k].(string); ok {
			args = append(args, dsmlArg{key: k, value: s, isString: true})
			continue
		}
		var b strings.Builder
		writePy(&b, obj.vals[k])
		args = append(args, dsmlArg{key: k, value: b.String()})
	}
	return args
}

func decodePyBytes(raw []byte) (any, error) {
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

// parseDSML splits a V4.1 reply into its prose and the calls in its DSML
// blocks. As with the other formats, a block that never closes, or an invoke
// that cannot be read, stays in the content: a reply cut off mid-call is
// something the caller should see.
func parseDSML(reply string) (content string, calls []Call) {
	var text strings.Builder
	rest := reply
	for {
		start, bodyAt, ok := findDSMLTag(rest, false, "tool_calls>", "calls>")
		if !ok {
			text.WriteString(rest)
			break
		}
		end, after, ok := findDSMLTag(rest[bodyAt:], true, "tool_calls>", "calls>")
		if !ok {
			text.WriteString(rest) // unterminated
			break
		}
		text.WriteString(rest[:start])
		body := rest[bodyAt : bodyAt+end]
		cs, unread := parseDSMLInvokes(body)
		calls = append(calls, cs...)
		if unread != "" {
			text.WriteString(rest[start : bodyAt+after])
		}
		rest = rest[bodyAt+after:]
	}
	return strings.TrimSpace(text.String()), calls
}

// parseDSMLInvokes reads every invoke in a block. unread is non-empty when
// something in the block could not be read as a call.
func parseDSMLInvokes(body string) (calls []Call, unread string) {
	rest := body
	for {
		i, at, ok := findDSMLTag(rest, false, "invoke")
		if !ok {
			if strings.TrimSpace(rest) != "" {
				unread = rest
			}
			return calls, unread
		}
		j, after, ok := findDSMLTag(rest[at:], true, "invoke>")
		if !ok {
			return calls, rest[i:]
		}
		c, ok := parseDSMLInvoke(rest[at : at+j])
		if ok {
			calls = append(calls, c)
		} else {
			unread = rest[i : at+after]
		}
		rest = rest[at+after:]
	}
}

// parseDSMLInvoke reads ` name="f">` and the parameters that follow it, and
// builds the arguments as DeepSeek's encoder does: a string parameter quoted
// as JSON, any other inserted as written. A value flagged as JSON that is not
// valid JSON is kept as a string instead, since a caller cannot decode broken
// arguments at all.
func parseDSMLInvoke(s string) (Call, bool) {
	name, rest, ok := attr(s, "name")
	if !ok || name == "" {
		return Call{}, false
	}
	gt := strings.IndexByte(rest, '>')
	if gt < 0 {
		return Call{}, false
	}
	rest = rest[gt+1:]

	var parts []string
	seen := map[string]bool{}
	for {
		_, at, ok := findDSMLTag(rest, false, "parameter")
		if !ok {
			break
		}
		key, r, ok := attr(rest[at:], "name")
		if !ok {
			return Call{}, false
		}
		flag, r, ok := attr(r, "string")
		if !ok || !strings.HasPrefix(r, ">") {
			return Call{}, false
		}
		r = r[1:]
		end, after, ok := findDSMLTag(r, true, "parameter>")
		if !ok {
			return Call{}, false
		}
		value := r[:end]
		rest = r[after:]
		if seen[key] {
			continue
		}
		seen[key] = true

		k, _ := json.Marshal(key)
		if flag != "true" && json.Valid([]byte(value)) {
			parts = append(parts, fmt.Sprintf("%s: %s", k, strings.TrimSpace(value)))
			continue
		}
		var v strings.Builder
		writePyString(&v, value)
		parts = append(parts, fmt.Sprintf("%s: %s", k, v.String()))
	}
	return Call{Name: name, Arguments: json.RawMessage("{" + strings.Join(parts, ", ") + "}")}, true
}

// attr reads ` key="value"` at the start of s, allowing spaces before it.
func attr(s, key string) (value, rest string, ok bool) {
	s = strings.TrimLeft(s, " \t\n")
	if !strings.HasPrefix(s, key+`="`) {
		return "", s, false
	}
	s = s[len(key)+2:]
	q := strings.IndexByte(s, '"')
	if q < 0 {
		return "", s, false
	}
	return s[:q], s[q+1:], true
}

// findDSMLTag finds the first DSML tag in s whose name (after the token and an
// optional space) starts with one of names, opening or closing. start is where
// the tag begins and after where its name ends.
func findDSMLTag(s string, closing bool, names ...string) (start, after int, ok bool) {
	open := dsmlOpen
	if closing {
		open = "</" + dsmlToken
	}
	from := 0
	for {
		i := strings.Index(s[from:], open)
		if i < 0 {
			return 0, 0, false
		}
		i += from
		j := i + len(open)
		if j < len(s) && s[j] == ' ' {
			j++
		}
		for _, n := range names {
			if strings.HasPrefix(s[j:], n) {
				return i, j + len(n), true
			}
		}
		from = i + len(open)
	}
}
