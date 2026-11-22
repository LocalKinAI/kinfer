package tools

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// JSONGrammar is a GBNF grammar for a reply that is one JSON value and nothing
// else: any JSON object when schema is empty or the string "json" (Ollama's
// format and OpenAI's json_object), a value matching the JSON Schema
// otherwise.
//
// What a grammar can hold a reply to is its shape: the types, an object's
// properties and which are required, in the order the schema declares them, an
// array's items and how many, enums and constants, anyOf and oneOf, and $ref
// into the schema's own $defs. What it cannot, it leaves to the model: a
// string's pattern, format or length, a number's range, not, if and then. A
// reply is still JSON of the right shape then, and the caller checks the rest
// as it would have.
//
// An object takes the properties it declares and no others, unless it says
// additionalProperties, as OpenAI's strict mode has it; one that declares
// none is any JSON object.
func JSONGrammar(schema json.RawMessage) (string, error) {
	g := &schemaGrammar{refs: map[string]string{}}
	root := "jobj"
	if s := bytes.TrimSpace(schema); len(s) > 0 && string(s) != `"json"` && string(s) != "null" {
		g.root = s
		var err error
		if root, err = g.visit(s, 0); err != nil {
			return "", err
		}
	}
	var b strings.Builder
	b.WriteString("root ::= jws " + root + " jws\n")
	b.WriteString(g.rules.String())
	b.WriteString(`jint ::= "-"? ( "0" | [1-9] [0-9]{0,18} )` + "\n")
	b.WriteString(jsonRules)
	return b.String(), nil
}

// The most rules a schema may come to, and how deep it may nest: a schema is
// the caller's, and a grammar is compiled before the reply's first token.
const (
	maxSchemaRules = 2000
	maxSchemaDepth = 64
)

type schemaGrammar struct {
	rules strings.Builder
	n     int
	root  json.RawMessage
	refs  map[string]string // a $ref's rule, made once: a schema may refer to itself
}

func (g *schemaGrammar) define(body string) (string, error) {
	name := fmt.Sprintf("s%d", g.n)
	return name, g.defineAs(name, body)
}

func (g *schemaGrammar) defineAs(name, body string) error {
	if g.n++; g.n > maxSchemaRules {
		return errors.New("the schema is too large to hold a reply to")
	}
	fmt.Fprintf(&g.rules, "%s ::= %s\n", name, body)
	return nil
}

func (g *schemaGrammar) newName() string {
	g.n++
	return fmt.Sprintf("s%d", g.n-1)
}

type schemaNode struct {
	Ref        string            `json:"$ref"`
	Type       json.RawMessage   `json:"type"`
	Enum       []json.RawMessage `json:"enum"`
	Const      json.RawMessage   `json:"const"`
	AnyOf      []json.RawMessage `json:"anyOf"`
	OneOf      []json.RawMessage `json:"oneOf"`
	AllOf      []json.RawMessage `json:"allOf"`
	Properties json.RawMessage   `json:"properties"`
	Required   []string          `json:"required"`
	Additional json.RawMessage   `json:"additionalProperties"`
	Items      json.RawMessage   `json:"items"`
	Prefix     []json.RawMessage `json:"prefixItems"`
	MinItems   *int              `json:"minItems"`
	MaxItems   *int              `json:"maxItems"`
	Nullable   bool              `json:"nullable"`
}

// visit is the expression a value of schema has to match.
func (g *schemaGrammar) visit(schema json.RawMessage, depth int) (string, error) {
	if depth > maxSchemaDepth {
		return "", errors.New("the schema nests too deep to hold a reply to")
	}
	s := bytes.TrimSpace(schema)
	if len(s) == 0 || string(s) == "true" || string(s) == "{}" {
		return "jv", nil
	}
	if string(s) == "false" {
		return "", errors.New("a schema of false matches nothing")
	}
	var n schemaNode
	if err := json.Unmarshal(s, &n); err != nil {
		return "", fmt.Errorf("the schema is not a JSON Schema: %w", err)
	}
	expr, err := g.node(&n, depth)
	if err != nil {
		return "", err
	}
	if n.Nullable && expr != "jv" {
		expr = `( ` + expr + ` | "null" )`
	}
	return expr, nil
}

func (g *schemaGrammar) node(n *schemaNode, depth int) (string, error) {
	switch {
	case n.Ref != "":
		return g.ref(n.Ref, depth)
	case len(n.Const) > 0:
		return jsonLiteral(n.Const)
	case len(n.Enum) > 0:
		alts := make([]string, 0, len(n.Enum))
		for _, v := range n.Enum {
			l, err := jsonLiteral(v)
			if err != nil {
				return "", err
			}
			alts = append(alts, l)
		}
		return "( " + strings.Join(alts, " | ") + " )", nil
	case len(n.AnyOf) > 0 || len(n.OneOf) > 0:
		return g.oneOf(append(append([]json.RawMessage{}, n.AnyOf...), n.OneOf...), depth)
	case len(n.AllOf) == 1:
		return g.visit(n.AllOf[0], depth+1)
	case len(n.AllOf) > 1:
		return "", errors.New("allOf of several schemas cannot be held to by a grammar")
	}

	var typ string
	if len(n.Type) > 0 && json.Unmarshal(n.Type, &typ) != nil {
		var types []string
		if err := json.Unmarshal(n.Type, &types); err != nil || len(types) == 0 {
			return "", fmt.Errorf("a schema's type is %s", n.Type)
		}
		alts := make([]string, 0, len(types))
		for _, t := range types {
			e, err := g.typed(t, n, depth)
			if err != nil {
				return "", err
			}
			alts = append(alts, e)
		}
		return "( " + strings.Join(alts, " | ") + " )", nil
	}
	if typ == "" {
		switch {
		case len(n.Properties) > 0 || len(n.Additional) > 0:
			typ = "object"
		case len(n.Items) > 0 || len(n.Prefix) > 0:
			typ = "array"
		default:
			return "jv", nil
		}
	}
	return g.typed(typ, n, depth)
}

func (g *schemaGrammar) typed(typ string, n *schemaNode, depth int) (string, error) {
	switch typ {
	case "string":
		return "jstr", nil
	case "integer":
		return "jint", nil
	case "number":
		return "jnum", nil
	case "boolean":
		return `( "true" | "false" )`, nil
	case "null":
		return `"null"`, nil
	case "array":
		return g.array(n, depth)
	case "object":
		return g.object(n, depth)
	}
	return "", fmt.Errorf("a schema's type is %q", typ)
}

func (g *schemaGrammar) oneOf(alts []json.RawMessage, depth int) (string, error) {
	exprs := make([]string, 0, len(alts))
	for _, a := range alts {
		e, err := g.visit(a, depth+1)
		if err != nil {
			return "", err
		}
		exprs = append(exprs, e)
	}
	return "( " + strings.Join(exprs, " | ") + " )", nil
}

// ref is the rule of a schema the root schema holds under #/$defs or
// #/definitions (or the root itself, "#").
func (g *schemaGrammar) ref(ref string, depth int) (string, error) {
	if name, ok := g.refs[ref]; ok {
		return name, nil
	}
	target := g.root
	if ref != "#" {
		if !strings.HasPrefix(ref, "#/") {
			return "", fmt.Errorf("$ref %q: only the schema's own definitions can be referred to", ref)
		}
		for _, part := range strings.Split(ref[2:], "/") {
			part = strings.NewReplacer("~1", "/", "~0", "~").Replace(part)
			var m map[string]json.RawMessage
			if json.Unmarshal(target, &m) != nil || m[part] == nil {
				return "", fmt.Errorf("$ref %q: the schema has nothing there", ref)
			}
			target = m[part]
		}
	}
	name := g.newName()
	g.refs[ref] = name // before it is read: it may refer to itself
	body, err := g.visit(target, depth+1)
	if err != nil {
		return "", err
	}
	return name, g.defineAs(name, body)
}

func (g *schemaGrammar) array(n *schemaNode, depth int) (string, error) {
	if len(n.Prefix) > 0 {
		parts := make([]string, 0, len(n.Prefix))
		for _, p := range n.Prefix {
			e, err := g.visit(p, depth+1)
			if err != nil {
				return "", err
			}
			parts = append(parts, e)
		}
		return g.define(`"[" jws ` + strings.Join(parts, ` jws "," jws `) + ` jws "]"`)
	}
	item := "jv"
	if len(n.Items) > 0 {
		var err error
		if item, err = g.visit(n.Items, depth+1); err != nil {
			return "", err
		}
	}
	lo, hi := 0, -1
	if n.MinItems != nil && *n.MinItems > 0 {
		lo = *n.MinItems
	}
	if n.MaxItems != nil && *n.MaxItems >= 0 {
		hi = *n.MaxItems
	}
	if hi >= 0 && hi < lo {
		return "", fmt.Errorf("an array of at least %d and at most %d items", lo, hi)
	}
	if lo > 1000 || hi > 1000 {
		return "", errors.New("an array's minItems or maxItems is past 1,000")
	}
	more := func(lo, hi int) string { // how many ", item" follow the first
		switch {
		case hi < 0 && lo == 0:
			return "*"
		case hi < 0:
			return fmt.Sprintf("{%d,}", lo)
		default:
			return fmt.Sprintf("{%d,%d}", lo, hi)
		}
	}
	rest := `( "," jws ` + item + ` jws )`
	switch {
	case hi == 0:
		return `"[" jws "]"`, nil
	case lo == 0:
		return g.define(`"[" jws ( ` + item + ` jws ` + rest + more(0, hi-1) + ` )? "]"`)
	default:
		return g.define(`"[" jws ` + item + ` jws ` + rest + more(lo-1, hi-1) + ` "]"`)
	}
}

// object is an object's rule: its declared properties in their declared
// order, the required ones always and the others or not, then, where it
// allows them, properties it did not declare.
func (g *schemaGrammar) object(n *schemaNode, depth int) (string, error) {
	props, err := orderedProperties(n.Properties)
	if err != nil {
		return "", err
	}
	extra := "" // the value of a property not declared, "" for none allowed
	switch a := string(bytes.TrimSpace(n.Additional)); {
	case a == "" && len(props) == 0, a == "true":
		extra = "jv"
	case a == "", a == "false":
	default:
		if extra, err = g.visit(n.Additional, depth+1); err != nil {
			return "", err
		}
	}
	if len(props) == 0 {
		if extra == "jv" {
			return "jobj", nil
		}
		if extra == "" {
			return `"{" jws "}"`, nil
		}
		return g.define(`"{" jws ( jstr jws ":" jws ` + extra + ` jws ( "," jws jstr jws ":" jws ` + extra + ` jws )* )? "}"`)
	}
	required := map[string]bool{}
	for _, r := range n.Required {
		required[r] = true
	}
	kvs := make([]string, len(props))
	for i, p := range props {
		v, err := g.visit(p.schema, depth+1)
		if err != nil {
			return "", fmt.Errorf("property %s: %w", p.name, err)
		}
		key, _ := json.Marshal(p.name)
		kvs[i] = literal(string(key)) + ` jws ":" jws ` + v + " jws"
	}

	// tail[first] is the rule for what may follow the declared properties:
	// undeclared ones, each after a comma unless it is the object's first.
	tail := map[bool]string{}
	if extra != "" {
		pair := `jstr jws ":" jws ` + extra + " jws"
		more, err := g.define(`( "," jws ` + pair + ` )*`)
		if err != nil {
			return "", err
		}
		tail[false] = more
		if tail[true], err = g.define(`( ` + pair + ` ` + more + ` )?`); err != nil {
			return "", err
		}
	}
	// seq(i, first) is the rule for the properties from the i-th on, first
	// saying whether none has been written yet; "" matches nothing more.
	memo := map[[2]int]string{}
	var seq func(i int, first bool) (string, error)
	seq = func(i int, first bool) (string, error) {
		if i == len(props) {
			return tail[first], nil
		}
		k := [2]int{i, 0}
		if first {
			k[1] = 1
		}
		if name, ok := memo[k]; ok {
			return name, nil
		}
		after, err := seq(i+1, false)
		if err != nil {
			return "", err
		}
		with := kvs[i]
		if !first {
			with = `"," jws ` + with
		}
		if after != "" {
			with += " " + after
		}
		body := with
		if !required[props[i].name] {
			without, err := seq(i+1, first)
			if err != nil {
				return "", err
			}
			if without == "" {
				body = "( " + with + " )?"
			} else {
				body = with + " | " + without
			}
		}
		name, err := g.define(body)
		memo[k] = name
		return name, err
	}
	body, err := seq(0, true)
	if err != nil {
		return "", err
	}
	return g.define(`"{" jws ` + body + ` "}"`)
}

// orderedProperties reads an object's properties in the order the schema
// gives them, which is the order the reply writes them in.
func orderedProperties(raw json.RawMessage) ([]property, error) {
	if len(bytes.TrimSpace(raw)) == 0 || string(bytes.TrimSpace(raw)) == "null" {
		return nil, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	if t, err := dec.Token(); err != nil || t != json.Delim('{') {
		return nil, errors.New("a schema's properties are not an object")
	}
	var props []property
	for dec.More() {
		t, err := dec.Token()
		if err != nil {
			return nil, err
		}
		name, _ := t.(string)
		var schema json.RawMessage
		if err := dec.Decode(&schema); err != nil {
			return nil, err
		}
		props = append(props, property{name: name, schema: schema})
	}
	return props, nil
}

// jsonLiteral is v, a JSON value of a schema's enum or const, as the text a
// reply writes it in: compact JSON.
func jsonLiteral(v json.RawMessage) (string, error) {
	var b bytes.Buffer
	if err := json.Compact(&b, v); err != nil {
		return "", fmt.Errorf("a schema's enum or const is not JSON: %w", err)
	}
	return literal(b.String()), nil
}
