package tools

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// FunctionXMLGrammar is a GBNF grammar for the tool calls of a FunctionXML
// reply, from its first <tool_call> to the end: one call or more, each well
// formed, then nothing but the end of the turn.
//
// A parameter of a declared function is held to its schema where the schema
// is one plain type that text can be checked against — an integer, a number, a
// boolean, an enum, a JSON object or array. Anything else, a string above all,
// may be any text that does not close the parameter early — the parser takes
// a value up to the first </parameter> — nor write the closes that follow it;
// see closing. A file's contents, HTML included, are still written as they
// are.
//
// A function or parameter the request did not declare is written as it is,
// its values unchecked, and the caller reports it as it always has. Holding a
// model to the declared names instead would turn the call it meant into one
// it did not: in 19,813 calls the LocalKin swarm's seats logged, 4 asked for a
// "bash" or "shell" they did not have, and a grammar of declared names only
// would have run something else of theirs in its place.
//
// Which parameters a call gives, in what order, is left to the model: a
// grammar that required them would force it to invent a value it has no use
// for, and the tool reports a missing one better than a made-up one.
func FunctionXMLGrammar(ts []Tool) (string, error) {
	g := grammar{others: map[string]string{}}
	var fns, names []string
	for _, t := range ts {
		name := t.Function.Name
		if name == "" {
			continue
		}
		fn := fmt.Sprintf("f%d", len(fns))
		props, err := properties(t.Function.Parameters)
		if err != nil {
			return "", fmt.Errorf("tool %s: %w", name, err)
		}
		var params, pnames []string
		for _, p := range props {
			rule := fmt.Sprintf("%sp%d", fn, len(params))
			g.rule(rule, literal("<parameter="+p.name+">")+" "+g.value(p.schema)+` "</parameter>" ws`)
			params = append(params, rule)
			pnames = append(pnames, p.name)
		}
		params = append(params, g.otherParam(pnames))
		g.rule(fn, literal("<function="+name+">")+" ws ( "+strings.Join(params, " | ")+` )* "</function>"`)
		fns = append(fns, fn)
		names = append(names, name)
	}
	if len(fns) == 0 {
		return "", errors.New("no tool has a name")
	}
	g.unlisted("fx", names)
	g.rule("fother", `"<function=" fx0 ">" ws ( "<parameter=" anyname ">" txt "</parameter>" ws )* "</function>"`)

	var b strings.Builder
	b.WriteString("root ::= call ( ws call )* ws\n")
	b.WriteString(`call ::= "<tool_call>" ws fn ws "</tool_call>"` + "\n")
	b.WriteString("fn ::= " + strings.Join(fns, " | ") + " | fother\n")
	b.WriteString("ws ::= [ \\t\\n]{0,16}\n")
	b.WriteString(`anyname ::= [^<>\n]+` + "\n")
	b.WriteString(g.rules.String())
	b.WriteString(notClosing("txt", closing))
	if g.json {
		b.WriteString(jsonRules)
	}
	return b.String(), nil
}

// grammar collects rules, and which shared ones they use.
type grammar struct {
	rules strings.Builder
	json  bool

	// others is the rule for a parameter outside each set of declared names,
	// by the set: most tools of an agent share theirs with another.
	others map[string]string
}

// otherParam is the rule for a parameter none of names, written as text.
func (g *grammar) otherParam(names []string) string {
	key := strings.Join(names, "\x00")
	if rule, ok := g.others[key]; ok {
		return rule
	}
	rule := fmt.Sprintf("px%d", len(g.others))
	g.unlisted(rule+"n", names)
	g.rule(rule, `"<parameter=" `+rule+`n0 ">" txt "</parameter>" ws`)
	g.others[key] = rule
	return rule
}

// unlisted writes rules prefix0, prefix1, … for a name that is none of names:
// one rule for each point of their trie, which a name either leaves — by a
// character none of them has there, or by ending where none of them ends — or
// follows on. A name is anything up to its >, on one line.
func (g *grammar) unlisted(prefix string, names []string) {
	type node struct {
		end  bool
		next map[rune]*node
	}
	root := &node{next: map[rune]*node{}}
	for _, name := range names {
		n := root
		for _, r := range name {
			if n.next[r] == nil {
				n.next[r] = &node{next: map[rune]*node{}}
			}
			n = n.next[r]
		}
		n.end = true
	}
	id := 0
	var walk func(n *node, mayEnd bool) string
	walk = func(n *node, mayEnd bool) string {
		rule := fmt.Sprintf("%s%d", prefix, id)
		id++
		runes := make([]rune, 0, len(n.next))
		for r := range n.next {
			runes = append(runes, r)
		}
		slices.Sort(runes)
		alts := []string{`[^<>\n` + classChars(runes) + `] [^<>\n]*`}
		for _, r := range runes {
			child := n.next[r]
			alts = append(alts, literal(string(r))+" "+walk(child, !child.end))
		}
		body := "( " + strings.Join(alts, " | ") + " )"
		if mayEnd {
			body += "?"
		}
		g.rule(rule, body)
		return rule
	}
	walk(root, false)
}

// classChars writes runes for the inside of a GBNF character class, after
// its first character.
func classChars(runes []rune) string {
	var b strings.Builder
	dash := false
	for _, r := range runes {
		switch r {
		case '-':
			dash = true // only literal as the class's last character
		case ']', '[', '\\', '"':
			b.WriteByte('\\')
			b.WriteRune(r)
		case '\n':
			b.WriteString(`\n`)
		case '\t':
			b.WriteString(`\t`)
		case '\r':
			b.WriteString(`\r`)
		default:
			if r < 0x20 || r == 0x7f {
				fmt.Fprintf(&b, `\x%02X`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	if dash {
		b.WriteByte('-')
	}
	return b.String()
}

func (g *grammar) rule(name, body string) {
	fmt.Fprintf(&g.rules, "%s ::= %s\n", name, body)
}

// value is the expression a parameter's value has to match, between its
// <parameter=…> and its </parameter>.
func (g *grammar) value(schema json.RawMessage) string {
	var s struct {
		Type json.RawMessage   `json:"type"`
		Enum []json.RawMessage `json:"enum"`
	}
	_ = json.Unmarshal(schema, &s)

	if len(s.Enum) > 0 {
		var alts []string
		for _, v := range s.Enum {
			alts = append(alts, literal(scalarText(v)))
		}
		return "ws ( " + strings.Join(alts, " | ") + " ) ws"
	}
	var typ string
	_ = json.Unmarshal(s.Type, &typ)
	switch typ {
	case "integer":
		return `ws "-"? [0-9]{1,19} ws`
	case "number":
		g.json = true
		return "ws jnum ws"
	case "boolean":
		return `ws ( "true" | "false" ) ws`
	case "object":
		g.json = true
		return "ws jobj ws"
	case "array":
		g.json = true
		return "ws jarr ws"
	}
	return "txt"
}

type property struct {
	name   string
	schema json.RawMessage
}

// properties reads a function's parameters, sorted so the same tools always
// make the same grammar.
func properties(params json.RawMessage) ([]property, error) {
	if len(params) == 0 || string(params) == "null" {
		return nil, nil
	}
	var s struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	if err := json.Unmarshal(params, &s); err != nil {
		return nil, err
	}
	var ps []property
	for name, schema := range s.Properties {
		if name != "" {
			ps = append(ps, property{name, schema})
		}
	}
	slices.SortFunc(ps, func(a, b property) int { return strings.Compare(a.name, b.name) })
	return ps, nil
}

// closing is what a value may not contain: its own close, and the closes
// after it. A model that leaves out </parameter> writes one of the others
// next; held to text that could contain them, it would write it as part of
// the value, find the end of its turn ruled out and write on to the end of its
// budget. Without them, its "</" can only go on as </parameter>, and the call
// closes as it meant to.
var closing = []string{"</parameter>", "</function>", "</tool_call>"}

// notClosing is a rule for any text that contains none of ends, which all
// start with "<" and have it nowhere else. Past each "<" the text either
// breaks away from every end before one is complete or stops, and stopping is
// only possible where a "<" comes next: the rule's own next part, or what
// follows the rule, which is an end itself.
func notClosing(name string, ends []string) string {
	type node struct {
		leaf bool
		next map[rune]*node
	}
	root := &node{next: map[rune]*node{}}
	for _, e := range ends {
		n := root
		for _, r := range e[1:] {
			if n.next[r] == nil {
				n.next[r] = &node{next: map[rune]*node{}}
			}
			n = n.next[r]
		}
		n.leaf = true
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s ::= [^<]* ( \"<\" %s0 )*\n", name, name)
	id := 0
	var walk func(n *node) string
	walk = func(n *node) string {
		rule := fmt.Sprintf("%s%d", name, id)
		id++
		runes := make([]rune, 0, len(n.next))
		for r := range n.next {
			runes = append(runes, r)
		}
		slices.Sort(runes)
		alts := []string{"[^<" + classChars(runes) + "] [^<]*"}
		for _, r := range runes {
			if child := n.next[r]; !child.leaf {
				alts = append(alts, literal(string(r))+" "+walk(child))
			}
		}
		fmt.Fprintf(&b, "%s ::= ( %s )?\n", rule, strings.Join(alts, " | "))
		return rule
	}
	walk(root)
	return b.String()
}

// literal quotes s as a GBNF string.
func literal(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch {
		case r == '"' || r == '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(&b, `\x%02X`, r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// jsonRules is JSON, for the parameters whose schema asks for an object, an
// array or a number; the template writes those with tojson.
const jsonRules = `jv ::= jobj | jarr | jstr | jnum | "true" | "false" | "null"
jobj ::= "{" jws ( jstr jws ":" jws jv jws ( "," jws jstr jws ":" jws jv jws )* )? "}"
jarr ::= "[" jws ( jv jws ( "," jws jv jws )* )? "]"
jstr ::= "\"" ( [^"\\\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""
jnum ::= "-"? ( "0" | [1-9] [0-9]{0,15} ) ( "." [0-9]{1,16} )? ( [eE] [-+]? [0-9]{1,3} )?
jws ::= [ \t\n]{0,16}
`
