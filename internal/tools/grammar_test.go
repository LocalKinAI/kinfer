package tools

import (
	"encoding/json"
	"strings"
	"testing"
)

// What the grammar lets a model write is checked against a real vocabulary in
// internal/sampling. These cover the text: that it is the same for the same
// tools, and that names cannot break out of their quotes.

func TestFunctionXMLGrammarIsStable(t *testing.T) {
	ts := []Tool{{Function: Function{Name: "f", Parameters: json.RawMessage(
		`{"properties":{"b":{"type":"integer"},"a":{"type":"string"},"c":{"type":"array"}}}`)}}}
	first, err := FunctionXMLGrammar(ts)
	if err != nil {
		t.Fatal(err)
	}
	for range 20 {
		if again, _ := FunctionXMLGrammar(ts); again != first {
			t.Fatal("the same tools made two grammars; parameter order must not follow map order")
		}
	}
	for _, want := range []string{
		`f0p0 ::= "<parameter=a>" txt "</parameter>" ws`,
		`f0p1 ::= "<parameter=b>" ws "-"? [0-9]{1,19} ws "</parameter>" ws`,
		`f0p2 ::= "<parameter=c>" ws jarr ws "</parameter>" ws`,
		"txt ::= ", "jarr ::= ",
	} {
		if !strings.Contains(first, want) {
			t.Errorf("grammar lacks %q:\n%s", want, first)
		}
	}
}

func TestFunctionXMLGrammarQuotesNames(t *testing.T) {
	g, err := FunctionXMLGrammar([]Tool{{Function: Function{Name: `say"hi\`}}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(g, `"<function=say\"hi\\>"`) {
		t.Errorf("name not escaped:\n%s", g)
	}
	if strings.Contains(g, "jv ::=") {
		t.Errorf("JSON rules nothing uses:\n%s", g)
	}
}

func TestFunctionXMLGrammarNeedsANamedTool(t *testing.T) {
	if _, err := FunctionXMLGrammar([]Tool{{Function: Function{Description: "no name"}}}); err == nil {
		t.Error("a grammar with no function to call")
	}
}
