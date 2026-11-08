package sampling

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/tools"
)

// These run llama.cpp's grammar sampler over a real vocabulary — Qwen2.5's,
// read without its weights — because what a grammar allows depends on how the
// text breaks into tokens, and a grammar that is right on paper can still
// strand a model the tokenizer splits differently.

const vocabModel = "../../models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

func testVocab(t *testing.T) llama.Vocab {
	t.Helper()
	dir, err := nativelib.Prepare()
	if err != nil {
		t.Skipf("no llama.cpp libraries: %v", err)
	}
	if err := llama.Bind(dir); err != nil {
		t.Skipf("llama.cpp would not bind: %v", err)
	}
	mp := llama.DefaultModelParams()
	mp.VocabOnly = 1
	m := llama.LoadModel(vocabModel, mp)
	if m == 0 {
		t.Skipf("no vocabulary at %s", vocabModel)
	}
	t.Cleanup(func() { llama.FreeModel(m) })
	return llama.GetVocab(m)
}

var weatherTools = []tools.Tool{
	{Type: "function", Function: tools.Function{
		Name: "get_weather",
		Parameters: json.RawMessage(`{"type":"object","properties":{
			"city":{"type":"string"},
			"days":{"type":"integer"},
			"unit":{"type":"string","enum":["celsius","fahrenheit"]},
			"hourly":{"type":"boolean"}}}`),
	}},
	{Type: "function", Function: tools.Function{
		Name: "write_file",
		Parameters: json.RawMessage(`{"type":"object","properties":{
			"path":{"type":"string"},
			"content":{"type":"string"},
			"tags":{"type":"array"}}}`),
	}},
}

// allows reports whether g, in its current state, lets tok come next.
func allows(g llama.Sampler, tok llama.Token) bool {
	one := llama.TokenData{ID: tok, Logit: 1}
	llama.SamplerApply(g, &llama.TokenDataArray{Data: &one, Size: 1, Selected: -1})
	return !math.IsInf(float64(one.Logit), -1)
}

// follow feeds text through a fresh grammar token by token and returns how
// many tokens it took before one was ruled out, and whether the end of the
// turn may follow. It never accepts a ruled-out token: that would abort.
func follow(t *testing.T, v llama.Vocab, rules, text string) (taken, of int, canEnd bool) {
	t.Helper()
	g := llama.SamplerGrammar(v, rules, "root")
	if g == 0 {
		t.Fatalf("llama.cpp did not take the grammar:\n%s", rules)
	}
	defer llama.SamplerFree(g)
	toks, err := llama.Tokenize(v, text, false, true)
	if err != nil {
		t.Fatal(err)
	}
	for i, tok := range toks {
		if !allows(g, tok) {
			return i, len(toks), false
		}
		llama.SamplerAccept(g, tok)
	}
	return len(toks), len(toks), allows(g, llama.EOS(v))
}

func TestGrammarHoldsAFunctionXMLCall(t *testing.T) {
	v := testVocab(t)
	rules, err := tools.FunctionXMLGrammar(weatherTools)
	if err != nil {
		t.Fatal(err)
	}

	good := []string{
		"<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
		// Parameters in any order, and an enum and a boolean.
		"<tool_call>\n<function=get_weather>\n<parameter=unit>\ncelsius\n</parameter>\n<parameter=hourly>\nfalse\n</parameter>\n<parameter=city>\n東京\n</parameter>\n</function>\n</tool_call>",
		// Two calls, and a file whose contents close tags of their own.
		"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=write_file>\n<parameter=path>\nindex.html\n</parameter>\n<parameter=content>\n<p>a < b</p>\n</div></param\n</parameter>\n<parameter=tags>\n[\"web\", 2]\n</parameter>\n</function>\n</tool_call>",
		// No parameters at all.
		"<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
		// A function, and a parameter, the request did not declare: written as
		// they are, for the caller to report, rather than turned into another.
		"<tool_call>\n<function=bash>\n<parameter=command>\nls -la | head\n</parameter>\n</function>\n</tool_call>",
		"<tool_call>\n<function=get_wether>\n<parameter=city>\nBerlin\n</parameter>\n</function>\n</tool_call>",
		"<tool_call>\n<function=get_weather>\n<parameter=country>\nDE\n</parameter>\n<parameter=cit>\nx\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
	}
	for _, text := range good {
		taken, of, canEnd := follow(t, v, rules, text)
		if taken != of {
			t.Errorf("a well-formed call was cut off after %d of %d tokens:\n%s", taken, of, text)
		} else if !canEnd {
			t.Errorf("the turn may not end after a complete call:\n%s", text)
		}
	}

	bad := map[string]string{
		"a word where an integer belongs": "<tool_call>\n<function=get_weather>\n<parameter=days>\nthree",
		"a value outside the enum":        "<tool_call>\n<function=get_weather>\n<parameter=unit>\nkelvin",
		"a boolean that is not one":       "<tool_call>\n<function=get_weather>\n<parameter=hourly>\nTrue",
		"prose after the call":            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>\nDone!",
		"a Python list for an array":      "<tool_call>\n<function=write_file>\n<parameter=tags>\n['web']",
		"a call left open":                "<tool_call>\n<function=get_weather>\n</tool_call>",
		"JSON where the XML form belongs": "<tool_call>\n{\"name\": \"get_weather\"",
		// A value left open runs into the closes after it; held to text, the
		// model would write on past the end of its turn.
		"a value run into </function>":  "<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</function>",
		"a value run into </tool_call>": "<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</tool_call>",
	}
	for why, text := range bad {
		if taken, of, _ := follow(t, v, rules, text); taken == of {
			t.Errorf("%s got through:\n%s", why, text)
		}
	}

	// An unfinished call may not end the turn either.
	if _, _, canEnd := follow(t, v, rules, "<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin"); canEnd {
		t.Error("the turn may end in the middle of a parameter")
	}
}

// TestHeldDrawFallsBackToTheGrammar checks the draw itself: when the chain's
// choice is ruled out, the token drawn is the best the grammar allows — and in
// the common case, where it is not, nothing changes.
func TestHeldDrawFallsBackToTheGrammar(t *testing.T) {
	v := testVocab(t)
	rules, err := tools.FunctionXMLGrammar(weatherTools)
	if err != nil {
		t.Fatal(err)
	}
	one := func(text string) llama.Token {
		toks, err := llama.Tokenize(v, text, false, true)
		if err != nil || len(toks) != 1 {
			t.Fatalf("%q is %d tokens, want 1", text, len(toks))
		}
		return toks[0]
	}
	open, nl, hello := one("<tool_call>"), one("\n"), one("Hello")

	s := New(Params{Temperature: 0})
	defer s.Close()
	if err := s.Constrain(Grammar{Vocab: v, Rules: rules, Trigger: open, Open: NoToken, Close: NoToken}); err != nil {
		t.Fatal(err)
	}
	s.saw(hello)
	if s.held {
		t.Fatal("held before the trigger was drawn")
	}
	s.saw(open)
	if !s.held {
		t.Fatal("not held after the trigger was drawn")
	}

	logits := make([]float32, s.nVocab)
	logits[hello], logits[nl] = 10, 5
	start := time.Now()
	if got := s.drawHeld(logits); got != nl {
		t.Errorf("drew %q, want the newline the grammar allows", llama.TokenToPiece(v, got, true))
	}
	t.Logf("a draw that fell back to the grammar took %v over %d candidates", time.Since(start), len(logits))

	// "<function=" next: its first token, ranked first, is drawn as it is.
	toks, _ := llama.Tokenize(v, "<function=", false, true)
	clear(logits)
	logits[toks[0]] = 10
	start = time.Now()
	if got := s.drawHeld(logits); got != toks[0] {
		t.Errorf("drew %q, want %q", llama.TokenToPiece(v, got, true), llama.TokenToPiece(v, toks[0], true))
	}
	t.Logf("a draw the grammar allowed took %v", time.Since(start))
}

func TestGrammarWaitsForTheThinkBlockToClose(t *testing.T) {
	v := testVocab(t)
	rules, err := tools.FunctionXMLGrammar(weatherTools)
	if err != nil {
		t.Fatal(err)
	}
	one := func(text string) llama.Token {
		toks, _ := llama.Tokenize(v, text, false, true)
		return toks[0]
	}
	// Stand-ins for <think> and </think>, which Qwen2.5 does not have.
	call, open, close := one("<tool_call>"), one("<|im_start|>"), one("<|im_end|>")

	s := New(Params{Temperature: 0})
	defer s.Close()
	if err := s.Constrain(Grammar{Vocab: v, Rules: rules, Trigger: call, Open: open, Close: close, Thinking: true}); err != nil {
		t.Fatal(err)
	}
	s.saw(call)
	if s.held {
		t.Fatal("a call written in the think block the prompt opened was held to the grammar")
	}
	s.saw(close)
	s.saw(open)
	s.saw(call)
	if s.held {
		t.Fatal("a call written in a think block the model opened was held to the grammar")
	}
	s.saw(close)
	s.saw(call)
	if !s.held {
		t.Fatal("a call after the think block was not held")
	}
	s.Reset()
	if s.held || !s.thinking {
		t.Error("Reset did not put the grammar back inside the prompt's think block")
	}
}

// TestGrammarTakesEveryCallOf checks a whole agent's tools, read from the JSON
// array of them at $KINFER_GRAMMAR_TOOLS — the LocalKin swarm's skills, say:
// that llama.cpp takes the grammar they make and how long that takes, that a
// call to each, written the way the template writes one, with values of every
// kind its schema allows, gets through, and what a draw that falls back to the
// grammar costs where the function's name begins.
func TestGrammarTakesEveryCallOf(t *testing.T) {
	path := os.Getenv("KINFER_GRAMMAR_TOOLS")
	if path == "" {
		t.Skip("set KINFER_GRAMMAR_TOOLS to a JSON array of tools")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var ts []tools.Tool
	if err := json.Unmarshal(raw, &ts); err != nil {
		t.Fatal(err)
	}
	v := testVocab(t)

	start := time.Now()
	rules, err := tools.FunctionXMLGrammar(ts)
	if err != nil {
		t.Fatal(err)
	}
	built := time.Since(start)
	start = time.Now()
	g := llama.SamplerGrammar(v, rules, "root")
	if g == 0 {
		t.Fatal("llama.cpp did not take the grammar")
	}
	llama.SamplerFree(g)
	t.Logf("%d tools: grammar of %d bytes built in %v, parsed by llama.cpp in %v",
		len(ts), len(rules), built, time.Since(start))

	texts := []string{
		"北京今天的天气",
		"line one\nline two\n\tindented",
		"<html><body><p class=\"x\">a < b && c > d</p></body></html>",
		`{"looks": "like json"}`,
		"say \"hi\" \\ </div></param",
		"x",
	}
	values := map[string][]string{
		"number":  {"3", "-2.5", "1e3", "0.75"},
		"integer": {"42", "-1", "0"},
		"boolean": {"true", "false"},
		"array":   {`["a", 1, {"k": [true, null]}]`, "[]"},
		"object":  {`{"q": "北京", "n": 3}`, "{}"},
	}
	calls := 0
	for round := range len(texts) {
		for i, tool := range ts {
			var schema struct {
				Properties map[string]struct {
					Type string `json:"type"`
				} `json:"properties"`
			}
			_ = json.Unmarshal(tool.Function.Parameters, &schema)
			args := map[string]json.RawMessage{}
			for name, p := range schema.Properties {
				if vs, ok := values[p.Type]; ok {
					args[name] = json.RawMessage(vs[(round+i)%len(vs)])
				} else {
					s, _ := json.Marshal(texts[(round+i)%len(texts)])
					args[name] = s
				}
			}
			a, _ := json.Marshal(args)
			text := strings.TrimPrefix(tools.FunctionXML.RenderCalls([]tools.Call{{Name: tool.Function.Name, Arguments: a}}), "\n")
			taken, of, canEnd := follow(t, v, rules, text)
			if taken != of || !canEnd {
				t.Errorf("a call to %s stopped after %d of %d tokens (may end: %v):\n%s", tool.Function.Name, taken, of, canEnd, text)
			}
			calls++
		}
	}
	t.Logf("%d calls got through", calls)

	g = llama.SamplerGrammar(v, rules, "root")
	defer llama.SamplerFree(g)
	toks, _ := llama.Tokenize(v, "<tool_call>\n<function=", false, true)
	for _, tok := range toks {
		llama.SamplerAccept(g, tok)
	}
	cur := make([]llama.TokenData, llama.NVocab(v))
	for i := range cur {
		cur[i] = llama.TokenData{ID: llama.Token(i)}
	}
	start = time.Now()
	llama.SamplerApply(g, &llama.TokenDataArray{Data: &cur[0], Size: uint64(len(cur)), Selected: -1})
	t.Logf("a draw that falls back to the grammar at a function's name: %v", time.Since(start))
}
