package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/engine"
	"github.com/LocalKinAI/kinfer/internal/grade"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
	"github.com/LocalKinAI/kinfer/internal/sampling"
	"github.com/LocalKinAI/kinfer/internal/store"
)

const mathSuffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."

// evalRun is how a run was taken, kept beside its results as <results>.run.json:
// a run goes on only with the settings it began with, and a comparison says
// what it compares.
type evalRun struct {
	Model     string      `json:"model"`
	Path      string      `json:"path"`
	Set       string      `json:"set"`
	Subset    int         `json:"subset,omitempty"`
	Think     bool        `json:"think"`
	MaxTokens int         `json:"max_tokens"`
	Temp      float64     `json:"temperature"`
	TopP      float64     `json:"top_p,omitempty"`
	TopK      int         `json:"top_k,omitempty"`
	Seed      int64       `json:"seed,omitempty"`
	Presence  float64     `json:"presence_penalty,omitempty"`
	Repeat    float64     `json:"repeat_penalty"`
	LastN     int         `json:"repeat_last_n"`
	Suffix    string      `json:"suffix"`
	Edit      *editRecord `json:"edit,omitempty"`
}

// sameSettings is evalRun without what may change between parts of a run.
func (r evalRun) sameSettings(o evalRun) bool {
	r.Edit, o.Edit = nil, nil
	return r == o
}

// evalResult is one problem's line in a results file.
type evalResult struct {
	ID      string  `json:"id"`
	Level   any     `json:"level,omitempty"`
	Subject string  `json:"subject,omitempty"`
	Answer  string  `json:"answer"`
	Got     *string `json:"got"`
	OK      bool    `json:"ok"`
	Tokens  int     `json:"tokens"`
	Cut     bool    `json:"cut"`
	Seconds float64 `json:"seconds"`
	Reply   string  `json:"reply"`
}

type problem struct {
	id, text, answer, subject string
	level                     any
}

// tiers are kinfer eval's presets, from seconds to an hour. smoke reads the
// subset's problems with both models and says how far their predictions
// part, which tells an edit that changed nothing or broke the model from one
// worth asking questions; quick asks those problems without thinking, the
// presence penalty keeping a small model from going round in circles; full
// asks every problem the weight-space-programming paper's way. A flag given
// on the command line wins over its tier's.
var tiers = map[string]map[string]string{
	"smoke": {"subset": "100"},
	"quick": {"subset": "100", "think": "false", "temp": "0.6", "top-p": "0.95", "top-k": "20",
		"presence-penalty": "1.5", "max-tokens": "4096", "seed": "1"},
	"full": {"subset": "0", "think": "true", "temp": "0.6", "top-p": "0.95", "top-k": "20",
		"presence-penalty": "1.5", "max-tokens": "32768", "seed": "1"},
}

// cmdEval runs a set of problems through a model, many at once, grades each
// reply against its answer, and keeps a line per problem; or compares two
// runs. With a tier and a model to compare with — an edit's base, unless
// -against names another — it runs that model too, once per tier, and sets
// the two side by side.
func cmdEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	compare := fs.Bool("compare", false, "compare two runs: kinfer eval -compare <a.jsonl> <b.jsonl>")
	tier := fs.String("tier", "", "smoke (seconds: how far the predictions part), quick (minutes: 100 problems), or full (an hour: all, thinking)")
	against := fs.String("against", "", "the model to compare with (default: the base of an edit)")
	out := fs.String("o", "", "results file (default ~/.kinfer/evals/<model>/<set>[-<tier>].jsonl); a run goes on from what is in it")
	n := fs.Int("n", 64, "replies at once")
	limit := fs.Int("limit", 0, "only the first n problems")
	subset := fs.Int("subset", 0, "only n problems, the same n every time, as many of each level as there are levels to share them")
	think := fs.Bool("think", false, "let the model think first; only what follows its thinking is graded")
	maxTok := fs.Int("max-tokens", 4096, "most tokens one reply may take; a reply cut off there is graded as it stands")
	temp := fs.Float64("temp", 0, "temperature; 0 is greedy, the same reply every time")
	topP := fs.Float64("top-p", 0, "nucleus sampling, with -temp")
	topK := fs.Int("top-k", 0, "top-k sampling, with -temp")
	seed := fs.Int64("seed", 0, "sampling seed, with -temp")
	presence := fs.Float64("presence-penalty", 0, "lowers the logit of every token the reply has used, by this much")
	repeat := fs.Float64("repeat-penalty", 1, "divides the logit of a token the reply has used; 1 is off")
	lastN := fs.Int("repeat-last-n", -1, "how many of the reply's last tokens the penalties look at; -1 is all of them")
	suffix := fs.String("suffix", mathSuffix, "added to every problem")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *compare {
		if fs.NArg() != 2 {
			return fmt.Errorf("usage: kinfer eval -compare <a.jsonl> <b.jsonl>")
		}
		return compareRuns(fs.Arg(0), fs.Arg(1))
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("usage: kinfer eval [-tier smoke|quick|full] [flags] <model> <set.jsonl>   (problems with \"problem\" and \"answer\")")
	}
	if *tier != "" {
		preset, ok := tiers[*tier]
		if !ok {
			return fmt.Errorf("-tier %q: the tiers are smoke, quick and full", *tier)
		}
		given := map[string]bool{}
		fs.Visit(func(f *flag.Flag) { given[f.Name] = true })
		for k, v := range preset {
			if !given[k] {
				fs.Set(k, v)
			}
		}
	}

	st, err := store.Open()
	if err != nil {
		return err
	}
	path, err := st.Resolve(fs.Arg(0))
	if err != nil {
		return err
	}
	probs, err := loadSet(fs.Arg(1))
	if err != nil {
		return err
	}
	if *limit > 0 && *limit < len(probs) {
		probs = probs[:*limit]
	}
	if *subset > 0 {
		probs = pickSubset(probs, *subset)
	}
	other := ""
	switch {
	case *against != "":
		if other, err = st.Resolve(*against); err != nil {
			return err
		}
	case *tier != "":
		if rec := readEditRecord(path); rec != nil {
			other = rec.BaseDir
		}
	}
	if *tier == "smoke" {
		if other == "" {
			return fmt.Errorf("a smoke test compares two models; %s is no edit, so name the other with -against", fs.Arg(0))
		}
		return smokeTest(path, other, probs, *suffix)
	}

	setPath, _ := filepath.Abs(fs.Arg(1))
	settings := evalRun{Set: setPath, Subset: *subset, Think: *think, MaxTokens: *maxTok, Temp: *temp, TopP: *topP,
		TopK: *topK, Seed: *seed, Presence: *presence, Repeat: *repeat, LastN: *lastN, Suffix: *suffix}
	name := strings.TrimSuffix(filepath.Base(fs.Arg(1)), filepath.Ext(fs.Arg(1)))
	if *tier != "" {
		name += "-" + *tier
	}
	outFor := func(model string) string {
		return filepath.Join(filepath.Dir(st.Root()), "evals", modelName(model), name+".jsonl")
	}
	if *out == "" {
		*out = outFor(path)
	}
	if other != "" {
		if err := runEval(other, probs, settings, outFor(other), *n); err != nil {
			return err
		}
	}
	if err := runEval(path, probs, settings, *out, *n); err != nil {
		return err
	}
	if other != "" {
		fmt.Println()
		return compareRuns(outFor(other), *out)
	}
	return nil
}

func modelName(path string) string { return strings.TrimSuffix(filepath.Base(path), ".gguf") }

func readEditRecord(dir string) *editRecord {
	b, err := os.ReadFile(filepath.Join(dir, "edit.json"))
	if err != nil {
		return nil
	}
	var rec editRecord
	if json.Unmarshal(b, &rec) != nil {
		return nil
	}
	return &rec
}

// pickSubset takes n problems, the same ones every time: the levels in turn,
// each giving its next problem in an order fixed by the problems' ids, until
// n are taken, so every level has its share while it has problems left.
func pickSubset(probs []problem, n int) []problem {
	if n >= len(probs) {
		return probs
	}
	byLevel := map[string][]problem{}
	var levels []string
	for _, p := range probs {
		l := fmt.Sprint(p.level)
		if _, ok := byLevel[l]; !ok {
			levels = append(levels, l)
		}
		byLevel[l] = append(byLevel[l], p)
	}
	sort.Strings(levels)
	hash := func(s string) uint64 { h := fnv.New64a(); h.Write([]byte(s)); return h.Sum64() }
	for _, l := range levels {
		ps := byLevel[l]
		sort.Slice(ps, func(i, j int) bool { return hash(ps[i].id) < hash(ps[j].id) })
	}
	var out []problem
	for len(out) < n {
		for _, l := range levels {
			if ps := byLevel[l]; len(ps) > 0 && len(out) < n {
				out = append(out, ps[0])
				byLevel[l] = ps[1:]
			}
		}
	}
	return out
}

// runEval asks the model at path the problems not yet in out, as settings
// say, n at once, and prints its score.
func runEval(path string, probs []problem, settings evalRun, out string, n int) error {
	run := settings
	run.Model, run.Path, run.Edit = modelName(path), path, readEditRecord(path)
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}

	// What an earlier part of this run finished is not asked again.
	done := map[string]bool{}
	if b, err := os.ReadFile(out + ".run.json"); err == nil {
		var was evalRun
		if json.Unmarshal(b, &was) == nil && !was.sameSettings(run) {
			return fmt.Errorf("%s was taken with other settings; name another file with -o", out)
		}
	}
	prev, _ := readResults(out)
	for _, r := range prev {
		done[r.ID] = true
	}
	var todo []problem
	for _, p := range probs {
		if !done[p.id] {
			todo = append(todo, p)
		}
	}
	b, _ := json.MarshalIndent(run, "", "  ")
	if err := os.WriteFile(out+".run.json", append(b, '\n'), 0o644); err != nil {
		return err
	}
	if len(todo) == 0 {
		fmt.Printf("%s: all %d problems are in %s already\n", run.Model, len(probs), out)
		return summarize(out, probs)
	}
	if len(todo) < len(probs) {
		fmt.Printf("%d of %d problems are in %s already; asking the other %d\n", len(probs)-len(todo), len(probs), out, len(todo))
	}

	slots := min(n, len(todo))
	opts := engine.Options{Slots: slots, GPULayers: 99}
	var gen interface {
		ChatFull(context.Context, []chat.Message, engine.GenParams, func(string)) (string, engine.Stats, error)
		Close()
	}
	start := time.Now()
	var err error
	if mlxengine.IsModelDir(path) {
		// A conversation's context there bounds only how long it may grow:
		// a reply of -max-tokens after the longest problem.
		opts.ContextSize = run.MaxTokens + 4096
		gen, err = mlxengine.Open(path, opts)
	} else {
		gen, err = engine.Open(path, opts)
	}
	if err != nil {
		return err
	}
	defer gen.Close()
	fmt.Printf("%s loaded in %v; %d problems, %d at once\n", run.Model, time.Since(start).Round(time.Millisecond), len(todo), slots)

	f, err := os.OpenFile(out, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	params := engine.GenParams{
		Params: sampling.Params{Temperature: float32(run.Temp), TopP: float32(run.TopP), TopK: run.TopK, Seed: run.Seed,
			RepeatPenalty: float32(run.Repeat), RepeatLastN: run.LastN, PresencePenalty: float32(run.Presence)},
		MaxTokens: run.MaxTokens, NoThink: !run.Think,
	}
	if params.TopP == 0 {
		params.TopP = 1
	}

	work := make(chan problem)
	var mu sync.Mutex
	var firstErr error
	var finished, right, tokens int
	began := time.Now()
	last := time.Now()
	var wg sync.WaitGroup
	for w := 0; w < slots; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range work {
				t := time.Now()
				msgs := []chat.Message{{Role: "user", Content: p.text + run.Suffix}}
				text, stats, err := gen.ChatFull(ctx, msgs, params, nil)
				mu.Lock()
				if err != nil {
					if firstErr == nil && ctx.Err() == nil {
						firstErr = fmt.Errorf("%s: %w", p.id, err)
					}
					stop()
					mu.Unlock()
					continue
				}
				r := evalResult{ID: p.id, Level: p.level, Subject: p.subject, Answer: p.answer,
					Tokens: stats.EvalTokens, Cut: stats.Truncated || stats.Timeout, Seconds: time.Since(t).Seconds(), Reply: text}
				if got, ok := grade.Boxed(answerPart(text)); ok {
					r.Got = &got
					r.OK = grade.Same(got, p.answer)
				}
				line, _ := json.Marshal(r)
				f.Write(append(line, '\n'))
				finished++
				tokens += r.Tokens
				if r.OK {
					right++
				}
				if time.Since(last) >= 10*time.Second {
					last = time.Now()
					el := time.Since(began)
					fmt.Printf("  %d/%d  %.1f%% right  %.0f tok/s  %v left\n", finished, len(todo),
						100*float64(right)/float64(finished), float64(tokens)/el.Seconds(),
						(time.Duration(float64(el) / float64(finished) * float64(len(todo)-finished))).Round(time.Second))
				}
				mu.Unlock()
			}
		}()
	}
feed:
	for _, p := range todo {
		select {
		case work <- p:
		case <-ctx.Done():
			break feed
		}
	}
	close(work)
	wg.Wait()
	el := time.Since(began)
	fmt.Printf("%d replies in %v, %.0f tok/s together\n", finished, el.Round(time.Second), float64(tokens)/el.Seconds())
	if firstErr != nil {
		return firstErr
	}
	if ctx.Err() != nil {
		return fmt.Errorf("stopped; %d of %d are in %s, and the same command goes on from there", len(done)+finished, len(probs), out)
	}
	return summarize(out, probs)
}

// answerPart is the part of a reply that is graded: what follows the
// thinking, or nothing when the reply was cut off thinking.
func answerPart(text string) string {
	if i := strings.LastIndex(text, "</think>"); i >= 0 {
		return text[i+len("</think>"):]
	}
	if strings.HasPrefix(strings.TrimSpace(text), "<think>") {
		return ""
	}
	return text
}

// loadSet reads problems from a JSON array or a file of JSON lines, each with
// a problem (or prompt, or question) and its answer.
func loadSet(path string) ([]problem, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var items []map[string]any
	if t := strings.TrimSpace(string(b)); strings.HasPrefix(t, "[") {
		if err := json.Unmarshal(b, &items); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
	} else {
		sc := bufio.NewScanner(strings.NewReader(t))
		sc.Buffer(make([]byte, 1<<20), 64<<20)
		for sc.Scan() {
			if strings.TrimSpace(sc.Text()) == "" {
				continue
			}
			var m map[string]any
			if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
				return nil, fmt.Errorf("%s line %d: %w", path, len(items)+1, err)
			}
			items = append(items, m)
		}
	}
	str := func(m map[string]any, keys ...string) string {
		for _, k := range keys {
			switch v := m[k].(type) {
			case string:
				return v
			case float64:
				return fmt.Sprint(v)
			}
		}
		return ""
	}
	var out []problem
	seen := map[string]bool{}
	for i, m := range items {
		p := problem{id: str(m, "uid", "unique_id", "id"), text: str(m, "problem", "prompt", "question"),
			answer: str(m, "answer"), subject: str(m, "subject"), level: m["level"]}
		if p.id == "" {
			p.id = fmt.Sprint(i)
		}
		if p.text == "" || p.answer == "" {
			return nil, fmt.Errorf("%s: item %d has no problem or no answer", path, i+1)
		}
		if seen[p.id] {
			return nil, fmt.Errorf("%s: two problems are %s", path, p.id)
		}
		seen[p.id] = true
		out = append(out, p)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s holds no problems", path)
	}
	return out, nil
}

func readResults(path string) ([]evalResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []evalResult
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	for sc.Scan() {
		var r evalResult
		if json.Unmarshal(sc.Bytes(), &r) == nil && r.ID != "" {
			out = append(out, r)
		}
	}
	return out, sc.Err()
}

// summarize prints a run's score over the problems of the set it was asked.
func summarize(path string, probs []problem) error {
	res, err := readResults(path)
	if err != nil {
		return err
	}
	want := map[string]bool{}
	for _, p := range probs {
		want[p.id] = true
	}
	var n, ok, cut, tokens int
	byLevel := map[string][2]int{}
	for _, r := range res {
		if !want[r.ID] {
			continue
		}
		n++
		tokens += r.Tokens
		if r.Cut {
			cut++
		}
		lv := byLevel[fmt.Sprint(r.Level)]
		lv[1]++
		if r.OK {
			ok++
			lv[0]++
		}
		byLevel[fmt.Sprint(r.Level)] = lv
	}
	if n == 0 {
		return fmt.Errorf("%s holds none of these problems", path)
	}
	fmt.Printf("\n%s\n  %d of %d right (%.1f%%), %.0f tokens a reply, %d cut off\n",
		path, ok, n, 100*float64(ok)/float64(n), float64(tokens)/float64(n), cut)
	if len(byLevel) > 1 {
		var levels []string
		for l := range byLevel {
			levels = append(levels, l)
		}
		sort.Strings(levels)
		for _, l := range levels {
			v := byLevel[l]
			fmt.Printf("  level %-6s %3d of %3d (%.0f%%)\n", l, v[0], v[1], 100*float64(v[0])/float64(v[1]))
		}
	}
	return nil
}

// compareRuns sets two runs side by side over the problems both answered:
// how many each got right, which problems one got right and the other wrong,
// and how likely that many would split the way they did by chance.
func compareRuns(a, b string) error {
	ra, err := readResults(a)
	if err != nil {
		return err
	}
	rb, err := readResults(b)
	if err != nil {
		return err
	}
	byID := map[string]evalResult{}
	for _, r := range ra {
		byID[r.ID] = r
	}
	var n, okA, okB, tokA, tokB, cutA, cutB, onlyA, onlyB int
	var fixed, broke []string
	for _, y := range rb {
		x, ok := byID[y.ID]
		if !ok {
			continue
		}
		n++
		tokA += x.Tokens
		tokB += y.Tokens
		if x.Cut {
			cutA++
		}
		if y.Cut {
			cutB++
		}
		switch {
		case x.Cut && !y.Cut:
			onlyA++
		case y.Cut && !x.Cut:
			onlyB++
		}
		if x.OK {
			okA++
		}
		if y.OK {
			okB++
		}
		switch {
		case !x.OK && y.OK:
			fixed = append(fixed, y.ID)
		case x.OK && !y.OK:
			broke = append(broke, y.ID)
		}
	}
	if n == 0 {
		return fmt.Errorf("%s and %s share no problems", a, b)
	}
	name := func(path string) string {
		var r evalRun
		if b, err := os.ReadFile(path + ".run.json"); err == nil && json.Unmarshal(b, &r) == nil {
			s := r.Model
			if r.Edit != nil && r.Edit.Note != "" {
				s += " (" + r.Edit.Note + ")"
			}
			return s
		}
		return filepath.Base(path)
	}
	pct := func(k int) string { return fmt.Sprintf("%d (%.1f%%)", k, 100*float64(k)/float64(n)) }
	fmt.Printf("%d problems both runs answered\n\n", n)
	fmt.Printf("  %-22s %-24s %s\n", "", "A: "+name(a), "B: "+name(b))
	fmt.Printf("  %-22s %-24s %s   (%+d)\n", "right", pct(okA), pct(okB), okB-okA)
	fmt.Printf("  %-22s %-24.0f %.0f\n", "tokens a reply", float64(tokA)/float64(n), float64(tokB)/float64(n))
	fmt.Printf("  %-22s %-24d %d\n", "cut off", cutA, cutB)
	fmt.Printf("\n  B got right where A did not: %d\n  A got right where B did not: %d\n", len(fixed), len(broke))
	fmt.Printf("  chance of a split this uneven or more if they were alike: p = %.3g (McNemar, exact)\n",
		mcnemar(len(fixed), len(broke)))
	if onlyA+onlyB > 0 {
		// Running out of tokens is a finding of its own: a model that thinks
		// in circles, or one that has stopped.
		fmt.Printf("\n  cut off in A alone: %d, in B alone: %d; p = %.3g (McNemar, exact)\n", onlyA, onlyB, mcnemar(onlyA, onlyB))
	}
	show := func(what string, ids []string) {
		if len(ids) == 0 {
			return
		}
		if len(ids) > 12 {
			ids = append(ids[:12:12], fmt.Sprintf("and %d more", len(ids)-12))
		}
		fmt.Printf("\n  %s:\n    %s\n", what, strings.Join(ids, "\n    "))
	}
	show("B right, A wrong", fixed)
	show("A right, B wrong", broke)
	return nil
}

// mcnemar is the two-sided exact McNemar p-value of b and c discordant pairs:
// twice the binomial(b+c, 1/2) tail at the smaller of the two.
func mcnemar(b, c int) float64 {
	n, k := b+c, min(b, c)
	if n == 0 {
		return 1
	}
	tail := 0.0
	for i := 0; i <= k; i++ {
		lg, _ := math.Lgamma(float64(n + 1))
		li, _ := math.Lgamma(float64(i + 1))
		lr, _ := math.Lgamma(float64(n - i + 1))
		tail += math.Exp(lg - li - lr - float64(n)*math.Ln2)
	}
	return math.Min(1, 2*tail)
}
