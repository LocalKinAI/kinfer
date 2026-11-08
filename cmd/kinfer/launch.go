// kinfer launch: a coding agent, pointed at kinfer.
//
// Ollama's `ollama launch claude` showed what was missing: an agent that talks
// to a server needs nothing from the server but an address and a model, and
// getting those into the agent's own variables and flags is the part people
// get wrong — Claude Code alone asks for a model under six names. kinfer
// already speaks what the agents speak: Anthropic's Messages API for Claude
// Code, OpenAI's Responses API for Codex. So launch starts
// the server if nothing is serving, settles on a model, and hands the agent its
// settings in its environment and on its command line. Nothing is written to
// an agent's own configuration, so there is nothing to restore afterwards: the
// agent started without launch is the agent it was. An agent that takes its
// provider only from files (Pi, Hermes) is given a directory of its own under
// ~/.kinfer/agents instead of the one it keeps; one that runs as a background
// service of its own (OpenClaw) is told how, and nothing is run.
package main

import (
	"bufio"
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"
)

// agent is one coding agent launch knows how to point at kinfer.
type agent struct {
	name    string // as typed: kinfer launch <name>
	bin     string // the program looked for on PATH
	what    string // what it is, in the menu
	install string // how to get it, when it is missing
	// setup gives the arguments that go before the user's own and the
	// variables set for it, for a model whose longest conversation is window
	// tokens (0 when the server does not say); drop is what is taken out of
	// the environment.
	setup func(base, model string, window int) (args, env []string)
	drop  []string
	// files are what kinfer writes, by name, into agentDir for an agent that
	// reads its provider only from files; setup points the agent there.
	files func(base, model string, window int) map[string][]byte
	// printOnly, for an agent that runs as a background service of its own,
	// is how to point it at kinfer: launch prints it and runs nothing.
	printOnly func(base, model string, window int) string
	// minWindow is the context the agent asks for at least.
	minWindow int
}

var agents = []agent{
	{
		name: "claude", bin: "claude", what: "Claude Code",
		install: "npm install -g @anthropic-ai/claude-code",
		setup:   claudeSetup,
		// An API key beside the token is a conflict Claude Code stops to ask
		// about, and kinfer needs neither.
		drop: []string{"ANTHROPIC_API_KEY"},
	},
	{
		name: "codex", bin: "codex", what: "Codex",
		install: "npm install -g @openai/codex",
		setup:   codexSetup,
	},
	// The three below are written from their documentation (2026-11-07) and
	// have not been run against kinfer.
	{
		name: "pi", bin: "pi", what: "Pi",
		install: "npm install -g @earendil-works/pi-coding-agent",
		setup:   piSetup, files: piFiles,
	},
	{
		name: "hermes", bin: "hermes", what: "Hermes Agent",
		install: "its installer, from https://hermes-agent.nousresearch.com",
		setup:   hermesSetup, files: hermesFiles,
		// Hermes says it needs 64k tokens of context at least.
		minWindow: 65536,
	},
	{
		name: "openclaw", bin: "openclaw", what: "OpenClaw",
		install:   "npm install -g openclaw@latest",
		printOnly: openclawSetup,
	},
}

// agentDir is the directory kinfer keeps an agent's settings in, for an agent
// whose own it does not touch.
func agentDir(name string) string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".kinfer", "agents", name)
}

// piSetup and piFiles: Pi reads its providers from models.json in its agent
// directory (~/.pi/agent), which PI_CODING_AGENT_DIR moves. kinfer's has kinfer
// as its one provider. Pi's --model takes a ":<thinking>" suffix, so the model
// is picked by a name without colons; the id it sends is kinfer's.
func piSetup(base, model string, window int) (args, env []string) {
	return []string{"--provider", "kinfer", "--model", piName(model)}, []string{"PI_CODING_AGENT_DIR=" + agentDir("pi")}
}

func piName(model string) string { return strings.ReplaceAll(model, ":", "-") }

func piFiles(base, model string, window int) map[string][]byte {
	m := map[string]any{"id": model, "name": piName(model)}
	if window > 0 {
		m["contextWindow"] = window
	}
	b, _ := json.MarshalIndent(map[string]any{"providers": map[string]any{"kinfer": map[string]any{
		"baseUrl": base + "/v1", "api": "openai-completions", "apiKey": "kinfer", "models": []any{m},
	}}}, "", "  ")
	return map[string][]byte{"models.json": b}
}

// hermesSetup and hermesFiles: Hermes keeps its settings in HERMES_HOME
// (~/.hermes); kinfer's config.yaml there names kinfer as its custom,
// OpenAI-compatible provider, and the context it compresses history against.
func hermesSetup(base, model string, window int) (args, env []string) {
	return nil, []string{"HERMES_HOME=" + agentDir("hermes")}
}

func hermesFiles(base, model string, window int) map[string][]byte {
	var b strings.Builder
	b.WriteString("# Written by kinfer launch hermes, again at every launch.\n")
	fmt.Fprintf(&b, "model:\n  provider: \"custom\"\n  base_url: %q\n  default: %q\n", base+"/v1", model)
	if window > 0 {
		fmt.Fprintf(&b, "  context_length: %d\n", window)
	}
	return map[string][]byte{"config.yaml": []byte(b.String())}
}

// openclawSetup: OpenClaw runs as a background gateway of its own, and its
// documentation warns against starting it against another state directory, so
// launch says how to point it at kinfer and runs nothing.
func openclawSetup(base, model string, window int) string {
	m := map[string]any{"id": model, "name": model, "input": []string{"text"}}
	if window > 0 {
		m["contextWindow"] = window
	}
	b, _ := json.MarshalIndent(map[string]any{
		"models": map[string]any{"mode": "merge", "providers": map[string]any{"kinfer": map[string]any{
			"baseUrl": base + "/v1", "apiKey": "kinfer-local", "api": "openai-completions", "models": []any{m},
		}}},
		"agents": map[string]any{"defaults": map[string]any{"model": map[string]any{"primary": "kinfer/" + model}}},
	}, "", "  ")
	return "OpenClaw runs as a background service of its own, which kinfer does not start or change.\n" +
		"To point it at kinfer, merge this into its configuration:\n\n" + string(b) + "\n\n" +
		"or answer its onboarding (openclaw onboard) with Custom provider, base URL " + base + "/v1, model " + model + ".\n"
}

// claudeSetup: Claude Code asks for a model by role as well as by name — a
// small one for titles and summaries, others when a session switches or starts
// a subagent — and a server that knows no claude-haiku answers 404, which
// Claude Code reports as the whole session failing. Every role is the one model.
//
// A model it does not know it takes for 200k tokens of context, and compacts a
// session only near that: told the real window, it summarizes before kinfer
// would have to drop the oldest turns unsaid.
func claudeSetup(base, model string, window int) (args, env []string) {
	env = []string{"ANTHROPIC_BASE_URL=" + base, "ANTHROPIC_AUTH_TOKEN=kinfer"}
	for _, role := range []string{"ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
		"ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"} {
		env = append(env, role+"="+model)
	}
	if window > 0 {
		env = append(env, fmt.Sprintf("CLAUDE_CODE_MAX_CONTEXT_TOKENS=%d", window))
	}
	return nil, env
}

// codexSetup gives Codex a provider of its own on the command line: -c values
// are TOML, so the provider is an inline table, and it speaks the Responses API.
// The window is the model's context, which Codex compacts against.
func codexSetup(base, model string, window int) (args, env []string) {
	args = []string{
		"-c", `model_provider="kinfer"`,
		"-c", fmt.Sprintf(`model_providers.kinfer={name="kinfer",base_url=%q,wire_api="responses"}`, base+"/v1"),
		"-m", model,
	}
	if window > 0 {
		args = append(args, "-c", fmt.Sprintf("model_context_window=%d", window))
	}
	return args, nil
}

func findAgent(name string) (agent, bool) {
	i := slices.IndexFunc(agents, func(a agent) bool { return a.name == name })
	if i < 0 {
		return agent{}, false
	}
	return agents[i], true
}

func agentNames() string {
	var names []string
	for _, a := range agents {
		names = append(names, a.name)
	}
	return strings.Join(names, ", ")
}

// agentEnv is env without the names a.drop lists, with a's own variables set.
func agentEnv(env []string, a agent, set []string) []string {
	out := make([]string, 0, len(env)+len(set))
	for _, kv := range env {
		name, _, _ := strings.Cut(kv, "=")
		if slices.Contains(a.drop, name) || slices.ContainsFunc(set, func(s string) bool { return strings.HasPrefix(s, name+"=") }) {
			continue
		}
		out = append(out, kv)
	}
	return append(out, set...)
}

// hostURL turns -host into a base URL: a bare host:port gets http://, and a
// trailing slash goes.
func hostURL(h string) string {
	h = strings.TrimRight(h, "/")
	if !strings.Contains(h, "://") {
		h = "http://" + h
	}
	return h
}

func cmdLaunch(args []string) error {
	fs := flag.NewFlagSet("launch", flag.ContinueOnError)
	fs.SetOutput(io.Discard) // its errors are said once, by launchFlagErr
	model := fs.String("model", "", "the model the agent uses (default: chosen from a list)")
	host := fs.String("host", os.Getenv("KINFER_HOST"), "a kinfer serving elsewhere, as host:port or a URL (default: the daemon on this machine, started if nothing answers; or $KINFER_HOST)")
	addr := fs.String("addr", defaultAddr, "the daemon's address on this machine")
	keepAlive := fs.Duration("keepalive", 30*time.Minute, "how long a daemon this starts keeps an idle model loaded: an agent is idle while you read")
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "usage: kinfer launch [flags] [%s] [flags] [-- the agent's own arguments]\n\n"+
			"Starts a coding agent talking to kinfer. With no agent, or no -model, it asks.\n\n", strings.ReplaceAll(agentNames(), ", ", "|"))
		fs.PrintDefaults()
	}
	// Flags before the agent's name and after it, up to "--": what follows
	// "--" is the agent's own, as with ollama launch.
	if err := fs.Parse(args); err != nil {
		return launchFlagErr(fs, err)
	}
	name, rest := "", fs.Args()
	if len(rest) > 0 {
		name, rest = rest[0], rest[1:]
		if err := fs.Parse(rest); err != nil {
			return launchFlagErr(fs, err)
		}
		rest = fs.Args()
	}

	in := bufio.NewReader(os.Stdin)
	asking := isTerminal(int(os.Stdin.Fd()))
	if name == "" {
		if !asking {
			return fmt.Errorf("name an agent: kinfer launch <%s>", strings.ReplaceAll(agentNames(), ", ", "|"))
		}
		var err error
		if name, err = pickAgent(in); err != nil {
			return err
		}
	}
	a, ok := findAgent(name)
	if !ok {
		return fmt.Errorf("kinfer launch knows %s, not %q", agentNames(), name)
	}
	bin, err := exec.LookPath(a.bin)
	if err != nil && a.printOnly == nil {
		return fmt.Errorf("%s is not installed here (%s is not on PATH): %s", a.what, a.bin, a.install)
	}

	var base string
	if *host != "" {
		base = hostURL(*host)
		if !daemonAlive(base) {
			return fmt.Errorf("nothing answers at %s", base)
		}
	} else {
		peers, asked := []string{"auto"}, false
		if v := os.Getenv("KINFER_PEERS"); v != "" {
			if peers, err = parsePeers(v); err != nil {
				return fmt.Errorf("KINFER_PEERS: %w", err)
			}
			asked = true
		}
		if base, err = ensureDaemon(*addr, 99, 0, *keepAlive, peers, asked); err != nil {
			return err
		}
	}

	if *model == "" {
		if !asking {
			return fmt.Errorf("name a model: kinfer launch %s -model <name> (kinfer list shows them)", a.name)
		}
		if *model, err = pickModel(in, base); err != nil {
			return err
		}
	}
	if err := checkChatModel(base, *model); err != nil {
		return err
	}
	// Loaded now, the model is ready for the agent's first turn, and the
	// server can say how long a conversation it holds.
	fmt.Fprintf(os.Stderr, "kinfer launch: loading %s at %s\n", *model, base)
	window, err := loadForAgent(base, *model)
	if err != nil {
		return err
	}

	if a.printOnly != nil {
		fmt.Print(a.printOnly(base, *model, window))
		return nil
	}
	if a.minWindow > 0 && window > 0 && window < a.minWindow {
		fmt.Fprintf(os.Stderr, "kinfer launch: %s asks for %d tokens of context and this model holds %d; "+
			"-ctx %d where kinfer serves gives it that\n", a.what, a.minWindow, window, a.minWindow)
	}
	if a.files != nil {
		dir, err := writeAgentFiles(a, base, *model, window)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "kinfer launch: %s's settings for kinfer are in %s; its own are not read or changed\n", a.what, dir)
	}
	pre, set := a.setup(base, *model, window)
	argv := append(append([]string{a.bin}, pre...), rest...)
	held := ""
	if window > 0 {
		held = fmt.Sprintf(", %d tokens of context", window)
		// Claude Code's own instructions and tools are about 25k tokens before
		// the first word of a task, and Codex's 7k.
		if window < 65536 {
			fmt.Fprintf(os.Stderr, "kinfer launch: %d tokens leaves an agent little room once its own prompt is in; "+
				"a larger -ctx where kinfer serves gives it more\n", window)
		}
	}
	fmt.Fprintf(os.Stderr, "kinfer launch: %s with %s at %s%s\n", a.what, *model, base, held)
	return execAgent(bin, argv, agentEnv(os.Environ(), a, set))
}

// writeAgentFiles writes a's files into its directory under ~/.kinfer/agents.
func writeAgentFiles(a agent, base, model string, window int) (string, error) {
	dir := agentDir(a.name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	for name, b := range a.files(base, model, window) {
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			return "", err
		}
	}
	return dir, nil
}

// loadForAgent loads model at base and returns the longest conversation the
// server says it can hold (0 from a kinfer older than v0.1.10, which does not
// say).
func loadForAgent(base, model string) (int, error) {
	body, _ := json.Marshal(map[string]string{"model": model})
	c := &http.Client{Timeout: 15 * time.Minute}
	resp, err := c.Post(base+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return 0, fmt.Errorf("load %s: %w", model, err)
	}
	var out struct {
		Error string `json:"error"`
	}
	_ = json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("load %s: %s", model, cmp.Or(out.Error, resp.Status))
	}
	var ps struct {
		Models []struct {
			ContextMax int `json:"context_max"`
		} `json:"models"`
	}
	if err := getJSON(base+"/api/ps", &ps); err != nil {
		return 0, nil
	}
	// One chat model is loaded at a time; an embedding model beside it says
	// no context.
	for _, m := range ps.Models {
		if m.ContextMax > 0 {
			return m.ContextMax, nil
		}
	}
	return 0, nil
}

// launchFlagErr says where an agent's own flags go when one lands among
// kinfer's, and prints the usage for -h.
func launchFlagErr(fs *flag.FlagSet, err error) error {
	if errors.Is(err, flag.ErrHelp) {
		fs.SetOutput(os.Stderr)
		fs.Usage()
		return nil
	}
	return fmt.Errorf("%w — the agent's own flags go after --: kinfer launch claude -model qwen3.8 -- --continue", err)
}

// pickAgent asks which agent, listing the ones installed first.
func pickAgent(in *bufio.Reader) (string, error) {
	var names, lines []string
	for _, a := range agents {
		if _, err := exec.LookPath(a.bin); err == nil {
			names = append(names, a.name)
			lines = append(lines, fmt.Sprintf("%-8s %s", a.name, a.what))
		}
	}
	if len(names) == 0 {
		return "", fmt.Errorf("none of %s is installed here", agentNames())
	}
	i, err := choose(in, "Agent", lines)
	if err != nil {
		return "", err
	}
	return names[i], nil
}

// pickModel asks which of the models the server can load, leaving out the
// ones that only make vectors.
func pickModel(in *bufio.Reader, base string) (string, error) {
	type listed struct {
		Name string `json:"name"`
		Size int64  `json:"size"`
	}
	var tags, ps struct {
		Models []listed `json:"models"`
	}
	if err := getJSON(base+"/api/tags", &tags); err != nil {
		return "", fmt.Errorf("list the models at %s: %w", base, err)
	}
	// A model already loaded goes first, so Enter takes it rather than the
	// first name in the list — on the box, a model of 249 GB.
	_ = getJSON(base+"/api/ps", &ps)
	loaded := map[string]bool{}
	for _, m := range ps.Models {
		loaded[m.Name] = true
	}
	slices.SortStableFunc(tags.Models, func(a, b listed) int {
		switch {
		case loaded[a.Name] && !loaded[b.Name]:
			return -1
		case loaded[b.Name] && !loaded[a.Name]:
			return 1
		}
		return 0
	})
	var names, lines []string
	for _, m := range tags.Models {
		if caps, err := modelCaps(base, m.Name); err == nil && !canChat(caps) {
			continue
		}
		note := ""
		if loaded[m.Name] {
			note = "  (loaded)"
		}
		names = append(names, m.Name)
		lines = append(lines, fmt.Sprintf("%-48s %6.1f GB%s", m.Name, float64(m.Size)/1e9, note))
	}
	if len(names) == 0 {
		return "", fmt.Errorf("%s has no model to chat with: kinfer pull one", base)
	}
	i, err := choose(in, "Model", lines)
	if err != nil {
		return "", err
	}
	return names[i], nil
}

// choose prints lines numbered from 1 and reads a number; Enter takes the first.
func choose(in *bufio.Reader, what string, lines []string) (int, error) {
	for i, l := range lines {
		fmt.Fprintf(os.Stderr, "  %d) %s\n", i+1, l)
	}
	for {
		fmt.Fprintf(os.Stderr, "%s [1]: ", what)
		line, err := in.ReadString('\n')
		line = strings.TrimSpace(line)
		if line == "" {
			if err != nil {
				return 0, fmt.Errorf("no %s chosen", strings.ToLower(what))
			}
			return 0, nil
		}
		if n, convErr := strconv.Atoi(line); convErr == nil && n >= 1 && n <= len(lines) {
			return n - 1, nil
		}
		if err != nil {
			return 0, fmt.Errorf("no %s chosen", strings.ToLower(what))
		}
		fmt.Fprintf(os.Stderr, "  a number from 1 to %d\n", len(lines))
	}
}

// checkChatModel makes sure the server has model and that it writes text,
// before an agent is started that would only find out on its first turn.
func checkChatModel(base, model string) error {
	caps, err := modelCaps(base, model)
	if err != nil {
		return fmt.Errorf("%s at %s: %w (kinfer list shows the models)", model, base, err)
	}
	if !canChat(caps) {
		return fmt.Errorf("%s is an embedding model: it makes vectors, and an agent needs one that writes", model)
	}
	return nil
}

// canChat: a model that says what it can do, and says completion. One that
// says nothing is given the benefit of the doubt.
func canChat(caps []string) bool {
	return len(caps) == 0 || slices.Contains(caps, "completion")
}

func modelCaps(base, model string) ([]string, error) {
	body, _ := json.Marshal(map[string]string{"model": model})
	c := &http.Client{Timeout: 30 * time.Second}
	resp, err := c.Post(base+"/api/show", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var show struct {
		Capabilities []string `json:"capabilities"`
		Error        string   `json:"error"`
	}
	_ = json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&show)
	if resp.StatusCode != http.StatusOK {
		if show.Error != "" {
			return nil, errors.New(show.Error)
		}
		return nil, fmt.Errorf("the server answered %s", resp.Status)
	}
	return show.Capabilities, nil
}

func getJSON(url string, v any) error {
	c := &http.Client{Timeout: 30 * time.Second}
	resp, err := c.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("the server answered %s", resp.Status)
	}
	return json.NewDecoder(io.LimitReader(resp.Body, 1<<22)).Decode(v)
}
