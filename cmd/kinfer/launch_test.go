package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// Claude Code is pointed at kinfer for every model it might ask for, and the
// API key that would conflict with the token is not passed on.
func TestClaudeAsksKinferForEveryRole(t *testing.T) {
	a, _ := findAgent("claude")
	args, set := a.setup("http://127.0.0.1:11500", "qwen3.8", 131072)
	if len(args) != 0 {
		t.Errorf("args = %q, want none", args)
	}
	env := agentEnv([]string{"PATH=/bin", "ANTHROPIC_API_KEY=sk-x", "ANTHROPIC_MODEL=claude-opus"}, a, set)
	for _, want := range []string{
		"PATH=/bin",
		"ANTHROPIC_BASE_URL=http://127.0.0.1:11500",
		"ANTHROPIC_AUTH_TOKEN=kinfer",
		"ANTHROPIC_MODEL=qwen3.8",
		"ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8",
		"ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8",
		"ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8",
		"ANTHROPIC_SMALL_FAST_MODEL=qwen3.8",
		"CLAUDE_CODE_SUBAGENT_MODEL=qwen3.8",
		"CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072",
	} {
		if !slices.Contains(env, want) {
			t.Errorf("env lacks %s: %q", want, env)
		}
	}
	for _, kv := range env {
		if strings.HasPrefix(kv, "ANTHROPIC_API_KEY=") || kv == "ANTHROPIC_MODEL=claude-opus" {
			t.Errorf("env keeps %s", kv)
		}
	}
}

// Codex gets a provider of its own on the command line, speaking the Responses
// API at /v1, and nothing in its environment.
func TestCodexGetsAProviderOnTheCommandLine(t *testing.T) {
	a, _ := findAgent("codex")
	args, set := a.setup("http://192.168.0.21:11590", "ornith-1.5:35b", 262144)
	want := []string{
		"-c", `model_provider="kinfer"`,
		"-c", `model_providers.kinfer={name="kinfer",base_url="http://192.168.0.21:11590/v1",wire_api="responses"}`,
		"-m", "ornith-1.5:35b",
		"-c", "model_context_window=262144",
	}
	if !slices.Equal(args, want) || len(set) != 0 {
		t.Errorf("args = %q, env = %q; want %q and none", args, set, want)
	}
}

func TestHostURL(t *testing.T) {
	for in, want := range map[string]string{
		"192.168.0.21:11590":         "http://192.168.0.21:11590",
		"http://box:11590/":          "http://box:11590",
		"https://kinfer.example.com": "https://kinfer.example.com",
	} {
		if got := hostURL(in); got != want {
			t.Errorf("hostURL(%q) = %q, want %q", in, got, want)
		}
	}
}

// A model that does not say what it can do is offered; one that only embeds
// is not.
func TestCanChat(t *testing.T) {
	for _, c := range []struct {
		caps []string
		want bool
	}{{nil, true}, {[]string{"completion", "vision"}, true}, {[]string{"embedding"}, false}} {
		if got := canChat(c.caps); got != c.want {
			t.Errorf("canChat(%q) = %v, want %v", c.caps, got, c.want)
		}
	}
}

// Pi is pointed at a directory of kinfer's, whose models.json has kinfer as
// its one provider, and the model is picked by a name without colons, since
// Pi's --model reads ":<suffix>" as a thinking level.
func TestPiGetsADirectoryOfItsOwn(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	a, _ := findAgent("pi")
	args, set := a.setup("http://127.0.0.1:11500", "qwen3.8:27b-mlx", 32768)
	dir := filepath.Join(home, ".kinfer", "agents", "pi")
	if !slices.Equal(args, []string{"--provider", "kinfer", "--model", "qwen3.8-27b-mlx"}) ||
		!slices.Equal(set, []string{"PI_CODING_AGENT_DIR=" + dir}) {
		t.Errorf("args %q, env %q", args, set)
	}
	if got, err := writeAgentFiles(a, "http://127.0.0.1:11500", "qwen3.8:27b-mlx", 32768); err != nil || got != dir {
		t.Fatalf("written to %q, %v; want %s", got, err, dir)
	}
	var cfg struct {
		Providers map[string]struct {
			BaseURL string `json:"baseUrl"`
			API     string `json:"api"`
			Models  []struct {
				ID            string `json:"id"`
				Name          string `json:"name"`
				ContextWindow int    `json:"contextWindow"`
			} `json:"models"`
		} `json:"providers"`
	}
	b, err := os.ReadFile(filepath.Join(dir, "models.json"))
	if err != nil || json.Unmarshal(b, &cfg) != nil {
		t.Fatalf("models.json: %s, %v", b, err)
	}
	p := cfg.Providers["kinfer"]
	if len(cfg.Providers) != 1 || p.BaseURL != "http://127.0.0.1:11500/v1" || p.API != "openai-completions" || len(p.Models) != 1 ||
		p.Models[0].ID != "qwen3.8:27b-mlx" || p.Models[0].Name != "qwen3.8-27b-mlx" || p.Models[0].ContextWindow != 32768 {
		t.Errorf("models.json: %s", b)
	}
}

// Hermes gets a home of kinfer's, whose config.yaml names kinfer as its custom
// provider and the context it compresses against.
func TestHermesGetsAHomeOfItsOwn(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	a, _ := findAgent("hermes")
	if _, set := a.setup("http://192.168.0.21:11590", "ornith-1.5:35b", 65536); !slices.Equal(set,
		[]string{"HERMES_HOME=" + filepath.Join(home, ".kinfer", "agents", "hermes")}) {
		t.Errorf("env %q", set)
	}
	got := string(a.files("http://192.168.0.21:11590", "ornith-1.5:35b", 65536)["config.yaml"])
	for _, want := range []string{"model:\n", `  provider: "custom"`, `  base_url: "http://192.168.0.21:11590/v1"`,
		`  default: "ornith-1.5:35b"`, "  context_length: 65536"} {
		if !strings.Contains(got, want) {
			t.Errorf("config.yaml lacks %q:\n%s", want, got)
		}
	}
}

// OpenClaw is told, not run: the snippet has kinfer as a provider and the
// model as its agents' primary.
func TestOpenClawIsOnlyToldHow(t *testing.T) {
	a, _ := findAgent("openclaw")
	if a.printOnly == nil || a.setup != nil || a.files != nil {
		t.Fatal("openclaw should only print")
	}
	got := a.printOnly("http://127.0.0.1:11500", "qwen3.8-27b-q4_k_m", 32768)
	for _, want := range []string{`"baseUrl": "http://127.0.0.1:11500/v1"`, `"primary": "kinfer/qwen3.8-27b-q4_k_m"`, `"contextWindow": 32768`} {
		if !strings.Contains(got, want) {
			t.Errorf("lacks %s:\n%s", want, got)
		}
	}
}
