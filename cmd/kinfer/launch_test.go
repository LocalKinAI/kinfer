package main

import (
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
