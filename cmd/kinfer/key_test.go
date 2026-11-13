package main

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"
)

// With KINFER_API_KEY set, kinfer's requests to a server carry it, and the
// agents launch starts are given it; unset, nothing is sent and any word is.
func TestTheKeyGoesWithKinfersRequests(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { got = r.Header.Get("Authorization") }))
	defer srv.Close()

	t.Setenv("KINFER_API_KEY", "")
	if _, err := kinferClient(5 * time.Second).Get(srv.URL); err != nil || got != "" {
		t.Fatalf("no key: %q %v", got, err)
	}
	if _, env := claudeSetup("http://h:1", "m", 0); !slices.Contains(env, "ANTHROPIC_AUTH_TOKEN=kinfer") {
		t.Errorf("claude, no key: %v", env)
	}
	if args, _ := codexSetup("http://h:1", "m", 0); strings.Contains(strings.Join(args, " "), "env_key") {
		t.Errorf("codex, no key: %v", args)
	}

	t.Setenv("KINFER_API_KEY", "s3cret")
	if _, err := kinferClient(5 * time.Second).Get(srv.URL); err != nil || got != "Bearer s3cret" {
		t.Fatalf("with the key: %q %v", got, err)
	}
	if _, env := claudeSetup("http://h:1", "m", 0); !slices.Contains(env, "ANTHROPIC_AUTH_TOKEN=s3cret") {
		t.Errorf("claude: %v", env)
	}
	if args, _ := codexSetup("http://h:1", "m", 0); !strings.Contains(strings.Join(args, " "), `env_key="KINFER_API_KEY"`) {
		t.Errorf("codex: %v", args)
	}
	if !strings.Contains(string(piFiles("http://h:1", "m", 0)["models.json"]), `"apiKey": "s3cret"`) {
		t.Error("pi is not given the key")
	}
	if s := openclawSetup("http://h:1", "m", 0); strings.Contains(s, "s3cret") {
		t.Error("the key is printed for OpenClaw")
	}
}
