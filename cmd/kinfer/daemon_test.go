package main

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestParsePeers(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"", nil},
		{"none", nil},
		{"auto", []string{"auto"}},
		{"169.254.154.219:50052", []string{"169.254.154.219:50052"}},
		{" a:1 , b:2, ", []string{"a:1", "b:2"}},
	} {
		got, err := parsePeers(c.in)
		if err != nil || !slices.Equal(got, c.want) {
			t.Errorf("parsePeers(%q) = %q, %v; want %q", c.in, got, err, c.want)
		}
	}
	for _, bad := range []string{"169.254.154.219", "auto,a:1"} {
		if _, err := parsePeers(bad); err == nil {
			t.Errorf("parsePeers(%q) was taken", bad)
		}
	}
}

// A daemon already running is used as it is, unless the run asked for peers it
// does not borrow: then the model would load somewhere other than asked, and
// the run says so instead. The default (auto, not asked) takes any daemon.
func TestEnsureDaemonChecksTheRunningDaemonsPeers(t *testing.T) {
	daemon := func(body string) string {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte(body))
		}))
		t.Cleanup(srv.Close)
		return strings.TrimPrefix(srv.URL, "http://")
	}
	alone := daemon(`{"status":"ok"}`)
	auto := daemon(`{"status":"ok","peers":["auto"]}`)
	named := daemon(`{"status":"ok","peers":["a:1"]}`)

	for _, c := range []struct {
		name, addr string
		peers      []string
		asked      bool
		refused    string
	}{
		{"alone, the default", alone, []string{"auto"}, false, ""},
		{"named, the default", named, []string{"auto"}, false, ""},
		{"auto, asked for auto", auto, []string{"auto"}, true, ""},
		{"alone, asked for auto", alone, []string{"auto"}, true, "serves on this machine alone"},
		{"alone, asked for a worker", alone, []string{"a:1"}, true, "asks for a:1"},
		{"named, the same worker", named, []string{"a:1"}, true, ""},
		{"named, another worker", named, []string{"b:2"}, true, "borrows a:1"},
		{"named, asked for none", named, nil, true, "this machine alone"},
	} {
		_, err := ensureDaemon(c.addr, 99, 0, time.Minute, c.peers, c.asked)
		switch {
		case c.refused == "" && err != nil:
			t.Errorf("%s: refused: %v", c.name, err)
		case c.refused != "" && (err == nil || !strings.Contains(err.Error(), c.refused)):
			t.Errorf("%s: got %v, want a refusal saying %q", c.name, err, c.refused)
		}
	}
}

func TestChatStatsLine(t *testing.T) {
	s := chatStats{PromptEvalCount: 7016, PromptEvalDuration: 5_500_000_000, EvalCount: 200, EvalDuration: 4_500_000_000, DoneReason: "length"}
	want := "[prompt 7016 tokens in 5.50 s (1276 tok/s) · reply 200 tokens at 44.4 tok/s · stopped at the token limit]"
	if got := s.String(); got != want {
		t.Errorf("got  %s\nwant %s", got, want)
	}
	if got := (chatStats{}).String(); got != "" {
		t.Errorf("an empty frame printed %q", got)
	}
}

func TestStopAddr(t *testing.T) {
	if got := stopAddr(defaultAddr); got != "" {
		t.Errorf("stopAddr(default) = %q, want none", got)
	}
	if got := stopAddr("127.0.0.1:11510"); got != " -addr 127.0.0.1:11510" {
		t.Errorf("stopAddr = %q", got)
	}
}
