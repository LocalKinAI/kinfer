package main

import (
	"strings"
	"testing"
)

// A launchd job that fails to start does so silently, at the next login, into
// a log nobody is reading. So what install records has to be right the first
// time: the arguments verbatim, the binary by absolute path, a restart on any
// exit, and no mistyped serve flag getting as far as the plist.
func TestLaunchdPlistRecordsTheServeCommand(t *testing.T) {
	svc := service{
		Exe:  "/Users/x/bin/kinfer",
		Args: []string{"-addr", ":11590", "-ctx", "32768", "-slots", "16", "-keepalive", "0"},
		Log:  "/Users/x/.kinfer/serve.log",
	}
	got := launchdPlist(svc)

	for _, want := range []string{
		"<string>" + serviceLabel + "</string>",
		"<string>/Users/x/bin/kinfer</string>",
		"<string>serve</string>",
		"<string>-ctx</string>",
		"<string>32768</string>",
		"<string>-keepalive</string>",
		"<string>0</string>",
		"<key>RunAtLoad</key><true/>",
		"<key>KeepAlive</key><true/>",
		"<string>/Users/x/.kinfer/serve.log</string>",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("plist lacks %s\n%s", want, got)
		}
	}
	// The arguments are an ordered list; "serve" has to come before its flags.
	if strings.Index(got, "<string>serve</string>") > strings.Index(got, "<string>-addr</string>") {
		t.Errorf("serve must precede its flags\n%s", got)
	}
}

func TestLaunchdPlistEscapesXML(t *testing.T) {
	got := launchdPlist(service{Exe: "/a&b/kinfer", Log: "/tmp/<x>.log"})
	if !strings.Contains(got, "/a&amp;b/kinfer") || !strings.Contains(got, "/tmp/&lt;x&gt;.log") {
		t.Errorf("paths with XML-special characters must be escaped\n%s", got)
	}
}

func TestInstallRefusesFlagsServeWouldRefuse(t *testing.T) {
	if err := checkServeFlags([]string{"-addr", ":11590", "-ctx", "32768"}); err != nil {
		t.Errorf("valid serve flags refused: %v", err)
	}
	for _, bad := range [][]string{
		{"-ctx"},                   // missing value
		{"-slot", "16"},            // misspelled
		{"-max-gen", "-1"},         // the unparseable duration that cost an afternoon
		{"qwen"},                   // serve takes no model argument
		{"-ctx", "32768", "extra"}, // trailing positional
	} {
		if err := checkServeFlags(bad); err == nil {
			t.Errorf("checkServeFlags(%q) = nil, want an error — launchd would retry this forever", bad)
		}
	}
}
