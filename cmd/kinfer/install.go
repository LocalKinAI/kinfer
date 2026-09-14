// kinfer install: keep `kinfer serve` running the way Ollama's menu-bar app
// keeps its server running — started at login, restarted when it dies.
//
// `kinfer run` already starts a daemon on demand, but that daemon is a plain
// background process: a reboot loses it, and so does a crash. A machine that
// serves a fleet needs the process to come back without anyone noticing it was
// gone, and the operating system's service manager is the thing that does
// that. The launchd job is per-user (a LaunchAgent), which needs no sudo and
// matches what Ollama installs.
//
// The serve flags are recorded in the job, verbatim, so what starts at login is
// exactly what was tested by hand: `kinfer install -ctx 32768 -slots 16` is
// `kinfer serve -ctx 32768 -slots 16` forever.
package main

import (
	"encoding/xml"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// serviceLabel is the launchd job name; the plist file is named after it.
const serviceLabel = "ai.localkin.kinfer"

// service is what the job runs, independent of how the platform records it.
type service struct {
	Exe  string   // absolute path of the kinfer binary
	Args []string // everything after "serve" on the command line
	Log  string   // stdout and stderr, appended
}

func cmdInstall(args []string) error {
	fs := flag.NewFlagSet("install", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: kinfer install [serve flags…]\n"+
			"       the flags are passed to `kinfer serve` unchanged, e.g.\n"+
			"       kinfer install -addr :11500 -ctx 32768 -slots 16")
	}
	// Nothing is parsed here on purpose: every flag belongs to serve, and
	// validating them means running serve's own flag set over them.
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		fs.Usage()
		return nil
	}
	if err := checkServeFlags(args); err != nil {
		return err
	}

	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("locate kinfer binary: %w", err)
	}
	if exe, err = filepath.EvalSymlinks(exe); err != nil {
		return fmt.Errorf("resolve kinfer binary: %w", err)
	}
	logPath, err := daemonLogPath()
	if err != nil {
		return err
	}
	svc := service{Exe: exe, Args: args, Log: logPath}

	where, err := installService(svc)
	if err != nil {
		return err
	}
	fmt.Printf("installed %s\n", where)
	fmt.Printf("  runs: %s serve %s\n", exe, strings.Join(args, " "))
	fmt.Printf("  log : %s\n", logPath)
	fmt.Printf("  starts at login and restarts if it exits; kinfer uninstall removes it\n")
	return nil
}

func cmdUninstall(args []string) error {
	if len(args) > 0 {
		return fmt.Errorf("usage: kinfer uninstall")
	}
	where, err := uninstallService()
	if err != nil {
		return err
	}
	fmt.Printf("removed %s — the server is stopped\n", where)
	return nil
}

// checkServeFlags parses args with serve's flag set so a typo is refused now,
// at the keyboard, rather than by a job that fails to start after every login.
func checkServeFlags(args []string) error {
	fs, _ := serveFlags(flag.ContinueOnError)
	fs.SetOutput(nopWriter{})
	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("serve would refuse these flags: %w", err)
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("serve takes no arguments, and %q would be one", fs.Arg(0))
	}
	return nil
}

type nopWriter struct{}

func (nopWriter) Write(p []byte) (int, error) { return len(p), nil }

// launchdPlist renders a LaunchAgent for svc.
//
// KeepAlive without conditions means launchd restarts the job however it
// ended — a crash, a kill, a clean exit — which is the behaviour of a service
// as opposed to a program. Stopping it for real is `kinfer uninstall`.
func launchdPlist(svc service) string {
	var b strings.Builder
	b.WriteString(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>` + serviceLabel + `</string>
  <key>ProgramArguments</key>
  <array>
`)
	for _, a := range append([]string{svc.Exe, "serve"}, svc.Args...) {
		b.WriteString("    <string>")
		_ = xml.EscapeText(&b, []byte(a))
		b.WriteString("</string>\n")
	}
	b.WriteString(`  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>`)
	_ = xml.EscapeText(&b, []byte(svc.Log))
	b.WriteString(`</string>
  <key>StandardErrorPath</key><string>`)
	_ = xml.EscapeText(&b, []byte(svc.Log))
	b.WriteString(`</string>
</dict>
</plist>
`)
	return b.String()
}
