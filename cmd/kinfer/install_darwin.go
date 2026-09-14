package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// The job lives in the user's launchd domain: ~/Library/LaunchAgents, loaded
// into gui/<uid>. No sudo, and it starts when this user logs in — which on a
// headless Mac means "after auto-login", the same condition Ollama's agent
// runs under.

func plistPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("locate home directory: %w", err)
	}
	return filepath.Join(home, "Library", "LaunchAgents", serviceLabel+".plist"), nil
}

func launchdDomain() string { return fmt.Sprintf("gui/%d", os.Getuid()) }

func installService(svc service) (string, error) {
	path, err := plistPath()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}

	// Reinstalling is the upgrade path: unload whatever is there, then load the
	// new job. bootout of a job that is not loaded fails, and that is fine.
	_ = launchctl("bootout", launchdDomain()+"/"+serviceLabel)

	if err := os.WriteFile(path, []byte(launchdPlist(svc)), 0o644); err != nil {
		return "", err
	}
	if err := launchctl("bootstrap", launchdDomain(), path); err != nil {
		return "", fmt.Errorf("load %s: %w", path, err)
	}
	return path, nil
}

func uninstallService() (string, error) {
	path, err := plistPath()
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("%s is not installed (no %s)", serviceLabel, path)
	}
	if err := launchctl("bootout", launchdDomain()+"/"+serviceLabel); err != nil {
		// Not loaded — a leftover file from a session that never loaded it.
		// Removing the file is still the right outcome.
		fmt.Fprintf(os.Stderr, "note: %v\n", err)
	}
	if err := os.Remove(path); err != nil {
		return "", err
	}
	return path, nil
}

func launchctl(args ...string) error {
	out, err := exec.Command("launchctl", args...).CombinedOutput()
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if msg == "" {
			msg = err.Error()
		}
		return fmt.Errorf("launchctl %s: %s", strings.Join(args, " "), msg)
	}
	return nil
}
