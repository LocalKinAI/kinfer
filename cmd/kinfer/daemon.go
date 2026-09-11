// Daemon plumbing: kinfer's CLI never loads a model itself.
//
// Ollama feels instant because `ollama run` does not load anything — it talks
// to a daemon that already has the model resident, and starts that daemon
// transparently the first time. kinfer does the same: a model is loaded once
// and stays warm between invocations, so the second `kinfer run` pays nothing.
//
// Measured on an M4: a cold `kinfer run` was 0.85s and a fresh process 0.29s
// even before this — Metal's shader cache makes startup cheap. What it did NOT
// make cheap is loading the weights again, which costs seconds on a 7B. That is
// the tax this file removes.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const defaultAddr = "127.0.0.1:11500"

// daemonURL turns an -addr flag into a base URL. ":11500" is a valid listen
// address but not a valid URL, so the host has to be filled in.
func daemonURL(addr string) string {
	if strings.HasPrefix(addr, ":") {
		addr = "127.0.0.1" + addr
	}
	return "http://" + addr
}

// daemonAlive reports whether something answers /health at base.
//
// The timeout is short on purpose: this runs on every `kinfer run`, and the
// common case is either an instant answer or an instant connection refused.
func daemonAlive(base string) bool {
	c := &http.Client{Timeout: 700 * time.Millisecond}
	resp, err := c.Get(base + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == http.StatusOK
}

// ensureDaemon returns a base URL that is serving, starting a background
// `kinfer serve` if nothing answers yet.
//
// The child is detached (own session, stdio to a log file) so it outlives the
// shell that spawned it — otherwise closing the terminal would take the warm
// model with it, which defeats the point.
func ensureDaemon(addr string, ngl, nCtx int, keepAlive time.Duration) (string, error) {
	base := daemonURL(addr)
	if daemonAlive(base) {
		return base, nil
	}

	exe, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("locate kinfer binary: %w", err)
	}

	logPath, err := daemonLogPath()
	if err != nil {
		return "", err
	}
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return "", fmt.Errorf("open %s: %w", logPath, err)
	}
	defer logFile.Close()

	cmd := exec.Command(exe, "serve",
		"-addr", addr,
		"-ngl", fmt.Sprint(ngl),
		"-ctx", fmt.Sprint(nCtx),
		"-keepalive", keepAlive.String(),
	)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	detach(cmd)
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start daemon: %w", err)
	}
	// Deliberately not Wait()ed: the daemon outlives this process. Release the
	// child handle so this process does not keep a zombie around.
	_ = cmd.Process.Release()

	fmt.Fprintf(os.Stderr, "starting kinfer daemon on %s (log: %s)\n", addr, logPath)

	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if daemonAlive(base) {
			return base, nil
		}
		time.Sleep(120 * time.Millisecond)
	}
	return "", fmt.Errorf("daemon did not become ready within 20s — see %s", logPath)
}

func daemonLogPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("locate home directory: %w", err)
	}
	dir := filepath.Join(home, ".kinfer")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create %s: %w", dir, err)
	}
	return filepath.Join(dir, "serve.log"), nil
}

// ─── chat client ─────────────────────────────────────────────────────────────

type clientMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// streamChat posts one conversation and writes each fragment to out as it
// arrives, returning the complete reply.
//
// It reads the `error` field the server now sends in its final frame. Before
// that field existed, a failed generation arrived as a 200 OK with an empty
// message and this function would have reported success with no text.
func streamChat(ctx context.Context, base, model string, msgs []clientMsg, maxTok int, out io.Writer) (string, error) {
	body, err := json.Marshal(map[string]any{
		"model":    model,
		"messages": msgs,
		"stream":   true,
		"options":  map[string]any{"num_predict": maxTok},
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error string `json:"error"`
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		if json.Unmarshal(raw, &e) == nil && e.Error != "" {
			return "", errors.New(e.Error)
		}
		return "", fmt.Errorf("%s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	dec := json.NewDecoder(resp.Body)
	var full strings.Builder
	for {
		var frame struct {
			Message clientMsg `json:"message"`
			Done    bool      `json:"done"`
			Error   string    `json:"error"`
		}
		if err := dec.Decode(&frame); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return full.String(), err
		}
		if frame.Error != "" {
			return full.String(), errors.New(frame.Error)
		}
		if frame.Message.Content != "" {
			fmt.Fprint(out, frame.Message.Content)
			full.WriteString(frame.Message.Content)
		}
		if frame.Done {
			break
		}
	}
	return full.String(), nil
}

// ─── REPL ────────────────────────────────────────────────────────────────────

const replHelp = `  /bye, /exit    leave (the daemon keeps the model warm)
  /clear         forget this conversation
  /?, /help      this message`

// repl is an interactive conversation against a warm daemon.
//
// History is kept client-side and resent whole each turn, because the engine
// clears its KV cache between turns. That is correct but not free — it is the
// re-prefill cost a prefix cache will remove.
func repl(ctx context.Context, base, model, system string, maxTok int) error {
	fmt.Printf("kinfer · %s\n", model)
	fmt.Printf("%s\n", replHelp)

	var msgs []clientMsg
	reset := func() {
		msgs = nil
		if system != "" {
			msgs = append(msgs, clientMsg{Role: "system", Content: system})
		}
	}
	reset()

	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64<<10), 4<<20) // a pasted soul is longer than the default 64K

	for {
		fmt.Print("\n> ")
		if !sc.Scan() {
			fmt.Println()
			return sc.Err() // nil on a clean EOF (^D)
		}
		line := strings.TrimSpace(sc.Text())
		switch {
		case line == "":
			continue
		case line == "/bye", line == "/exit", line == "/quit":
			return nil
		case line == "/clear":
			reset()
			fmt.Println("(conversation cleared)")
			continue
		case line == "/?", line == "/help":
			fmt.Println(replHelp)
			continue
		}

		msgs = append(msgs, clientMsg{Role: "user", Content: line})
		fmt.Println()
		reply, err := streamChat(ctx, base, model, msgs, maxTok, os.Stdout)
		fmt.Println()
		if err != nil {
			fmt.Fprintf(os.Stderr, "kinfer: %v\n", err)
			// Drop the turn that failed rather than carrying a question the
			// model never answered into the next prompt.
			msgs = msgs[:len(msgs)-1]
			continue
		}
		msgs = append(msgs, clientMsg{Role: "assistant", Content: reply})
	}
}
