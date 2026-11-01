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
	"slices"
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
	return daemonHealth(base).alive
}

// health is what a daemon's /health says: that it answers, the workers it
// borrows GPUs from, and its pid — none of either from a daemon older than
// v0.1.3, which did not say.
type health struct {
	alive bool
	peers []string
	pid   int
}

func daemonHealth(base string) health {
	c := &http.Client{Timeout: 700 * time.Millisecond}
	resp, err := c.Get(base + "/health")
	if err != nil {
		return health{}
	}
	defer resp.Body.Close()
	var h struct {
		Peers []string `json:"peers"`
		PID   int      `json:"pid"`
	}
	_ = json.NewDecoder(io.LimitReader(resp.Body, 1<<16)).Decode(&h)
	return health{alive: resp.StatusCode == http.StatusOK, peers: h.Peers, pid: h.PID}
}

// ensureDaemon returns a base URL that is serving, starting a background
// `kinfer serve` if nothing answers yet.
//
// The child is detached (own session, stdio to a log file) so it outlives the
// shell that spawned it — otherwise closing the terminal would take the warm
// model with it, which defeats the point.
//
// A daemon it starts borrows peers as given (-peers auto by default). One
// already running is used as it is, unless the caller asked for peers (asked:
// -peers or KINFER_PEERS) other than the ones it borrows: it would load the
// model somewhere other than asked — slower, or not at all, and nothing would
// say why — so it is refused rather than used.
func ensureDaemon(addr string, ngl, nCtx int, keepAlive time.Duration, peers []string, asked bool) (string, error) {
	base := daemonURL(addr)
	if h := daemonHealth(base); h.alive {
		if asked && !slices.Equal(h.peers, peers) {
			return "", fmt.Errorf("the kinfer daemon on %s %s, and this run asks for %s\n"+
				"       stop it (kinfer stop%s) and run again, or give this run another -addr",
				addr, peersDoing(h.peers), peersWant(peers), stopAddr(addr))
		}
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

	args := []string{"serve",
		"-addr", addr,
		"-ngl", fmt.Sprint(ngl),
		"-ctx", fmt.Sprint(nCtx),
		"-keepalive", keepAlive.String(),
	}
	if len(peers) > 0 {
		args = append(args, "-peers", strings.Join(peers, ","))
	}
	cmd := exec.Command(exe, args...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	detach(cmd)
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start daemon: %w", err)
	}
	// Deliberately not Wait()ed: the daemon outlives this process. Release the
	// child handle so this process does not keep a zombie around.
	_ = cmd.Process.Release()

	fmt.Fprintf(os.Stderr, "starting kinfer daemon on %s, which %s (log: %s)\n", addr, peersDoing(peers), logPath)

	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if daemonAlive(base) {
			return base, nil
		}
		time.Sleep(120 * time.Millisecond)
	}
	return "", fmt.Errorf("daemon did not become ready within 20s — see %s", logPath)
}

// stopAddr is the -addr kinfer stop needs for the daemon on addr: none for the
// default one.
func stopAddr(addr string) string {
	if addr == defaultAddr {
		return ""
	}
	return " -addr " + addr
}

// peersDoing says what a daemon with these peers does with a model.
func peersDoing(peers []string) string {
	switch {
	case len(peers) == 0:
		return "serves on this machine alone"
	case isAuto(peers):
		return "borrows a worker over the Thunderbolt cable for a model that does not fit here"
	}
	return "borrows " + strings.Join(peers, ",")
}

// peersWant says what a run asked of the daemon's peers.
func peersWant(peers []string) string {
	switch {
	case len(peers) == 0:
		return "this machine alone"
	case isAuto(peers):
		return "workers found over the Thunderbolt cable"
	}
	return strings.Join(peers, ",")
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

// chatStats is what the final frame says about a reply's speed, in Ollama's
// fields: the prompt's tokens and the nanoseconds reading them, the reply's
// tokens and the nanoseconds making them, and why it stopped.
type chatStats struct {
	PromptEvalCount    int    `json:"prompt_eval_count"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalCount          int    `json:"eval_count"`
	EvalDuration       int64  `json:"eval_duration"`
	DoneReason         string `json:"done_reason"`
}

// String is the line run prints after a reply, as `ollama run --verbose`
// would and play.sh did; "" when the frame said nothing.
func (s chatStats) String() string {
	var parts []string
	if s.PromptEvalCount > 0 && s.PromptEvalDuration > 0 {
		secs := float64(s.PromptEvalDuration) / 1e9
		parts = append(parts, fmt.Sprintf("prompt %d tokens in %.2f s (%.0f tok/s)", s.PromptEvalCount, secs, float64(s.PromptEvalCount)/secs))
	}
	if s.EvalCount > 0 && s.EvalDuration > 0 {
		parts = append(parts, fmt.Sprintf("reply %d tokens at %.1f tok/s", s.EvalCount, float64(s.EvalCount)/(float64(s.EvalDuration)/1e9)))
	}
	if s.DoneReason == "length" {
		parts = append(parts, "stopped at the token limit")
	}
	if len(parts) == 0 {
		return ""
	}
	return "[" + strings.Join(parts, " · ") + "]"
}

// stderrIsTerminal is whether a person is reading stderr.
func stderrIsTerminal() bool {
	fi, err := os.Stderr.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

const grey, plain = "\033[90m", "\033[0m"

// printStats writes s on its own line to stderr, grey on a terminal, so a
// reply piped from stdout carries none of it.
func printStats(s chatStats) {
	line := s.String()
	if line == "" {
		return
	}
	if stderrIsTerminal() {
		line = grey + line + plain
	}
	fmt.Fprintln(os.Stderr, line)
}

// streamChat posts one conversation and writes each fragment to out as it
// arrives, returning the complete reply and its speed.
//
// It reads the `error` field the server now sends in its final frame. Before
// that field existed, a failed generation arrived as a 200 OK with an empty
// message and this function would have reported success with no text.
func streamChat(ctx context.Context, base, model string, msgs []clientMsg, maxTok int, out io.Writer) (string, chatStats, error) {
	var st chatStats
	body, err := json.Marshal(map[string]any{
		"model":    model,
		"messages": msgs,
		"stream":   true,
		"options":  map[string]any{"num_predict": maxTok},
	})
	if err != nil {
		return "", st, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", st, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", st, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error string `json:"error"`
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		if json.Unmarshal(raw, &e) == nil && e.Error != "" {
			return "", st, errors.New(e.Error)
		}
		return "", st, fmt.Errorf("%s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	// A reasoning model thinks first, and the thinking arrives apart from the
	// reply. On a terminal it is shown in grey on stderr, as it comes, so a
	// long think does not look like a hang; piped, only the reply is written.
	showThinking := stderrIsTerminal()
	thinking := false
	dec := json.NewDecoder(resp.Body)
	var full strings.Builder
	for {
		var frame struct {
			chatStats
			Message struct {
				Content  string `json:"content"`
				Thinking string `json:"thinking"`
			} `json:"message"`
			Done  bool   `json:"done"`
			Error string `json:"error"`
		}
		if err := dec.Decode(&frame); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return full.String(), st, err
		}
		if frame.Error != "" {
			if thinking {
				fmt.Fprintln(os.Stderr, plain)
			}
			return full.String(), st, errors.New(frame.Error)
		}
		if t := frame.Message.Thinking; t != "" && showThinking {
			if !thinking {
				fmt.Fprint(os.Stderr, grey)
				thinking = true
			}
			fmt.Fprint(os.Stderr, t)
		}
		if c := frame.Message.Content; c != "" {
			if thinking {
				fmt.Fprint(os.Stderr, plain+"\n\n")
				thinking = false
			}
			fmt.Fprint(out, c)
			full.WriteString(c)
		}
		if frame.Done {
			if thinking {
				fmt.Fprint(os.Stderr, plain)
			}
			st = frame.chatStats
			break
		}
	}
	return full.String(), st, nil
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
func repl(ctx context.Context, base, model, system string, maxTok int, stats bool) error {
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
		reply, st, err := streamChat(ctx, base, model, msgs, maxTok, os.Stdout)
		fmt.Println()
		if stats && err == nil {
			printStats(st)
		}
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
