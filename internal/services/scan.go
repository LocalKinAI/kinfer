package services

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Found is one program Scan saw listening.
type Found struct {
	Service
	PID       int
	Footprint uint64 // bytes, the process and its children
	// Why is set when the front could not run it itself: watched only.
	Why string
}

// Scan lists the programs of this user listening on TCP ports, other than
// kinfer itself and the system's and applications' own, each as a service
// entry: its command with its port and host as {port} and {host}, its
// directory, the environment it was started with less the session's own, and
// the launchd job it belongs to.
func Scan() ([]Found, error) {
	ls, err := listening()
	if err != nil {
		return nil, err
	}
	jobs := launchdJobs()
	tree := children()
	self := os.Getpid()
	var out []Found
	names := map[string]int{}
	pids := make([]int, 0, len(ls))
	for pid := range ls {
		pids = append(pids, pid)
	}
	sort.Ints(pids)
	for _, pid := range pids {
		if pid == self {
			continue
		}
		argv, env, err := procArgs(pid)
		if err != nil || len(argv) == 0 || skipped(argv) {
			continue
		}
		addr := ls[pid][0]
		f := Found{PID: pid, Service: Service{Addr: addr, Dir: cwd(pid)}}
		f.Command, f.Env = template(argv, env, addr)
		f.Launchd = jobs[pid]
		f.Footprint = treeFootprint(pid, tree)
		f.Need = f.Footprint
		f.Name = unique(nameFor(argv, f.Dir), names)
		f.Log = filepath.Join(filepath.Dir(DefaultPath()), "services", f.Name+".log")
		switch {
		case f.Launchd != "":
			f.Why = "launchd's job " + f.Launchd + " keeps it running"
		case !f.Runnable():
			f.Why = "its port is not on its command line or in its environment, so it cannot be moved to another"
		}
		if len(ls[pid]) > 1 {
			f.Why = strings.TrimPrefix(f.Why+"; it listens on "+strings.Join(ls[pid], ", ")+" — only the first is taken in", "; ")
		}
		out = append(out, f)
	}
	return out, nil
}

// listening maps each pid of this user to the addresses it listens on.
func listening() (map[int][]string, error) {
	b, err := exec.Command("lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-Fpn").Output()
	if err != nil && len(b) == 0 {
		return nil, fmt.Errorf("lsof: %w", err)
	}
	out := map[int][]string{}
	pid := 0
	sc := bufio.NewScanner(bytes.NewReader(b))
	for sc.Scan() {
		l := sc.Text()
		if l == "" {
			continue
		}
		switch l[0] {
		case 'p':
			pid, _ = strconv.Atoi(l[1:])
		case 'n':
			a := strings.Replace(l[1:], "*:", "0.0.0.0:", 1)
			if strings.HasPrefix(a, "[") { // IPv6 beside an IPv4 one: the v4 is kept
				continue
			}
			if !contains(out[pid], a) {
				out[pid] = append(out[pid], a)
			}
		}
	}
	return out, nil
}

func contains(xs []string, x string) bool {
	for _, s := range xs {
		if s == x {
			return true
		}
	}
	return false
}

// skipped are kinfer itself, its runners, and the system's and installed
// applications' own servers: not programs one would hand to kinfer.
func skipped(argv []string) bool {
	exe := argv[0]
	for _, p := range []string{"/System/", "/usr/libexec/", "/usr/sbin/", "/sbin/", "/Library/Apple/", "/Library/PrivilegedHelperTools/"} {
		if strings.HasPrefix(exe, p) {
			return true
		}
	}
	// An application's own server (Ollama, an IDE's, a VPN's), not a
	// program run on its own — but Python's framework runs itself from a
	// Python.app, and is.
	if strings.Contains(exe, ".app/Contents/") && !strings.Contains(exe, "/Python.app/") {
		return true
	}
	base := filepath.Base(exe)
	return strings.HasPrefix(base, "kinfer") || base == "ollama" || strings.Contains(strings.Join(argv, " "), "-runners=false")
}

func cwd(pid int) string {
	b, _ := exec.Command("lsof", "-a", "-d", "cwd", "-p", strconv.Itoa(pid), "-Fn").Output()
	for _, l := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(l, "n") {
			return l[1:]
		}
	}
	return ""
}

// launchdJobs maps the pids launchd runs for this user to their labels.
func launchdJobs() map[int]string {
	b, _ := exec.Command("launchctl", "list").Output()
	out := map[int]string{}
	for _, l := range strings.Split(string(b), "\n") {
		f := strings.Fields(l)
		if len(f) == 3 {
			if pid, err := strconv.Atoi(f[0]); err == nil {
				out[pid] = f[2]
			}
		}
	}
	return out
}

// children maps each pid to its children's.
func children() map[int][]int {
	b, _ := exec.Command("ps", "-A", "-o", "pid=,ppid=").Output()
	out := map[int][]int{}
	for _, l := range strings.Split(string(b), "\n") {
		f := strings.Fields(l)
		if len(f) == 2 {
			pid, _ := strconv.Atoi(f[0])
			ppid, _ := strconv.Atoi(f[1])
			out[ppid] = append(out[ppid], pid)
		}
	}
	return out
}

func treeFootprint(pid int, tree map[int][]int) uint64 {
	n := footprint(pid)
	for _, c := range tree[pid] {
		n += treeFootprint(c, tree)
	}
	return n
}

// The environment a shell or launchd gives every process: not what the
// program was started with on purpose.
var sessionEnv = regexp.MustCompile(`^(SSH_.*|TERM.*|SHELL|PWD|OLDPWD|SHLVL|_|LOGNAME|USER|HOME|TMPDIR|LANG|LC_.*|__CF.*|XPC_.*|COMMAND_MODE|LaunchInstanceID|SECURITYSESSIONID|MallocNanoZone|OSLogRateLimit|TERM_PROGRAM.*|COLORTERM|ITERM.*|VSCODE.*|MOTD_SHOWN|MAIL|DISPLAY|SSH_AUTH_SOCK|__CFBundleIdentifier)$`)

// portArgs are the flags a program's own port follows; hostArgs its host's.
var (
	portArgs = map[string]bool{"--port": true, "-p": true, "--listen-port": true, "--http-port": true, "--server-port": true}
	hostArgs = map[string]bool{"--host": true, "--listen": true, "--bind": true, "--address": true, "--ip": true, "-H": true}
)

// template writes argv and env with the listening address's port and host
// as {port} and {host}: in --port 8188 or --port=8188, after --host or
// --listen, and in an environment value equal to the port (PORT=8105).
func template(argv, env []string, addr string) ([]string, map[string]string) {
	s := Service{Addr: addr}
	port, host := strconv.Itoa(s.Port()), s.Host()
	out := append([]string(nil), argv...)
	for i := 0; i < len(out); i++ {
		a := out[i]
		if k, v, ok := strings.Cut(a, "="); ok && strings.HasPrefix(k, "-") {
			switch {
			case portArgs[k] && v == port:
				out[i] = k + "={port}"
			case hostArgs[k] && (v == host || v == "0.0.0.0" || v == "127.0.0.1" || v == "localhost"):
				out[i] = k + "={host}"
			}
			continue
		}
		if i+1 < len(out) {
			switch {
			case portArgs[a] && out[i+1] == port:
				out[i+1] = "{port}"
				i++
			case hostArgs[a] && (out[i+1] == host || out[i+1] == "0.0.0.0" || out[i+1] == "127.0.0.1" || out[i+1] == "localhost"):
				out[i+1] = "{host}"
				i++
			}
		}
	}
	// No flag names it: a word that is the port and nothing else is it —
	// python -m http.server 8000 takes its port so.
	named := false
	for _, a := range out {
		named = named || strings.Contains(a, "{port}")
	}
	for _, kv := range env {
		named = named || strings.HasSuffix(kv, "="+port)
	}
	if !named {
		for i := 1; i < len(out); i++ {
			if out[i] == port {
				out[i] = "{port}"
				break
			}
		}
	}
	e := map[string]string{}
	for _, kv := range env {
		k, v, ok := strings.Cut(kv, "=")
		if !ok || sessionEnv.MatchString(k) {
			continue
		}
		if v == port {
			v = "{port}"
		}
		e[k] = v
	}
	return out, e
}

// nameFor is a short name: the folder it runs in, or, run from the home
// folder, its script.
func nameFor(argv []string, dir string) string {
	home, _ := os.UserHomeDir()
	if u, err := user.Current(); err == nil { // the user's, whatever HOME says
		home = u.HomeDir
	}
	n := filepath.Base(dir)
	if dir == "" || dir == home || dir == "/" {
		n = filepath.Base(argv[0])
		for _, a := range argv[1:] {
			if !strings.HasPrefix(a, "-") && (strings.HasSuffix(a, ".py") || strings.Contains(a, "/")) {
				n = strings.TrimSuffix(filepath.Base(a), ".py")
				break
			}
		}
	}
	n = strings.ToLower(regexp.MustCompile(`[^A-Za-z0-9]+`).ReplaceAllString(n, "-"))
	return strings.Trim(n, "-")
}

func unique(n string, seen map[string]int) string {
	if n == "" {
		n = "service"
	}
	seen[n]++
	if seen[n] > 1 {
		return fmt.Sprintf("%s-%d", n, seen[n])
	}
	return n
}
