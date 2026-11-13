//go:build !windows

package services

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// macOnly skips a test of what reads another process's command line: kinfer
// does that on macOS only (proc_darwin.go), so elsewhere a service is watched
// and not taken over, and nothing a crashed server left is ended.
func macOnly(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("taking over a program reads its command line, which kinfer does on macOS only")
	}
}

func freeAddr(t *testing.T) string {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().String()
}

func fake(t *testing.T, addr string, adopted bool, idle time.Duration) (string, Service) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3, the stand-in service, is not here")
	}
	dir := t.TempDir()
	svc := Service{Name: "fake", Addr: addr, Command: []string{"python3", "-m", "http.server", "{port}", "--bind", "{host}"},
		Dir: dir, Adopted: adopted, Idle: Duration(idle), Log: filepath.Join(dir, "fake.log")}
	path := filepath.Join(dir, "services.json")
	if err := Save(path, File{Services: []Service{svc}}); err != nil {
		t.Fatal(err)
	}
	return path, svc
}

func get(t *testing.T, addr string) int {
	c := http.Client{Timeout: 30 * time.Second}
	r, err := c.Get("http://" + addr + "/")
	if err != nil {
		t.Fatalf("GET %s: %v", addr, err)
	}
	io.Copy(io.Discard, r.Body)
	r.Body.Close()
	return r.StatusCode
}

func waitFor(t *testing.T, what string, ok func() bool) {
	for i := 0; i < 150; i++ {
		if ok() {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("%s did not happen", what)
}

func state(m *Manager) Stat {
	for _, s := range m.Stats() {
		if s.Name == "fake" {
			return s
		}
	}
	return Stat{}
}

// Adopted, a service is started by the first request to its address, its
// answer passed back; idle, it stops; asked for again, it starts again; and
// requests that arrive together while it is stopped start it once.
func TestAdoptedServiceStartsOnDemandAndStopsIdle(t *testing.T) {
	addr := freeAddr(t)
	path, svc := fake(t, addr, true, time.Second)
	m := NewManager(path)
	if err := m.Start(); err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	waitFor(t, "the address taken", func() bool {
		c, err := net.Dial("tcp", addr)
		if err == nil {
			c.Close()
		}
		return err == nil
	})
	if st := state(m); st.State != "stopped" {
		t.Fatalf("before any request: %+v", st)
	}
	if code := get(t, addr); code != 200 {
		t.Fatalf("first request: %d", code)
	}
	first := state(m)
	if first.State != "running" || first.PID == 0 {
		t.Fatalf("after the first request: %+v", first)
	}
	waitFor(t, "the idle stop", func() bool { return state(m).State == "stopped" })

	var wg sync.WaitGroup
	codes := make([]int, 4)
	for i := range codes {
		wg.Add(1)
		go func() { defer wg.Done(); codes[i] = get(t, addr) }()
	}
	wg.Wait()
	second := state(m)
	for i, c := range codes {
		if c != 200 {
			t.Errorf("request %d together: %d", i, c)
		}
	}
	if second.PID == 0 || second.PID == first.PID {
		t.Errorf("started again as pid %d (first %d)", second.PID, first.PID)
	}
	// Started twice in all: by the first request, and once more by the four.
	logged, _ := os.ReadFile(svc.Log)
	if n := strings.Count(string(logged), "==== kinfer started fake"); n != 2 {
		t.Errorf("started %d times, want 2: once, then once for the four requests together", n)
	}
}

// A watched service adopted: its own process, and only it, is ended, and the
// manager answers on its address in its place.
func TestAdoptEndsTheServicesOwnProcess(t *testing.T) {
	macOnly(t)
	addr := freeAddr(t)
	path, svc := fake(t, addr, false, time.Minute)
	_, port, _ := net.SplitHostPort(addr)
	own := exec.Command("python3", "-m", "http.server", port, "--bind", "127.0.0.1")
	own.Dir = svc.Dir
	if err := own.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() { own.Wait(); close(done) }()
	defer syscall.Kill(own.Process.Pid, syscall.SIGKILL)
	waitFor(t, "the service's own process listening", func() bool { return listenerPID(svc.Port()) == own.Process.Pid })
	// The entry as Scan writes it, from the command that is running: Python
	// on macOS runs itself again from another path, so not python3 as typed.
	argv, _, err := procArgs(own.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	for i, a := range argv {
		if a == port {
			argv[i] = "{port}"
		}
		if i > 0 && argv[i-1] == "--bind" {
			argv[i] = "{host}"
		}
	}
	svc.Command = argv
	if err := Save(path, File{Services: []Service{svc}}); err != nil {
		t.Fatal(err)
	}

	m := NewManager(path)
	if err := m.Start(); err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if st := state(m); st.State != "up" || st.PID != own.Process.Pid {
		t.Fatalf("watched: %+v", st)
	}
	if err := m.Adopt("fake", 0); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("its own process did not end")
	}
	waitFor(t, "the manager on its address", func() bool {
		c, err := net.Dial("tcp", addr)
		if err == nil {
			c.Close()
		}
		return err == nil
	})
	if code := get(t, addr); code != 200 {
		t.Fatalf("after adopting: %d", code)
	}
	if st := state(m); !st.Adopted || st.State != "running" || st.PID == own.Process.Pid {
		t.Fatalf("adopted: %+v", st)
	}

	pid, err := m.Release("fake")
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Kill(pid, syscall.SIGKILL)
	waitFor(t, "it on its own again", func() bool { return listenerPID(svc.Port()) == pid })
	if code := get(t, addr); code != 200 {
		t.Fatalf("released: %d", code)
	}
}

func TestAdoptRefusesAnotherProgramOnTheAddress(t *testing.T) {
	addr := freeAddr(t)
	path, _ := fake(t, addr, false, time.Minute)
	l, err := net.Listen("tcp", addr) // this test, not the service's command
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	m := NewManager(path)
	m.Start()
	defer m.Close()
	if err := m.Adopt("fake", 0); err == nil {
		t.Fatal("adopted an address another program holds")
	}
}

func TestTemplateAndMatch(t *testing.T) {
	argv := []string{"venv/bin/python", "main.py", "--listen", "0.0.0.0", "--port", "8188"}
	cmd, env := template(argv, []string{"PORT=8188", "SHELL=/bin/zsh", "FACE_ENGINE=quicktalk"}, "0.0.0.0:8188")
	if want := []string{"venv/bin/python", "main.py", "--listen", "{host}", "--port", "{port}"}; strconv.Quote(join(cmd)) != strconv.Quote(join(want)) {
		t.Errorf("command %q", cmd)
	}
	if env["PORT"] != "{port}" || env["FACE_ENGINE"] != "quicktalk" || env["SHELL"] != "" {
		t.Errorf("env %v", env)
	}
	if !matches(cmd, []string{"venv/bin/python", "main.py", "--listen", "127.0.0.1", "--port", "51234"}) {
		t.Error("the same command on another port did not match")
	}
	if matches(cmd, []string{"venv/bin/python", "other.py", "--listen", "127.0.0.1", "--port", "51234"}) {
		t.Error("another script matched")
	}
	pos, _ := template([]string{"python", "-m", "http.server", "8000", "--bind", "127.0.0.1"}, nil, "127.0.0.1:8000")
	if join(pos) != "python -m http.server {port} --bind {host}" {
		t.Errorf("a port given as a word: %q", pos)
	}
	eq, _ := template([]string{"x", "--port=9000", "--host=127.0.0.1"}, nil, "127.0.0.1:9000")
	if join(eq) != "x --port={port} --host={host}" {
		t.Errorf("--flag=value: %q", eq)
	}
}

func join(a []string) string {
	s := ""
	for i, x := range a {
		if i > 0 {
			s += " "
		}
		s += x
	}
	return s
}

// A server that ended without stopping its services (a crash) has the next
// one end them before it starts its own.
func TestLeftoversOfAServerBeforeAreEnded(t *testing.T) {
	macOnly(t)
	addr := freeAddr(t)
	path, _ := fake(t, addr, true, time.Hour)
	m := NewManager(path)
	if err := m.Start(); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the address taken", func() bool {
		c, err := net.Dial("tcp", addr)
		if err == nil {
			c.Close()
		}
		return err == nil
	})
	get(t, addr)
	left := state(m).PID
	if left == 0 {
		t.Fatal("not running")
	}
	// The server goes without stopping it: its listeners only, and its lock,
	// as a process that ends lets go of it.
	m.mu.Lock()
	for _, s := range m.svcs {
		if s.srv != nil {
			s.srv.Close()
			s.srv = nil
		}
	}
	m.mu.Unlock()
	m.lock.Close()

	next := NewManager(path)
	if err := next.Start(); err != nil {
		t.Fatal(err)
	}
	defer next.Close()
	if syscall.Kill(left, 0) == nil {
		syscall.Kill(-left, syscall.SIGKILL)
		t.Fatalf("pid %d, left by the server before, still runs", left)
	}
}

// A quiet path neither starts a stopped service — the front answers it — nor,
// running, keeps the service from its idle stop.
func TestQuietPathsDoNotKeepItUp(t *testing.T) {
	defer func(d time.Duration) { workCheck = d }(workCheck)
	workCheck = 300 * time.Millisecond
	addr := freeAddr(t)
	path, svc := fake(t, addr, true, time.Second)
	svc.Quiet = map[string]json.RawMessage{
		"/queue":                json.RawMessage(`{"queue_running":[],"queue_pending":[]}`),
		"/tasks?status=running": json.RawMessage(`[]`),
		"/health":               json.RawMessage(`{"ok":true}`),
	}
	// Running, the stand-in answers /health otherwise than written here (a
	// file of that name): not being in busy, that is no sign of work.
	if err := os.WriteFile(filepath.Join(svc.Dir, "health"), []byte(`{"ok":true,"loaded":true}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := Save(path, File{Services: []Service{svc}}); err != nil {
		t.Fatal(err)
	}
	m := NewManager(path)
	if err := m.Start(); err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	waitFor(t, "the address taken", func() bool {
		c, err := net.Dial("tcp", addr)
		if err == nil {
			c.Close()
		}
		return err == nil
	})
	c := http.Client{Timeout: 10 * time.Second}
	r, err := c.Get("http://" + addr + "/queue")
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(r.Body)
	r.Body.Close()
	if string(b) != `{"queue_running":[],"queue_pending":[]}` || state(m).State != "stopped" {
		t.Fatalf("stopped, /queue answered %q and it is %s", b, state(m).State)
	}
	for _, q := range []string{"/queue?x=1", "/tasks?status=running", "/health"} {
		if r, err := c.Get("http://" + addr + q); err == nil {
			r.Body.Close()
		}
		if st := state(m).State; st != "stopped" {
			t.Fatalf("%s started it", q)
		}
	}
	// Another question on a quiet path written with its query is a request.
	if r, err := c.Get("http://" + addr + "/tasks?status=done"); err == nil {
		r.Body.Close()
	}
	if st := state(m); st.State != "running" {
		t.Fatalf("/tasks?status=done did not start it: %+v", st)
	}
	// Polled every 200 ms, it still stops once idle.
	stop := time.After(4 * time.Second)
	for stopped := false; !stopped; {
		select {
		case <-stop:
			t.Fatalf("polling /queue kept it up: %+v", state(m))
		default:
		}
		for _, q := range []string{"/queue", "/health"} {
			if r, err := c.Get("http://" + addr + q); err == nil {
				io.Copy(io.Discard, r.Body)
				r.Body.Close()
			}
		}
		stopped = state(m).State == "stopped"
		time.Sleep(200 * time.Millisecond)
	}
}

// A service left at work with nobody waiting — a ComfyUI job let run on its
// own, its queue not empty — is not stopped idle, nor taken for room, until
// its quiet path answers as the front does for it stopped.
func TestAtWorkItIsNotStopped(t *testing.T) {
	defer func(d time.Duration) { workCheck = d }(workCheck)
	workCheck = 300 * time.Millisecond
	addr := freeAddr(t)
	path, svc := fake(t, addr, true, time.Second)
	idle := `{"queue_running":[],"queue_pending":[]}`
	svc.Quiet = map[string]json.RawMessage{"/queue": json.RawMessage(idle)}
	svc.Busy = []string{"/queue"}
	if err := Save(path, File{Services: []Service{svc}}); err != nil {
		t.Fatal(err)
	}
	queue := filepath.Join(svc.Dir, "queue") // what the stand-in answers on /queue
	if err := os.WriteFile(queue, []byte(`{"queue_running":[[0,"a job"]],"queue_pending":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	m := NewManager(path)
	if err := m.Start(); err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	waitFor(t, "the address taken", func() bool {
		c, err := net.Dial("tcp", addr)
		if err == nil {
			c.Close()
		}
		return err == nil
	})
	get(t, addr)
	time.Sleep(3 * time.Second) // three times its idle time
	if st := state(m); st.State != "running" {
		t.Fatalf("stopped at work: %+v", st)
	}
	if name, _, ok := m.Idlest(); ok {
		t.Fatalf("%s, at work, offered for room", name)
	}
	if err := os.WriteFile(queue, []byte(idle), 0o644); err != nil {
		t.Fatal(err)
	}
	if name, _, ok := m.Idlest(); !ok || name != "fake" {
		t.Errorf("done, not offered for room: %q %v", name, ok)
	}
	waitFor(t, "the idle stop once done", func() bool { return state(m).State == "stopped" })
}
