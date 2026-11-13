package server

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// workEngine is a model with requests on it or not, which moves or not.
type workEngine struct {
	fakeEngine
	busy, waiting atomic.Int64
	eval, steps   atomic.Int64
}

func (e *workEngine) Load() (int, int, int) { return int(e.waiting.Load()), int(e.busy.Load()), 1 }
func (e *workEngine) EvalTokens() int64     { return e.eval.Load() }
func (e *workEngine) Steps() int64          { return e.steps.Load() }

// A model is stuck when it has had requests on it and not moved for the
// whole time; idle, or moving however slowly, it is not.
func TestTheWatchdogKnowsAStuckModelFromABusyOne(t *testing.T) {
	w := newWatchdog("", 10*time.Minute)
	e := &workEngine{}
	t0 := time.Date(2026, 11, 13, 12, 0, 0, 0, time.Local)
	at := func(m int) time.Time { return t0.Add(time.Duration(m) * time.Minute) }
	look := func(m int) string { return w.model(e, "/m/ornith-1.5-35b.gguf", at(m)) }

	look(0)
	if why := look(30); why != "" { // idle half an hour
		t.Fatalf("idle: %s", why)
	}
	e.busy.Store(1)
	for m := 31; m <= 60; m++ { // a long reply, a pass a minute: the decodes move as the reply does not
		e.steps.Add(1)
		if why := look(m); why != "" {
			t.Fatalf("moving at minute %d: %s", m, why)
		}
	}
	e.eval.Add(500) // the reply done
	look(61)
	if why := look(70); why != "" { // nine minutes still
		t.Fatalf("still for nine minutes: %s", why)
	}
	why := look(71)
	if !strings.Contains(why, "ornith-1.5-35b has not moved for 10m0s with 1 requests on it") {
		t.Fatalf("still for ten minutes: %q", why)
	}

	// Another model loaded: its time starts with it.
	e2 := &workEngine{}
	e2.waiting.Store(2)
	if why := w.model(e2, "/m/other.gguf", at(72)); why != "" {
		t.Errorf("a model just loaded: %s", why)
	}
	if why := w.model(e2, "/m/other.gguf", at(81)); why != "" {
		t.Errorf("nine minutes after it loaded: %s", why)
	}
	if why := w.model(nil, "", at(200)); why != "" {
		t.Errorf("no model: %s", why)
	}
}

// Its own address unanswered four times in a row ends the process; once
// answered, the count starts again.
func TestTheWatchdogAsksItsOwnAddress(t *testing.T) {
	var hang atomic.Bool
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hang.Load() {
			<-release
		}
		w.Write([]byte(`{"version":"x"}`))
	}))
	defer srv.Close()
	defer close(release)

	w := newWatchdog(srv.URL, 10*time.Minute)
	w.client.Timeout = 50 * time.Millisecond
	if why := w.probe(); why != "" || w.misses != 0 {
		t.Fatalf("answered: %q %d", why, w.misses)
	}
	hang.Store(true)
	for i := 1; i < probeMisses; i++ {
		if why := w.probe(); why != "" {
			t.Fatalf("unanswered %d times: %s", i, why)
		}
	}
	hang.Store(false)
	if why := w.probe(); why != "" || w.misses != 0 {
		t.Fatalf("answered again: %q %d", why, w.misses)
	}
	hang.Store(true)
	var why string
	for i := 0; i < probeMisses; i++ {
		why = w.probe()
	}
	if !strings.Contains(why, "has not answered 4 times in a row") {
		t.Fatalf("unanswered four times in a row: %q", why)
	}
}

// A look at the server reads its model and asks its address; a lock held
// that moment skips the model, and the address is still asked.
func TestTheWatchdogLooksAtTheServer(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	e := &workEngine{}
	e.busy.Store(1)
	srv.cur = &lease{eng: e, path: "/m/alpha.gguf", refs: 1}
	w := newWatchdog("", time.Minute)
	t0 := time.Now()
	w.check(srv, t0)
	if why := w.check(srv, t0.Add(2*time.Minute)); !strings.Contains(why, "alpha has not moved") {
		t.Errorf("a model stuck: %q", why)
	}
	w = newWatchdog("http://127.0.0.1:1", time.Minute) // nothing listens there
	w.client.Timeout = 50 * time.Millisecond
	srv.mu.Lock()
	w.check(srv, t0)
	srv.mu.Unlock()
	if w.misses != 1 {
		t.Errorf("with the lock held, the address was not asked: %d", w.misses)
	}
}

// A speech or picture runner is given the server's -watchdog; a chat one
// has it among the flags it takes.
func TestRunnersAreGivenTheWatchdog(t *testing.T) {
	srv, _ := newTestServer(t, "alpha")
	srv.watchStall = 3 * time.Minute
	if args := srv.runnerArgs("5000", false); !slices.Contains(args, "-watchdog") || args[slices.Index(args, "-watchdog")+1] != "3m0s" {
		t.Errorf("a speech runner's flags: %v", args)
	}
}

// Giving up ends the process with 2, the reason in the log and every
// goroutine's stack in ~/.kinfer: run as a process of its own.
func TestGivingUpEndsTheProcess(t *testing.T) {
	if os.Getenv("KINFER_TEST_GIVE_UP") == "1" {
		giveUp("the test's model has not moved")
		return
	}
	home := t.TempDir()
	os.Mkdir(filepath.Join(home, ".kinfer"), 0o755)
	cmd := exec.Command(os.Args[0], "-test.run=^TestGivingUpEndsTheProcess$")
	cmd.Env = append(os.Environ(), "KINFER_TEST_GIVE_UP=1", "HOME="+home, "USERPROFILE="+home)
	out, err := cmd.CombinedOutput()
	var ee *exec.ExitError
	if !errors.As(err, &ee) || ee.ExitCode() != 2 {
		t.Fatalf("exit: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "the test's model has not moved: ending this process") {
		t.Errorf("the log:\n%s", out)
	}
	files, _ := filepath.Glob(filepath.Join(home, ".kinfer", "hung-*.txt"))
	if len(files) != 1 {
		t.Fatalf("stack files: %v", files)
	}
	b, _ := os.ReadFile(files[0])
	if !strings.HasPrefix(string(b), "the test's model has not moved") || !strings.Contains(string(b), "goroutine ") {
		t.Errorf("the stack file:\n%.300s", b)
	}
}
