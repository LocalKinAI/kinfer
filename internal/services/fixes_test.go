//go:build !windows

package services

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"testing"
	"time"
)

func takeAddr(t *testing.T, m *Manager, addr string) {
	t.Helper()
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
}

// A second server on the same file — a kinfer run's daemon — took the first's
// running services for a crashed server's leftovers and ended them. Now it
// leaves them alone: it ends nothing, binds nothing, reports nothing, changes
// nothing; and once the first has gone, the next one is the one that runs them.
func TestASecondServerLeavesTheFirstsServicesAlone(t *testing.T) {
	addr := freeAddr(t)
	path, _ := fake(t, addr, true, time.Hour)
	first := NewManager(path)
	takeAddr(t, first, addr)
	get(t, addr)
	pid := state(first).PID
	if pid == 0 {
		t.Fatal("not running")
	}

	second := NewManager(path)
	if err := second.Start(); err != nil {
		t.Fatal(err)
	}
	if syscall.Kill(pid, 0) != nil {
		t.Fatal("the second server ended the first's running service")
	}
	if !second.passive || second.owner != os.Getpid() {
		t.Fatalf("the second server runs the file too (passive %v, owner %d)", second.passive, second.owner)
	}
	if len(second.Stats()) != 0 {
		t.Error("the second server reports services it does not run")
	}
	if second.Stop("fake", "test") {
		t.Error("the second server stopped the first's service")
	}
	if _, _, ok := second.Idlest(); ok {
		t.Error("the second server offers the first's service for room")
	}
	if err := second.Adopt("fake", 0); err == nil {
		t.Error("the second server changed the file")
	}
	second.Close()
	if get(t, addr) != 200 || syscall.Kill(pid, 0) != nil {
		t.Fatal("the first server's service is not as it was")
	}

	first.Close()
	third := NewManager(path)
	takeAddr(t, third, addr)
	defer third.Close()
	if third.passive {
		t.Fatal("with the first gone, the next server does not run the file")
	}
}

// A request that comes while the service is being stopped waits for the stop
// and starts it again; it was forwarded to the process on its way out, a 502.
func TestARequestDuringAStopStartsItAgain(t *testing.T) {
	addr := freeAddr(t)
	path, svc := fake(t, addr, true, time.Hour)
	m := NewManager(path)
	takeAddr(t, m, addr)
	defer m.Close()
	get(t, addr)
	m.mu.Lock()
	s := m.svcs["fake"]
	m.mu.Unlock()
	before := state(m).PID

	// The stop is held between being decided and the process ending.
	m.mu.Lock()
	ch, ok := m.beginStop(s)
	m.mu.Unlock()
	if !ok {
		t.Fatal("nothing to stop")
	}
	codes := make(chan int, 1)
	go func() { codes <- get(t, addr) }()
	time.Sleep(300 * time.Millisecond)
	select {
	case c := <-codes:
		t.Fatalf("answered %d while the service was being stopped", c)
	default:
	}
	m.finishStop(s, "test", ch)
	if c := <-codes; c != 200 {
		t.Fatalf("the request that waited: %d", c)
	}
	if after := state(m).PID; after == 0 || after == before {
		t.Errorf("not started again: pid %d, was %d", after, before)
	}
	_ = svc
}

// Stopping for room looks again under the lock: a request in it, or one since
// it was found unused, and it is not stopped.
func TestStopForRoomLooksAgain(t *testing.T) {
	addr := freeAddr(t)
	path, _ := fake(t, addr, true, time.Hour)
	m := NewManager(path)
	takeAddr(t, m, addr)
	defer m.Close()
	get(t, addr)
	name, since, ok := m.Idlest()
	if !ok || name != "fake" {
		t.Fatalf("idlest: %q %v", name, ok)
	}
	get(t, addr) // used after it was found unused
	if m.StopUnused("fake", since, "room") || state(m).State != "running" {
		t.Fatal("stopped though used since it was found unused")
	}
	m.mu.Lock()
	s := m.svcs["fake"]
	s.inflight++ // a request in it
	m.mu.Unlock()
	_, since, _ = m.Idlest()
	if m.StopUnused("fake", time.Now(), "room") || state(m).State != "running" {
		t.Fatal("stopped with a request in it")
	}
	m.mu.Lock()
	s.inflight--
	m.mu.Unlock()
	_, since, _ = m.Idlest()
	if !m.StopUnused("fake", since, "room") || state(m).State != "stopped" {
		t.Fatal("unused, not stopped for room")
	}
}

// A busy path that does not answer, or answers 5xx, is a service at work: it
// was taken for idle and stopped mid-job. A path it does not have says nothing.
func TestAnUnansweredBusyProbeIsWork(t *testing.T) {
	var mu sync.Mutex
	mode := "slow"
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		md := mode
		mu.Unlock()
		switch md {
		case "slow":
			time.Sleep(6 * time.Second) // past the probe's 5 s
		case "500":
			w.WriteHeader(500)
		case "404":
			w.WriteHeader(404)
		case "idle":
			io.WriteString(w, `{"queue_running":[]}`)
		}
	}))
	defer ts.Close()
	port := ts.Listener.Addr().(*net.TCPAddr).Port
	m := NewManager(filepath.Join(t.TempDir(), "services.json"))
	s := &managed{port: port, svc: Service{Busy: []string{"/queue"},
		Quiet: map[string]json.RawMessage{"/queue": json.RawMessage(`{"queue_running":[]}`)}}}
	for _, c := range []struct {
		mode string
		work bool
	}{{"slow", true}, {"500", true}, {"404", false}, {"idle", false}} {
		mu.Lock()
		mode = c.mode
		mu.Unlock()
		if got := m.atWork(s); got != c.work {
			t.Errorf("%s: at work %v, want %v", c.mode, got, c.work)
		}
	}
}

// What a service held is the most of all its runs: one stopped soon after it
// started wrote its few MB over the 11 GB it holds in use.
func TestTheMemoryAServiceNeedsIsKeptAtItsMost(t *testing.T) {
	path, svc := fake(t, freeAddr(t), true, time.Hour)
	svc.Need = 11 << 30
	if err := Save(path, File{Services: []Service{svc}}); err != nil {
		t.Fatal(err)
	}
	m := NewManager(path)
	m.remember("fake", 5<<20)
	if f, _ := Load(path); f.Services[0].Need != 11<<30 {
		t.Fatalf("a short run wrote %d over 11 GB", f.Services[0].Need)
	}
	m.remember("fake", 12<<30)
	if f, _ := Load(path); f.Services[0].Need != 12<<30 {
		t.Fatalf("a larger run was not kept: %d", f.Services[0].Need)
	}
}

// A file that cannot be read when the server starts is read once it is
// mended: it was read only when someone asked /api/ps.
func TestAFileMendedLaterIsTakenIn(t *testing.T) {
	addr := freeAddr(t)
	path, svc := fake(t, addr, true, time.Hour)
	good, _ := os.ReadFile(path)
	os.WriteFile(path, []byte(`{"services": [`), 0o600)
	m := NewManager(path)
	if err := m.Start(); err != nil {
		t.Fatalf("a broken file stopped the server: %v", err)
	}
	defer m.Close()
	time.Sleep(1100 * time.Millisecond) // a newer time on the file
	os.WriteFile(path, good, 0o600)
	waitFor(t, "the mended file taken in", func() bool {
		c, err := net.Dial("tcp", addr)
		if err == nil {
			c.Close()
		}
		return err == nil
	})
	_ = svc
}

// The services file and its pids are their owner's only: a scanned service's
// environment, keys with it, is kept in the file.
func TestTheFilesArePrivate(t *testing.T) {
	addr := freeAddr(t)
	path, _ := fake(t, addr, true, time.Hour)
	os.Chmod(path, 0o644) // as v0.8.0 wrote it
	m := NewManager(path)
	takeAddr(t, m, addr)
	defer m.Close()
	get(t, addr)
	for _, p := range []string{path, path + ".pids", path + ".lock"} {
		if st, err := os.Stat(p); err != nil || st.Mode().Perm()&0o077 != 0 {
			t.Errorf("%s: %v %v", filepath.Base(p), err, st.Mode())
		}
	}
}

// The last real request ending while a quiet one is still in flight starts the
// idle clock all the same: it was never armed, and the service never stopped.
func TestAQuietRequestInFlightDoesNotKeepTheClockFromStarting(t *testing.T) {
	m := NewManager(filepath.Join(t.TempDir(), "services.json"))
	s := &managed{svc: Service{Idle: Duration(time.Hour)}, inflight: 2, quietIn: 1}
	m.release(s) // the real one ends; the quiet one is still in
	m.mu.Lock()
	armed := s.idle != nil
	if s.idle != nil {
		s.idle.Stop()
	}
	m.mu.Unlock()
	if !armed {
		t.Fatal("the idle clock did not start")
	}
}
