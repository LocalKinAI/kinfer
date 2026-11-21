package server

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime/pprof"
	"time"
)

// ─── the watchdog ────────────────────────────────────────────────────────────
//
// launchd starts kinfer serve again when it exits, and the front server starts
// a runner again when its process ends. Neither of them can tell when a process
// has hung. A process that hangs gives no answers and no errors, and to a
// client it looks like a busy one. So each process watches itself, and ends
// once it is plainly stuck:
//
//   - The model it serves has requests on it and has not moved for -watchdog
//     (10 minutes): no prompt chunk read, no token written, no step taken. A
//     GPU that never finishes a pass looks like this. A runner that ends this
//     way is started again by the next request for its model; a server
//     without runners is started again by launchd.
//   - Its own address has not answered /api/version four times in a row, half
//     a minute apart, each time given 10 seconds.
//
// Before it ends it logs why, and writes every goroutine's stack to
// ~/.kinfer/hung-<time>-<pid>.txt for whoever looks into it.

const (
	watchEvery  = 30 * time.Second // between looks
	probeWait   = 10 * time.Second // for its own address to answer
	probeMisses = 4                // unanswered in a row, then it ends
)

// stepper is an engine that counts its passes as they run (engine.Engine).
// The others' counters move as their requests finish, which, for a speech
// or picture model, comes well inside the watchdog's time.
type stepper interface{ Steps() int64 }

// progress is how far g has got: it moves while g works.
func progress(g generator) int64 {
	n := g.PromptTokens() + g.EvalTokens()
	if st, ok := g.(stepper); ok {
		n += st.Steps()
	}
	return n
}

type watchdog struct {
	stall  time.Duration
	self   string // this server's own address, as a URL; "" looks at the model only
	client *http.Client
	misses int

	eng   generator // the model looked at last
	seen  int64     // its progress then
	since time.Time // when it was last seen idle, or moving
}

// Watch starts the watchdog of this process, which ends it when its model
// has not moved for stall with requests on it, or its own address addr has
// stopped answering. A stall of 0 starts none. It stops with ctx.
func (s *Server) Watch(ctx context.Context, addr string, stall time.Duration) {
	s.watchStall = stall
	if stall <= 0 {
		return
	}
	w := newWatchdog(loopbackURL(addr), stall)
	go func() {
		t := time.NewTicker(watchEvery)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case now := <-t.C:
				if why := w.check(s, now); why != "" && ctx.Err() == nil {
					giveUp(why)
					return
				}
			}
		}
	}()
}

func newWatchdog(self string, stall time.Duration) *watchdog {
	return &watchdog{stall: stall, self: self, client: &http.Client{
		Timeout: probeWait,
		// Its own connections: a probe must not wait behind the requests
		// the process is answering, nor leave one open between looks.
		Transport: &http.Transport{DisableKeepAlives: true},
	}}
}

// check is one look; it says why the process is stuck, or "".
func (w *watchdog) check(s *Server, now time.Time) string {
	var g generator
	var path string
	// A lock held this moment is no sign of anything: look next time.
	if s.mu.TryLock() {
		if s.cur != nil {
			g, path = s.cur.eng, s.cur.path
		}
		s.mu.Unlock()
		if why := w.model(g, path, now); why != "" {
			return why
		}
	}
	return w.probe()
}

// model is the look at the model this process serves, g at path.
func (w *watchdog) model(g generator, path string, now time.Time) string {
	if g != w.eng {
		w.eng, w.since = g, now
		if g != nil {
			w.seen = progress(g)
		}
		return ""
	}
	if g == nil {
		return ""
	}
	waiting, busy, _ := g.Load()
	if p := progress(g); waiting+busy == 0 || p != w.seen {
		w.seen, w.since = p, now
		return ""
	}
	if now.Sub(w.since) < w.stall {
		return ""
	}
	return fmt.Sprintf("%s has not moved for %s with %d requests on it", logName(path), w.stall.Round(time.Second), waiting+busy)
}

// probe asks this server's own address for its version.
func (w *watchdog) probe() string {
	if w.self == "" {
		return ""
	}
	resp, err := w.client.Get(w.self + "/api/version")
	if err == nil {
		io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			w.misses = 0
			return ""
		}
		err = fmt.Errorf("answered %s", resp.Status)
	}
	w.misses++
	log.Printf("watchdog: %s did not answer (%d of %d in a row): %v", w.self, w.misses, probeMisses, err)
	if w.misses < probeMisses {
		return ""
	}
	return fmt.Sprintf("its own address %s has not answered %d times in a row", w.self, probeMisses)
}

// giveUp ends this process, saying why first. Tests replace it.
var giveUp = func(why string) {
	log.Printf("watchdog: %s: ending this process, so that it is started again", why)
	if home, err := os.UserHomeDir(); err == nil {
		dir := filepath.Join(home, ".kinfer")
		name := filepath.Join(dir, fmt.Sprintf("hung-%s-%d.txt", time.Now().Format("20060102-150405"), os.Getpid()))
		if f, err := os.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600); err == nil {
			fmt.Fprintf(f, "%s\n\n", why)
			pprof.Lookup("goroutine").WriteTo(f, 2)
			f.Close()
			log.Printf("watchdog: what each goroutine was doing is in %s", name)
		}
	}
	// os.Exit runs the C libraries' exit handlers, and a GPU that is stuck
	// can hold one of those too: the process is killed if it is still here.
	go func() {
		time.Sleep(10 * time.Second)
		if p, err := os.FindProcess(os.Getpid()); err == nil {
			p.Kill()
		}
	}()
	os.Exit(2)
}
