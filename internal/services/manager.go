package services

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Manager runs the adopted services of a services file and watches the rest.
// The front owns one; see the package comment.
type Manager struct {
	path string
	// Room makes room on the GPU for need bytes before a service starts:
	// the front's own room-making, which also stops idle runners.
	Room func(name string, need uint64)
	// Listen wraps each address the manager takes over (the front's
	// orderly close).
	Listen func(net.Listener) net.Listener
	// Skip is an address the manager never takes: the front's own.
	Skip string

	mu   sync.Mutex
	svcs map[string]*managed
	// snap is the machine as Stats last looked: who listens where and whose
	// children are whose, two calls out (lsof, ps) that are not made again
	// for snapAge. /api/ps is asked often, by clients waiting for room.
	snap struct {
		at    time.Time
		ports map[int]int // port → listening pid
		tree  map[int][]int
	}
	mtime time.Time
	quit  chan struct{}
	wg    sync.WaitGroup
}

// managed is one service's state.
type managed struct {
	svc      Service
	srv      *http.Server // on svc.Addr while adopted and bound
	bindErr  string
	cmd      *exec.Cmd
	done     chan struct{} // closed when cmd has exited
	port     int
	ready    chan struct{} // closed once started, or failed (startErr)
	starting bool          // a request is starting it: the others wait on ready
	startErr error
	inflight int
	used     time.Time
	idle     *time.Timer
	peak     uint64
	started  time.Time
}

// startTimeout is how long a service may take to answer once started: a
// model of 10 GB loading from the disk takes most of a minute.
var startTimeout = 5 * time.Minute

// NewManager reads the services file at path. Start takes the adopted ones
// in.
func NewManager(path string) *Manager {
	return &Manager{path: path, svcs: map[string]*managed{}, quit: make(chan struct{})}
}

// Start takes in the file's adopted services and watches the file for
// changes from then on.
func (m *Manager) Start() error {
	m.endLeftovers()
	if err := m.reload(true); err != nil {
		return err
	}
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		t := time.NewTicker(3 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-m.quit:
				return
			case <-t.C:
				if err := m.reload(false); err != nil {
					log.Printf("services: %v", err)
				}
			}
		}
	}()
	return nil
}

// Close stops every service the manager started and lets go of their
// addresses.
func (m *Manager) Close() {
	close(m.quit)
	m.mu.Lock()
	var all []*managed
	for _, s := range m.svcs {
		all = append(all, s)
	}
	m.mu.Unlock()
	// All at once: launchd gives a job 20 s to end before it kills it, and
	// with it whatever it has not stopped.
	var stops sync.WaitGroup
	for _, s := range all {
		stops.Add(1)
		go func() {
			defer stops.Done()
			m.unbind(s)
			m.stop(s, "the server is stopping")
		}()
	}
	stops.Wait()
	m.wg.Wait()
}

// reload reads the file if it changed (or always, with force) and brings the
// running state in line with it.
func (m *Manager) reload(force bool) error {
	st, err := os.Stat(m.path)
	if errors.Is(err, os.ErrNotExist) {
		if force {
			return nil
		}
		st = nil
	} else if err != nil {
		return err
	}
	m.mu.Lock()
	if st != nil && !force && st.ModTime().Equal(m.mtime) {
		m.mu.Unlock()
		return nil
	}
	if st != nil {
		m.mtime = st.ModTime()
	}
	m.mu.Unlock()
	f, err := Load(m.path)
	if err != nil {
		return err
	}
	want := map[string]Service{}
	for _, s := range f.Services {
		want[s.Name] = s
	}
	m.mu.Lock()
	var unbind []*managed
	for name, s := range m.svcs {
		w, ok := want[name]
		if !ok || !w.Adopted || w.Addr != s.svc.Addr {
			unbind = append(unbind, s)
			if !ok {
				delete(m.svcs, name)
			}
		}
	}
	var bind []*managed
	for name, w := range want {
		s := m.svcs[name]
		if s == nil {
			s = &managed{svc: w}
			m.svcs[name] = s
		}
		s.svc = w
		if w.Adopted && s.srv == nil && w.Addr != m.Skip {
			bind = append(bind, s)
		}
	}
	m.mu.Unlock()
	for _, s := range unbind {
		m.unbind(s)
		m.stop(s, "no longer adopted")
	}
	for _, s := range bind {
		m.bind(s)
	}
	return nil
}

// bind listens on the service's address, trying again every two seconds
// while something else still holds it (its own process, ending).
func (m *Manager) bind(s *managed) {
	m.mu.Lock()
	if s.srv != nil {
		m.mu.Unlock()
		return
	}
	srv := &http.Server{Handler: m.handler(s), ReadHeaderTimeout: 10 * time.Second}
	s.srv = srv
	addr := s.svc.Addr
	m.mu.Unlock()
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		logged := false
		for {
			l, err := net.Listen("tcp", addr)
			if err == nil {
				if m.Listen != nil {
					l = m.Listen(l)
				}
				m.mu.Lock()
				s.bindErr = ""
				cur := s.srv
				m.mu.Unlock()
				if cur != srv { // released while waiting
					l.Close()
					return
				}
				log.Printf("services: %s at %s, started when first asked for", s.svc.Name, addr)
				if err := srv.Serve(l); err != nil && !errors.Is(err, http.ErrServerClosed) {
					log.Printf("services: %s at %s: %v", s.svc.Name, addr, err)
				}
				return
			}
			m.mu.Lock()
			s.bindErr = err.Error()
			gone := s.srv != srv
			m.mu.Unlock()
			if gone {
				return
			}
			if !logged {
				log.Printf("services: %s: %s is held (%v); trying again every 2 s", s.svc.Name, addr, err)
				logged = true
			}
			select {
			case <-m.quit:
				return
			case <-time.After(2 * time.Second):
			}
		}
	}()
}

func (m *Manager) unbind(s *managed) {
	m.mu.Lock()
	srv := s.srv
	s.srv = nil
	m.mu.Unlock()
	if srv != nil {
		srv.Close()
	}
}

// handler forwards to the service, started first if it is not running.
func (m *Manager) handler(s *managed) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		port, err := m.acquire(s)
		if err != nil {
			http.Error(w, fmt.Sprintf("kinfer: %s could not be started: %v", s.svc.Name, err), http.StatusBadGateway)
			return
		}
		defer m.release(s)
		target := &url.URL{Scheme: "http", Host: "127.0.0.1:" + strconv.Itoa(port)}
		p := &httputil.ReverseProxy{
			Rewrite: func(pr *httputil.ProxyRequest) {
				pr.SetURL(target)
				pr.Out.Host = pr.In.Host // as its clients address it
				pr.SetXForwarded()
			},
			FlushInterval: -1, // a stream goes out as it is written
			ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
				if errors.Is(err, context.Canceled) {
					return
				}
				http.Error(w, fmt.Sprintf("kinfer: %s: %v", s.svc.Name, err), http.StatusBadGateway)
			},
		}
		p.ServeHTTP(w, r) // a WebSocket is held here until it closes
	})
}

// acquire counts a request in, starting the service if it must be, and
// returns its port.
func (m *Manager) acquire(s *managed) (int, error) {
	m.mu.Lock()
	s.inflight++
	if s.idle != nil {
		s.idle.Stop()
		s.idle = nil
	}
	if !s.starting && (s.ready == nil || s.startErr != nil || exited(s.done)) {
		s.starting = true
		s.ready = make(chan struct{})
		s.startErr = nil
		ready := s.ready
		m.mu.Unlock()
		err := m.start(s)
		m.mu.Lock()
		s.startErr, s.starting = err, false
		close(ready)
	}
	ready := s.ready
	m.mu.Unlock()
	<-ready
	m.mu.Lock()
	defer m.mu.Unlock()
	if s.startErr != nil {
		s.inflight--
		return 0, s.startErr
	}
	return s.port, nil
}

func exited(done chan struct{}) bool {
	if done == nil {
		return true
	}
	select {
	case <-done:
		return true
	default:
		return false
	}
}

// release counts a request out; the last one out starts the idle clock.
func (m *Manager) release(s *managed) {
	m.mu.Lock()
	defer m.mu.Unlock()
	s.inflight--
	s.used = time.Now()
	if s.inflight == 0 && s.svc.Idle > 0 {
		if s.idle != nil {
			s.idle.Stop()
		}
		s.idle = time.AfterFunc(time.Duration(s.svc.Idle), func() {
			m.mu.Lock()
			busy := s.inflight > 0
			m.mu.Unlock()
			if !busy {
				m.stop(s, "idle for "+time.Duration(s.svc.Idle).String())
			}
		})
	}
}

// start runs the service on a free loopback port and waits until it answers.
func (m *Manager) start(s *managed) error {
	m.mu.Lock()
	svc := s.svc
	m.mu.Unlock()
	if !svc.Runnable() {
		return errors.New("its command does not name its port")
	}
	if m.Room != nil {
		m.Room(svc.Name, svc.Need)
	}
	port, err := freePort()
	if err != nil {
		return err
	}
	argv, env := svc.Args("127.0.0.1", port)
	exe := argv[0]
	if strings.Contains(exe, "/") && !filepath.IsAbs(exe) && svc.Dir != "" {
		exe = filepath.Join(svc.Dir, exe)
	}
	cmd := exec.Command(exe, argv[1:]...)
	cmd.Dir = svc.Dir
	cmd.Env = append(os.Environ(), env...)
	cmd.SysProcAttr = groupAttr() // its children stop with it
	logPath := svc.Log
	if logPath == "" {
		logPath = filepath.Join(filepath.Dir(m.path), "services", svc.Name+".log")
	}
	os.MkdirAll(filepath.Dir(logPath), 0o755)
	out, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "\n==== kinfer started %s on 127.0.0.1:%d at %s\n", svc.Name, port, time.Now().Format(time.RFC3339))
	cmd.Stdout, cmd.Stderr = out, out
	t0 := time.Now()
	if err := cmd.Start(); err != nil {
		out.Close()
		return err
	}
	done := make(chan struct{})
	go func() { cmd.Wait(); out.Close(); close(done) }()
	m.mu.Lock()
	s.cmd, s.done, s.port, s.started, s.peak = cmd, done, port, t0, 0
	m.mu.Unlock()
	m.notePID(svc.Name, cmd.Process.Pid)
	log.Printf("services: %s starting (pid %d, 127.0.0.1:%d)", svc.Name, cmd.Process.Pid, port)

	deadline := time.Now().Add(startTimeout)
	for {
		select {
		case <-done:
			return fmt.Errorf("it ended while starting; see %s", logPath)
		default:
		}
		if answers(port, svc.Health) {
			break
		}
		if time.Now().After(deadline) {
			m.stop(s, "it did not answer within "+startTimeout.String())
			return fmt.Errorf("it did not answer within %v; see %s", startTimeout, logPath)
		}
		time.Sleep(200 * time.Millisecond)
	}
	log.Printf("services: %s answering after %v", svc.Name, time.Since(t0).Round(100*time.Millisecond))
	m.wg.Add(1)
	go m.measure(s, cmd.Process.Pid, done)
	return nil
}

// answers reports the service up: its port accepts a connection, and its
// health path, if it has one, answers below 500.
func answers(port int, health string) bool {
	c, err := net.DialTimeout("tcp", "127.0.0.1:"+strconv.Itoa(port), 500*time.Millisecond)
	if err != nil {
		return false
	}
	c.Close()
	if health == "" {
		return true
	}
	cl := http.Client{Timeout: 2 * time.Second}
	r, err := cl.Get("http://127.0.0.1:" + strconv.Itoa(port) + "/" + strings.TrimPrefix(health, "/"))
	if err != nil {
		return false
	}
	io.Copy(io.Discard, r.Body)
	r.Body.Close()
	return r.StatusCode < 500
}

// measure follows what the running service holds, and keeps the most it was
// seen to: the room made for it next time.
func (m *Manager) measure(s *managed, pid int, done chan struct{}) {
	defer m.wg.Done()
	t := time.NewTicker(5 * time.Second)
	defer t.Stop()
	for {
		n := treeFootprint(pid, children())
		m.mu.Lock()
		if n > s.peak {
			s.peak = n
		}
		m.mu.Unlock()
		select {
		case <-done:
			return
		case <-m.quit:
			return
		case <-t.C:
		}
	}
}

// stop ends the service's process, if the manager started one: asked first,
// then made to after 15 s. The most memory it held is written to the file.
func (m *Manager) stop(s *managed, why string) {
	m.mu.Lock()
	cmd, done, peak := s.cmd, s.done, s.peak
	if s.idle != nil {
		s.idle.Stop()
		s.idle = nil
	}
	m.mu.Unlock()
	if cmd == nil || exited(done) {
		return
	}
	pgid := cmd.Process.Pid
	signalGroup(pgid, false)
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		signalGroup(pgid, true)
		<-done
	}
	m.notePID(s.svc.Name, 0)
	log.Printf("services: %s stopped (%s), having held %.1f GB at most", s.svc.Name, why, float64(peak)/1e9)
	if peak > 0 {
		m.remember(s.svc.Name, peak)
	}
}

// remember writes the most a service held into the file.
func (m *Manager) remember(name string, peak uint64) {
	f, err := Load(m.path)
	if err != nil {
		return
	}
	if _, i, ok := f.Find(name); ok && f.Services[i].Need != peak {
		f.Services[i].Need = peak
		if Save(m.path, f) == nil {
			m.mu.Lock()
			if st, err := os.Stat(m.path); err == nil {
				m.mtime = st.ModTime()
			}
			if s := m.svcs[name]; s != nil {
				s.svc.Need = peak
			}
			m.mu.Unlock()
		}
	}
}

// pidsPath is where the manager notes the processes it started: a server
// that ended without stopping them (a crash; launchd starts it again) has its
// successor end them, rather than start a second copy beside each.
func (m *Manager) pidsPath() string { return m.path + ".pids" }

func (m *Manager) notePID(name string, pid int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	pids := map[string]int{}
	if b, err := os.ReadFile(m.pidsPath()); err == nil {
		json.Unmarshal(b, &pids)
	}
	if pid == 0 {
		delete(pids, name)
	} else {
		pids[name] = pid
	}
	b, _ := json.Marshal(pids)
	os.WriteFile(m.pidsPath(), b, 0o644)
}

// endLeftovers ends what a server before this one started and left running:
// each noted process still alive and still running its service's command.
func (m *Manager) endLeftovers() {
	b, err := os.ReadFile(m.pidsPath())
	if err != nil {
		return
	}
	var pids map[string]int
	if json.Unmarshal(b, &pids) != nil {
		return
	}
	f, _ := Load(m.path)
	for name, pid := range pids {
		svc, _, ok := f.Find(name)
		if !ok || pid <= 1 || !alive(pid) {
			continue
		}
		// Its own process group, as the manager starts them, and its
		// command but for the program's path, which a launcher (python3)
		// may run itself again under.
		pgid := pgidOf(pid)
		if argv, _, err := procArgs(pid); err == nil && pgid == pid && len(argv) == len(svc.Command) &&
			len(argv) > 0 && matches(svc.Command[1:], argv[1:]) {
			log.Printf("services: ending %s (pid %d), left running by the server before this one", name, pid)
			signalGroup(pid, false) // its group, as it was started
			end(pid)
		}
	}
	os.Remove(m.pidsPath())
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// Idlest is the adopted service running that nothing uses and is not kept,
// unused longest: the one to stop for room, and since when it is unused.
func (m *Manager) Idlest() (name string, since time.Time, ok bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, s := range m.svcs {
		if s.inflight > 0 || s.svc.Idle == 0 || exited(s.done) || s.started.IsZero() {
			continue
		}
		u := s.used
		if u.IsZero() {
			u = s.started
		}
		if !ok || u.Before(since) {
			name, since, ok = s.svc.Name, u, true
		}
	}
	return name, since, ok
}

// Stop stops the service called name if the manager runs it.
func (m *Manager) Stop(name, why string) bool {
	m.mu.Lock()
	s := m.svcs[name]
	m.mu.Unlock()
	if s == nil || exited(s.done) {
		return false
	}
	m.stop(s, why)
	return true
}

// Stat is what /api/ps and kinfer service list say of a service.
type Stat struct {
	Name      string    `json:"name"`
	Addr      string    `json:"addr"`
	Adopted   bool      `json:"adopted"`
	State     string    `json:"state"` // running, starting, stopped; for one watched, up or down
	PID       int       `json:"pid,omitempty"`
	Footprint uint64    `json:"footprint,omitempty"` // bytes held now
	Need      uint64    `json:"need,omitempty"`      // the most it was seen to hold
	InFlight  int       `json:"in_flight,omitempty"`
	UsedAt    time.Time `json:"used_at,omitzero"`
	Idle      string    `json:"idle,omitempty"`
	Note      string    `json:"note,omitempty"`
}

// Stats is every service's state, by name, the file read again first if it
// changed.
func (m *Manager) Stats() []Stat {
	if err := m.reload(false); err != nil {
		log.Printf("services: %v", err)
	}
	m.mu.Lock()
	var list []*managed
	for _, s := range m.svcs {
		list = append(list, s)
	}
	m.mu.Unlock()
	ports, tree := m.machine()
	var out []Stat
	for _, s := range list {
		m.mu.Lock()
		st := Stat{Name: s.svc.Name, Addr: s.svc.Addr, Adopted: s.svc.Adopted, Need: s.svc.Need,
			InFlight: s.inflight, UsedAt: s.used, Note: s.bindErr}
		if s.svc.Idle > 0 {
			st.Idle = time.Duration(s.svc.Idle).String()
		} else if s.svc.Adopted {
			st.Idle = "kept"
		}
		var pid int
		switch {
		case !s.svc.Adopted:
			st.State = "down"
		case !exited(s.done):
			pid = s.cmd.Process.Pid
			st.State = "running"
			if s.ready != nil {
				select {
				case <-s.ready:
				default:
					st.State = "starting"
				}
			}
		default:
			st.State = "stopped"
		}
		m.mu.Unlock()
		if !s.svc.Adopted {
			if p := ports[s.svc.Port()]; p != 0 {
				pid, st.State = p, "up"
			}
		}
		if pid != 0 {
			st.PID = pid
			st.Footprint = treeFootprint(pid, tree)
		}
		out = append(out, st)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

const snapAge = 3 * time.Second

// machine is who listens on each port and the process tree, as of at most
// snapAge ago.
func (m *Manager) machine() (map[int]int, map[int][]int) {
	m.mu.Lock()
	if time.Since(m.snap.at) < snapAge && m.snap.ports != nil {
		p, t := m.snap.ports, m.snap.tree
		m.mu.Unlock()
		return p, t
	}
	m.mu.Unlock()
	ports := map[int]int{}
	if ls, err := listening(); err == nil {
		for pid, addrs := range ls {
			for _, a := range addrs {
				if n := (Service{Addr: a}).Port(); n != 0 {
					ports[n] = pid
				}
			}
		}
	}
	tree := children()
	m.mu.Lock()
	m.snap.at, m.snap.ports, m.snap.tree = time.Now(), ports, tree
	m.mu.Unlock()
	return ports, tree
}

// listenerPID is the process listening on port, 0 if none.
func listenerPID(port int) int {
	b, _ := exec.Command("lsof", "-t", "-nP", "-iTCP:"+strconv.Itoa(port), "-sTCP:LISTEN").Output()
	f := strings.Fields(string(b))
	if len(f) == 0 {
		return 0
	}
	n, _ := strconv.Atoi(f[0])
	return n
}
