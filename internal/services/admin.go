package services

import (
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// Adopt hands the service called name to the manager: its own process, the
// one listening on its address, is asked to end — that process only, not
// what was started beside it — and the manager listens there in its place,
// starting it again when first asked for.
func (m *Manager) Adopt(name string, idle time.Duration) error {
	if err := m.active(); err != nil {
		return err
	}
	f, err := Load(m.path)
	if err != nil {
		return err
	}
	svc, i, ok := f.Find(name)
	switch {
	case !ok:
		return fmt.Errorf("no service called %s in %s", name, m.path)
	case svc.Launchd != "":
		return fmt.Errorf("%s is launchd's job %s, which would start it again: unload that job first", name, svc.Launchd)
	case !svc.Runnable():
		return fmt.Errorf("%s's command does not name its port, so it cannot run on another", name)
	case svc.Addr == m.Skip:
		return fmt.Errorf("%s is this server's own address", svc.Addr)
	}
	pid := listenerPID(svc.Port())
	if pid != 0 {
		if !m.ownsPID(svc, pid) {
			return fmt.Errorf("the process listening on %s (pid %d) is not %s's command; nothing was changed", svc.Addr, pid, name)
		}
	}
	f.Services[i].Adopted = true
	if idle > 0 {
		f.Services[i].Idle = Duration(idle)
	}
	if err := Save(m.path, f); err != nil {
		return err
	}
	if pid != 0 {
		end(pid)
	}
	log.Printf("services: %s adopted", name)
	return m.reload(true)
}

// ownsPID reports whether pid runs svc's command, its port whatever it is.
func (m *Manager) ownsPID(svc Service, pid int) bool {
	argv, _, err := procArgs(pid)
	if err != nil {
		return false
	}
	return matches(svc.Command, argv)
}

// matches compares a command with {port} and {host} in it to one that ran.
func matches(tmpl, argv []string) bool {
	if len(tmpl) != len(argv) {
		return false
	}
	for i, t := range tmpl {
		if !strings.Contains(t, "{") {
			if t != argv[i] {
				return false
			}
			continue
		}
		re := "^" + strings.NewReplacer(regexp.QuoteMeta("{port}"), `\d+`, regexp.QuoteMeta("{host}"), `[^\s]+`).Replace(regexp.QuoteMeta(t)) + "$"
		if ok, _ := regexp.MatchString(re, argv[i]); !ok {
			return false
		}
	}
	return true
}

// end asks pid to end, and makes it after 20 s.
func end(pid int) {
	terminate(pid)
	for i := 0; i < 100; i++ {
		if !alive(pid) {
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	killNow(pid)
}

// Release gives the service back: the manager stops its copy and lets go of
// the address, and the service is started on it again the way it was, on its
// own, outliving this server.
func (m *Manager) Release(name string) (int, error) {
	if err := m.active(); err != nil {
		return 0, err
	}
	f, err := Load(m.path)
	if err != nil {
		return 0, err
	}
	svc, i, ok := f.Find(name)
	if !ok {
		return 0, fmt.Errorf("no service called %s", name)
	}
	if !svc.Adopted {
		return 0, fmt.Errorf("%s is not adopted", name)
	}
	f.Services[i].Adopted = false
	if err := Save(m.path, f); err != nil {
		return 0, err
	}
	if err := m.reload(true); err != nil {
		return 0, err
	}
	pid, err := runAlone(svc, m.path)
	if err != nil {
		return 0, fmt.Errorf("%s was released but did not start on its own: %w", name, err)
	}
	log.Printf("services: %s released, running on its own (pid %d)", name, pid)
	return pid, nil
}

// runAlone starts svc on its own address in a session of its own, so it
// outlives the server that started it.
func runAlone(svc Service, file string) (int, error) {
	argv, env := svc.Args(svc.Host(), svc.Port())
	exe := argv[0]
	if strings.Contains(exe, "/") && !filepath.IsAbs(exe) && svc.Dir != "" {
		exe = filepath.Join(svc.Dir, exe)
	}
	cmd := exec.Command(exe, argv[1:]...)
	cmd.Dir = svc.Dir
	cmd.Env = append(os.Environ(), env...)
	cmd.SysProcAttr = sessionAttr()
	logPath := svc.Log
	if logPath == "" {
		logPath = filepath.Join(filepath.Dir(file), "services", svc.Name+".log")
	}
	os.MkdirAll(filepath.Dir(logPath), 0o755)
	out, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return 0, err
	}
	defer out.Close()
	cmd.Stdout, cmd.Stderr = out, out
	if err := cmd.Start(); err != nil {
		return 0, err
	}
	pid := cmd.Process.Pid
	go cmd.Wait() // reaped if it ends while this server runs
	return pid, nil
}

// active refuses a change to services another server runs.
func (m *Manager) active() error {
	if m.passive {
		return fmt.Errorf("the services in %s are run by the kinfer serve with pid %d: ask that one", m.path, m.owner)
	}
	return nil
}

// ErrNoManager is what the front's endpoints answer with no services file.
var ErrNoManager = errors.New("this server manages no services")
