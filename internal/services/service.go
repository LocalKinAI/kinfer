// Package services lets kinfer serve's front take in the other programs on a
// machine that answer on a port — ComfyUI, a talking face, a navigation
// model — so that one manager knows what is on the GPU and makes room for
// what is asked for, the way it does for its own runners.
//
// A service is found (Scan, from what is listening and the exact command
// line, directory and environment that started it) or written into the
// services file by hand, and is then either watched — shown in /api/ps with
// what memory it holds — or adopted: the front listens on its address in its
// place, starts it on the first request on a port of its own, forwards to it
// (HTTP and WebSocket alike), and stops it once it has been idle, or when the
// GPU is wanted and it is the one unused longest. Its clients change nothing.
// Released, it is started again the way it was, on its own.
package services

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Service is one program in the services file.
type Service struct {
	Name string `json:"name"`
	// Addr is where its clients reach it, host:port, as it listened.
	Addr string `json:"addr"`
	// Command starts it. {port} and {host} stand for the address it is
	// given: its own Addr when it runs alone, a loopback port when adopted.
	Command []string          `json:"command"`
	Dir     string            `json:"dir,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
	// Health is an HTTP path that answers once it is up; empty, its port
	// accepting a connection is enough.
	Health string `json:"health,omitempty"`
	// Idle is how long an adopted service is left running unused; 0 keeps
	// it running, and the GPU is never taken from it.
	Idle Duration `json:"idle,omitempty"`
	// Quiet are paths asked only to see how it is — ComfyUI's /queue, which
	// kinhome asks to know whether the box is busy — each with the JSON the
	// front answers itself while the service is stopped. They neither start
	// it nor count as using it, so they do not keep it from its idle stop.
	Quiet map[string]json.RawMessage `json:"quiet,omitempty"`
	// Adopted: the front listens on Addr and runs it. Not adopted: watched.
	Adopted bool `json:"adopted"`
	// Need is the most memory it was seen to hold, in bytes: the room the
	// front makes for it before starting it.
	Need uint64 `json:"need,omitempty"`
	// Launchd is the launchd job it belongs to, if one: such a service is
	// launchd's to keep running, and is watched only.
	Launchd string `json:"launchd,omitempty"`
	// Log is where its output goes when kinfer starts it.
	Log string `json:"log,omitempty"`
}

// Duration is a time.Duration written as "30m" in the file.
type Duration time.Duration

func (d Duration) MarshalJSON() ([]byte, error) {
	if d == 0 {
		return []byte(`"0"`), nil
	}
	return json.Marshal(time.Duration(d).String())
}

func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return fmt.Errorf("a duration such as \"30m\": %w", err)
	}
	if s == "" || s == "0" {
		*d = 0
		return nil
	}
	v, err := time.ParseDuration(s)
	*d = Duration(v)
	return err
}

// Port is the port of Addr.
func (s Service) Port() int {
	_, p, err := net.SplitHostPort(s.Addr)
	if err != nil {
		return 0
	}
	n, _ := strconv.Atoi(p)
	return n
}

// Host is the host of Addr: "0.0.0.0" for one listening everywhere.
func (s Service) Host() string {
	h, _, err := net.SplitHostPort(s.Addr)
	if err != nil || h == "" || h == "*" {
		return "0.0.0.0"
	}
	return h
}

// Runnable reports whether the front can run it on a port of its choosing:
// its command names the port it listens on.
func (s Service) Runnable() bool {
	if len(s.Command) == 0 {
		return false
	}
	for _, a := range s.Command {
		if strings.Contains(a, "{port}") {
			return true
		}
	}
	for _, v := range s.Env {
		if strings.Contains(v, "{port}") {
			return true
		}
	}
	return false
}

// Args is its command and environment for host:port.
func (s Service) Args(host string, port int) (argv []string, env []string) {
	sub := strings.NewReplacer("{port}", strconv.Itoa(port), "{host}", host)
	for _, a := range s.Command {
		argv = append(argv, sub.Replace(a))
	}
	for k, v := range s.Env {
		env = append(env, k+"="+sub.Replace(v))
	}
	return argv, env
}

// File is the services file.
type File struct {
	Services []Service `json:"services"`
}

// DefaultPath is ~/.kinfer/services.json.
func DefaultPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".kinfer", "services.json")
}

// Load reads the services file; one that is not there is empty.
func Load(path string) (File, error) {
	var f File
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return f, nil
	}
	if err != nil {
		return f, err
	}
	if err := json.Unmarshal(b, &f); err != nil {
		return f, fmt.Errorf("%s: %w", path, err)
	}
	seen := map[string]bool{}
	for _, s := range f.Services {
		if s.Name == "" || s.Port() == 0 {
			return f, fmt.Errorf("%s: a service needs a name and an address host:port (%q, %q)", path, s.Name, s.Addr)
		}
		if seen[s.Name] {
			return f, fmt.Errorf("%s: two services are called %s", path, s.Name)
		}
		seen[s.Name] = true
	}
	return f, nil
}

// Save writes the file whole, through a temporary one, so a reader never
// sees half of it.
func Save(path string, f File) error {
	b, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Find is the service called name, and its index.
func (f File) Find(name string) (Service, int, bool) {
	for i, s := range f.Services {
		if s.Name == name {
			return s, i, true
		}
	}
	return Service{}, -1, false
}
