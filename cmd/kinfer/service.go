package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/services"
)

// cmdService manages the services file and asks the local front about the
// services in it; see internal/services.
func cmdService(args []string) error {
	if len(args) == 0 {
		args = []string{"help"}
	}
	switch args[0] {
	case "scan":
		return serviceScan(args[1:])
	case "list", "ls":
		return serviceList(args[1:])
	case "adopt", "release", "stop":
		return serviceAction(args[0], args[1:])
	case "add":
		return serviceAdd(args[1:])
	case "rm", "remove":
		return serviceRemove(args[1:])
	default:
		fmt.Print(serviceUsage)
		return nil
	}
}

const serviceUsage = `usage: kinfer service COMMAND

The other programs on this machine that answer on a port, taken in by kinfer
serve: watched (shown in kinfer service list and /api/ps, with what memory they
hold), or adopted — kinfer serve listens on the address in their place, starts
them when first asked for, stops them once idle or when the GPU is wanted,
and their clients change nothing. The services file is ~/.kinfer/services.json.

  scan [-save]               the programs listening here, as service entries;
                             -save writes the new ones to the file, watched
  list                       what kinfer serve knows of each, now
  adopt NAME [-idle 30m]     hand one to kinfer serve: its own process is ended,
                             and kinfer serve answers on its address; -idle 0 keeps it running
  release NAME               give it back, running on its own as it was
  stop NAME                  stop the copy kinfer serve runs; the next request starts it
  add NAME -addr HOST:PORT [-dir DIR] [-idle 30m] [-health /path] -- COMMAND…
                             write one by hand; {port} and {host} in COMMAND are its address
  rm NAME                    take one out of the file

  -host HOST:PORT            the kinfer serve to ask (default: the one running here)
`

// frontAddr is the kinfer serve front running on this machine: a kinfer
// listening that is not a runner.
func frontAddr() string {
	if h := os.Getenv("KINFER_HOST"); h != "" {
		return strings.TrimPrefix(strings.TrimPrefix(h, "http://"), "https://")
	}
	b, _ := exec.Command("lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-Fpcn").Output()
	var pid, cmd string
	for _, l := range strings.Split(string(b), "\n") {
		if l == "" {
			continue
		}
		switch l[0] {
		case 'p':
			pid = l[1:]
		case 'c':
			cmd = l[1:]
		case 'n':
			if !strings.HasPrefix(cmd, "kinfer") {
				continue
			}
			args, _ := exec.Command("ps", "-o", "command=", "-p", pid).Output()
			a := string(args)
			if strings.Contains(a, " serve") && !strings.Contains(a, "-runners=false") {
				_, port, _ := strings.Cut(strings.Replace(l[1:], "*:", "0.0.0.0:", 1), ":")
				if n, err := strconv.Atoi(port); err == nil {
					return "127.0.0.1:" + strconv.Itoa(n)
				}
			}
		}
	}
	return "127.0.0.1:11500"
}

func hostFlag(fs *flag.FlagSet) *string {
	return fs.String("host", "", "the kinfer serve to ask (default: the one running here)")
}

func askFront(host, method, path string, body any) (map[string]any, error) {
	if host == "" {
		host = frontAddr()
	}
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, "http://"+host+path, rd)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	c := http.Client{Timeout: 2 * time.Minute}
	resp, err := c.Do(req)
	if err != nil {
		return nil, fmt.Errorf("kinfer serve at %s: %w", host, err)
	}
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("kinfer serve at %s answered %s", host, resp.Status)
	}
	if e, ok := out["error"].(string); ok && resp.StatusCode >= 400 {
		return nil, fmt.Errorf("%s", e)
	}
	return out, nil
}

func serviceList(args []string) error {
	fs := flag.NewFlagSet("service list", flag.ExitOnError)
	host := hostFlag(fs)
	fs.Parse(args)
	out, err := askFront(*host, "GET", "/api/services", nil)
	if err != nil {
		return err
	}
	list, _ := out["services"].([]any)
	if len(list) == 0 {
		fmt.Println("no services; kinfer service scan -save writes the ones running here")
		return nil
	}
	fmt.Printf("%-16s %-18s %-8s %-9s %7s %8s %8s  %s\n", "NAME", "ADDRESS", "", "STATE", "PID", "HOLDS", "AT MOST", "IDLE")
	for _, x := range list {
		m, _ := x.(map[string]any)
		gb := func(k string) string {
			if v, ok := m[k].(float64); ok && v > 0 {
				return fmt.Sprintf("%.1f GB", v/1e9)
			}
			return "-"
		}
		who := "watched"
		if a, _ := m["adopted"].(bool); a {
			who = "adopted"
		}
		pid := "-"
		if p, ok := m["pid"].(float64); ok && p > 0 {
			pid = strconv.Itoa(int(p))
		}
		idle, _ := m["idle"].(string)
		note, _ := m["note"].(string)
		fmt.Printf("%-16s %-18s %-8s %-9s %7s %8s %8s  %s %s\n", m["name"], m["addr"], who, m["state"], pid, gb("footprint"), gb("need"), idle, note)
	}
	return nil
}

func serviceAction(action string, args []string) error {
	fs := flag.NewFlagSet("service "+action, flag.ExitOnError)
	host := hostFlag(fs)
	idle := fs.String("idle", "", "with adopt: how long it is left running unused (0 keeps it running)")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return fmt.Errorf("usage: kinfer service %s NAME", action)
	}
	name := args[0]
	fs.Parse(args[1:])
	out, err := askFront(*host, "POST", "/api/services/"+action, map[string]string{"name": name, "idle": *idle})
	if err != nil {
		return err
	}
	switch action {
	case "adopt":
		fmt.Printf("%s adopted: kinfer serve answers on its address, and starts it when first asked for\n", name)
	case "release":
		fmt.Printf("%s released, running on its own again (pid %v)\n", name, out["pid"])
	case "stop":
		if s, _ := out["stopped"].(bool); s {
			fmt.Printf("%s stopped; the next request starts it again\n", name)
		} else {
			fmt.Printf("%s was not running\n", name)
		}
	}
	return nil
}

func serviceAdd(args []string) error {
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return fmt.Errorf("usage: kinfer service add NAME -addr HOST:PORT -- COMMAND…")
	}
	name := args[0]
	rest := args[1:]
	var cmd []string
	for i, a := range rest {
		if a == "--" {
			cmd, rest = rest[i+1:], rest[:i]
			break
		}
	}
	fs := flag.NewFlagSet("service add", flag.ExitOnError)
	addr := fs.String("addr", "", "where its clients reach it, host:port")
	dir := fs.String("dir", "", "the folder it runs in")
	idle := fs.Duration("idle", 30*time.Minute, "how long it is left running unused once adopted (0 keeps it running)")
	health := fs.String("health", "", "an HTTP path that answers once it is up")
	fs.Parse(rest)
	svc := services.Service{Name: name, Addr: *addr, Command: cmd, Dir: *dir, Idle: services.Duration(*idle), Health: *health}
	if svc.Port() == 0 || len(cmd) == 0 {
		return fmt.Errorf("a service needs -addr host:port and a command after --")
	}
	if !svc.Runnable() {
		fmt.Fprintln(os.Stderr, "note: the command names no {port}, so it can be watched but not adopted")
	}
	path := services.DefaultPath()
	f, err := services.Load(path)
	if err != nil {
		return err
	}
	if _, _, ok := f.Find(name); ok {
		return fmt.Errorf("%s is in %s already", name, path)
	}
	f.Services = append(f.Services, svc)
	if err := services.Save(path, f); err != nil {
		return err
	}
	fmt.Printf("%s added to %s, watched; kinfer service adopt %s hands it to kinfer serve\n", name, path, name)
	return nil
}

func serviceRemove(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: kinfer service rm NAME")
	}
	path := services.DefaultPath()
	f, err := services.Load(path)
	if err != nil {
		return err
	}
	svc, i, ok := f.Find(args[0])
	if !ok {
		return fmt.Errorf("no service called %s", args[0])
	}
	if svc.Adopted {
		return fmt.Errorf("%s is adopted: kinfer service release %s first, or it is left stopped", svc.Name, svc.Name)
	}
	f.Services = append(f.Services[:i], f.Services[i+1:]...)
	if err := services.Save(path, f); err != nil {
		return err
	}
	fmt.Printf("%s taken out of %s\n", args[0], path)
	return nil
}

func serviceScan(args []string) error {
	save := len(args) > 0 && args[0] == "-save"
	found, err := services.Scan()
	if err != nil {
		return err
	}
	for _, f := range found {
		state := "can be adopted"
		if f.Why != "" {
			state = "watched only: " + f.Why
		}
		fmt.Printf("%-14s %-17s pid %-6d %6.1f GB  %s\n", f.Name, f.Addr, f.PID, float64(f.Footprint)/1e9, state)
		fmt.Printf("    %s\n", strings.Join(f.Command, " "))
		if f.Dir != "" {
			fmt.Printf("    in %s\n", f.Dir)
		}
		if len(f.Env) > 0 {
			var kv []string
			for k, v := range f.Env {
				if k != "PATH" {
					kv = append(kv, k+"="+v)
				}
			}
			if len(kv) > 0 {
				fmt.Printf("    with %s\n", strings.Join(kv, " "))
			}
		}
	}
	if !save {
		return nil
	}
	path := services.DefaultPath()
	file, err := services.Load(path)
	if err != nil {
		return err
	}
	added := 0
	for _, f := range found {
		if _, _, ok := file.Find(f.Name); ok || hasAddr(file, f.Addr) {
			continue
		}
		file.Services = append(file.Services, f.Service)
		added++
	}
	if err := services.Save(path, file); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%d new in %s, watched; kinfer service adopt NAME hands one to kinfer serve\n", added, path)
	return nil
}

func hasAddr(f services.File, addr string) bool {
	for _, s := range f.Services {
		if s.Addr == addr {
			return true
		}
	}
	return false
}
