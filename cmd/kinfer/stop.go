package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// cmdStop stops the background daemon `kinfer run` started, and with it the
// model it holds and any worker it borrows.
//
// It finds the daemon by asking it: /health says its pid. A daemon from
// v0.1.2 or earlier does not, and is found by the port it listens on, if that
// process is kinfer. The serve `kinfer install` keeps running is left alone,
// since launchd would start it again at once; `kinfer uninstall` stops that.
func cmdStop(args []string) error {
	fs := flag.NewFlagSet("stop", flag.ExitOnError)
	addr := fs.String("addr", defaultAddr, "the daemon's address")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("stop takes no arguments; %q was one (-addr names another daemon)", fs.Arg(0))
	}
	base := daemonURL(*addr)
	h := daemonHealth(base)
	if !h.alive {
		fmt.Printf("no kinfer daemon on %s\n", *addr)
		return nil
	}
	pid := h.pid
	if pid == 0 {
		pid = kinferListening(*addr)
	}
	if pid == 0 {
		return fmt.Errorf("a server answers on %s, but which process it is cannot be told; stop it by hand", *addr)
	}
	if pid == servicePID() {
		return fmt.Errorf("the server on %s is the one kinfer install keeps running, which launchd would start again; kinfer uninstall stops it", *addr)
	}
	p, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	// serve stops at SIGTERM once its requests are done; one still running
	// after that long is not stopping.
	if err := p.Signal(syscall.SIGTERM); err != nil {
		return fmt.Errorf("stop the daemon (pid %d): %w", pid, err)
	}
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if !daemonAlive(base) {
			fmt.Printf("stopped the kinfer daemon on %s (pid %d)\n", *addr, pid)
			return nil
		}
		time.Sleep(200 * time.Millisecond)
	}
	_ = p.Kill()
	fmt.Printf("the kinfer daemon on %s (pid %d) did not stop within 20 s, and was killed\n", *addr, pid)
	return nil
}

// kinferListening is the pid of the kinfer process listening on addr's port,
// or 0: lsof finds the listener, ps says whether it is kinfer.
func kinferListening(addr string) int {
	_, port, err := net.SplitHostPort(addr)
	if err != nil {
		return 0
	}
	out, err := exec.Command("lsof", "-nP", "-t", "-iTCP:"+port, "-sTCP:LISTEN").Output()
	if err != nil {
		return 0
	}
	for _, f := range strings.Fields(string(out)) {
		pid, err := strconv.Atoi(f)
		if err != nil {
			continue
		}
		comm, err := exec.Command("ps", "-o", "comm=", "-p", f).Output()
		if err == nil && strings.Contains(string(comm), "kinfer") {
			return pid
		}
	}
	return 0
}
