package main

import (
	"errors"
	"os"
	"os/exec"
)

// terminate ends a kinfer process at once: Windows cannot send another process
// a signal it could finish its requests on.
func terminate(p *os.Process) error { return p.Kill() }

// execAgent runs the agent and waits for it: Windows has no exec, so kinfer
// stays as its parent and leaves with its exit status.
func execAgent(bin string, argv, env []string) error {
	cmd := exec.Command(bin, argv[1:]...)
	cmd.Env = env
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	err := cmd.Run()
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		os.Exit(exit.ExitCode())
	}
	return err
}
