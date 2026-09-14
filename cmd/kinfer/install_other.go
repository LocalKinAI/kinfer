//go:build !darwin

package main

import (
	"fmt"
	"runtime"
)

// Only launchd is wired up so far. A systemd --user unit is the same idea and
// the same twenty lines; it is not here because nothing has run kinfer as a
// service on Linux yet, and a unit nobody has started is a unit nobody has
// tested.

func installService(service) (string, error) {
	return "", fmt.Errorf("kinfer install is macOS-only for now (this is %s); run kinfer serve under your service manager", runtime.GOOS)
}

func uninstallService() (string, error) {
	return "", fmt.Errorf("kinfer uninstall is macOS-only for now (this is %s)", runtime.GOOS)
}
