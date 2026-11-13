//go:build windows

package services

import "os"

// lockFile on Windows takes no lock: services are run by the Macs here, and
// one server on a Windows machine is assumed to be the only one.
func lockFile(path string) (*os.File, int, error) { return nil, 0, nil }
