//go:build !darwin

package services

import "errors"

func procArgs(pid int) (argv, env []string, err error) {
	return nil, nil, errors.New("reading another process's arguments is written for macOS only")
}

func footprint(pid int) uint64 { return 0 }
