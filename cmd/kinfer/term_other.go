//go:build !darwin && !linux

package main

import "errors"

// Elsewhere the REPL reads whole lines, edited by the console itself.

func isTerminal(fd int) bool { return false }

func makeRaw(fd int) (func(), error) { return nil, errors.New("no raw terminal here") }

func termWidth(fd int) int { return 80 }
