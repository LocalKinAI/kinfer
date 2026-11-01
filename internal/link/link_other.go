//go:build !darwin

package link

import (
	"context"
	"time"
)

// FindBridge finds nothing off macOS: only a Mac's Thunderbolt Bridge is known.
func FindBridge() (Bridge, error) { return Bridge{}, ErrNoBridge }

// Answer has no bridge to answer on off macOS.
func Answer(ctx context.Context, b Bridge, addr string) error { return ErrNoBridge }

// Find has no bridge to ask on off macOS.
func Find(b Bridge, wait time.Duration) ([]string, error) { return nil, ErrNoBridge }

// CablePeers has no bridge to look on off macOS.
func CablePeers(b Bridge) ([]string, error) { return nil, ErrNoBridge }
