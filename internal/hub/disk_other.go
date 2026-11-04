//go:build !darwin && !linux

package hub

import (
	"errors"
	"os"
)

func lock(*os.File) error { return nil }

func flush(f *os.File) error { return f.Sync() }

func free(string) (int64, error) { return 0, errors.ErrUnsupported }
