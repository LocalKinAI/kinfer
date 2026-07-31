// Package nativelib makes kinfer a single self-contained binary.
//
// The problem it solves: gollama.cpp declares `//go:embed libs/**` inside its
// OWN package directory, and the Go module cache is read-only — so a downstream
// project can never populate it. Building kinfer with libs/ sitting in the
// project root changes the binary size by exactly zero bytes (measured in
// Phase 0). Without this package, every user would have to fetch the native
// libraries separately, which is precisely the dependency mess kinfer exists to
// avoid.
//
// So kinfer embeds the libraries itself, unpacks them to a per-version cache
// directory on first run, and points gollama at that directory before any call
// into llama.cpp.
//
// Populate libs/ before building:
//
//	go run github.com/dianlight/gollama.cpp/cmd/gollama-download -download -copy-libs
package nativelib

import (
	"crypto/sha256"
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	gollama "github.com/dianlight/gollama.cpp"
)

// all: is required — without it, Go's embed skips files beginning with "." or
// "_", and skips nothing else silently, which makes a missing library look like
// a runtime error rather than a build error.
//
//go:embed all:libs
var embedded embed.FS

var (
	once     sync.Once
	prepared string
	prepErr  error
)

// Prepare unpacks the embedded native libraries and configures gollama to load
// them. It is safe to call repeatedly; the work happens once.
//
// Returns the directory the libraries were unpacked into.
func Prepare() (string, error) {
	once.Do(func() {
		prepared, prepErr = prepare()
	})
	return prepared, prepErr
}

func prepare() (string, error) {
	platform, err := platformDir()
	if err != nil {
		return "", err
	}

	files, err := fs.ReadDir(embedded, path("libs", platform))
	if err != nil {
		return "", fmt.Errorf("no embedded libraries for %s/%s: run `gollama-download -download -copy-libs` and rebuild: %w",
			runtime.GOOS, runtime.GOARCH, err)
	}

	dest, err := cacheDir(platform)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return "", fmt.Errorf("create library cache %s: %w", dest, err)
	}

	for _, f := range files {
		if f.IsDir() {
			continue
		}
		if err := extract(platform, f.Name(), dest); err != nil {
			return "", err
		}
	}

	return dest, nil
}

// Load unpacks the embedded libraries and initialises the llama.cpp backend
// from them. Call this instead of gollama.Backend_init.
//
// Getting the loader to use OUR copy took some doing, because two obvious
// routes are both dead ends in gollama.cpp v0.2.2:
//
//   - Config.LibraryPath is accepted and then ignored: ApplyConfig unloads the
//     current library but never stores the path, so the next load falls back to
//     the default search. (Worth a PR upstream.)
//   - Setting DYLD_LIBRARY_PATH from inside the process is too late — the
//     dynamic linker reads it at exec time, so os.Setenv has no effect on a
//     later dlopen.
//
// What does work: the loader calls dlopen with a BARE name ("libllama.dylib"),
// and a bare name resolves against the current working directory. So chdir into
// the unpacked directory for the duration of Backend_init, then restore.
//
// The chdir is process-wide, so this must not race with other goroutines doing
// relative-path I/O. In practice Backend_init happens once during startup,
// before anything else runs, and the directory is restored before returning.
func Load() error {
	dir, err := Prepare()
	if err != nil {
		return err
	}

	prev, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("record working directory: %w", err)
	}
	if err := os.Chdir(dir); err != nil {
		return fmt.Errorf("enter library directory %s: %w", dir, err)
	}
	defer func() {
		// A failure to restore would silently break every relative path the
		// program uses afterwards, so it is worth surfacing loudly.
		if cerr := os.Chdir(prev); cerr != nil {
			panic(fmt.Sprintf("kinfer: could not restore working directory %s: %v", prev, cerr))
		}
	}()

	if err := gollama.Backend_init(); err != nil {
		return fmt.Errorf("initialise llama.cpp backend from %s: %w", dir, err)
	}
	return nil
}

// extract writes one embedded library to disk, skipping the write when an
// identical file is already there. Content is hashed rather than trusting
// mtime, so a half-written file from a killed process gets replaced.
func extract(platform, name, dest string) error {
	data, err := embedded.ReadFile(path("libs", platform, name))
	if err != nil {
		return fmt.Errorf("read embedded %s: %w", name, err)
	}

	target := filepath.Join(dest, name)
	if existing, err := os.ReadFile(target); err == nil && sameContent(existing, data) {
		return nil
	}

	// Write to a temp file in the same directory, then rename. Two kinfer
	// processes starting at once must not see a partially written dylib.
	tmp, err := os.CreateTemp(dest, "."+name+".*")
	if err != nil {
		return fmt.Errorf("stage %s: %w", name, err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once the rename succeeds

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write %s: %w", name, err)
	}
	if err := tmp.Chmod(0o755); err != nil {
		tmp.Close()
		return fmt.Errorf("chmod %s: %w", name, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close %s: %w", name, err)
	}
	if err := os.Rename(tmpName, target); err != nil {
		return fmt.Errorf("install %s: %w", name, err)
	}
	return nil
}

func sameContent(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	ha, hb := sha256.Sum256(a), sha256.Sum256(b)
	return ha == hb
}

// platformDir finds the embedded directory for this OS/arch. Names look like
// "darwin_arm64_b6862" — the llama.cpp build number is part of the name, and it
// is not known at compile time, so match on the prefix.
func platformDir() (string, error) {
	prefix := fmt.Sprintf("%s_%s_", runtime.GOOS, runtime.GOARCH)

	entries, err := fs.ReadDir(embedded, "libs")
	if err != nil {
		return "", fmt.Errorf("no embedded libs directory: %w", err)
	}
	for _, e := range entries {
		if e.IsDir() && strings.HasPrefix(e.Name(), prefix) {
			return e.Name(), nil
		}
	}

	var have []string
	for _, e := range entries {
		if e.IsDir() {
			have = append(have, e.Name())
		}
	}
	return "", fmt.Errorf("this binary has no libraries for %s/%s (embedded: %v)",
		runtime.GOOS, runtime.GOARCH, have)
}

// cacheDir returns a per-platform, per-build directory under the user's cache
// root. Including the platform string means upgrading llama.cpp lands in a new
// directory instead of overwriting libraries a running process has mapped.
func cacheDir(platform string) (string, error) {
	root, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("locate user cache dir: %w", err)
	}
	return filepath.Join(root, "kinfer", "lib", platform), nil
}

func prependEnv(key, dir string) {
	cur := os.Getenv(key)
	if cur == "" {
		os.Setenv(key, dir)
		return
	}
	for _, p := range filepath.SplitList(cur) {
		if p == dir {
			return
		}
	}
	os.Setenv(key, dir+string(filepath.ListSeparator)+cur)
}

// path joins with forward slashes — embed.FS always uses them, regardless of
// the host platform's separator.
func path(parts ...string) string {
	return strings.Join(parts, "/")
}

// ErrNoLibraries reports that this build has no native libraries for the
// running platform.
var ErrNoLibraries = errors.New("no embedded native libraries for this platform")
