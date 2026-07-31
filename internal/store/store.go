// Package store manages the local model directory.
//
// Models are kept as plain files under ~/.kinfer/models with the names they had
// on Hugging Face. That is a deliberate difference from Ollama, which stores
// blobs named by digest: when something goes wrong at 2am you want `ls` to tell
// you what you have, and you want to be able to copy a model to another machine
// with `scp`.
package store

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Model is one GGUF file on disk.
type Model struct {
	// Name is how the user refers to it — the filename without .gguf.
	Name string

	// Path is the absolute location.
	Path string

	// Size in bytes.
	Size int64

	// Modified is the file's mtime.
	Modified time.Time
}

// Store is a directory of models.
type Store struct{ root string }

// Open returns the store at ~/.kinfer/models, creating it if needed.
func Open() (*Store, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("locate home directory: %w", err)
	}
	return OpenAt(filepath.Join(home, ".kinfer", "models"))
}

// OpenAt returns a store rooted at dir.
func OpenAt(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create model directory %s: %w", dir, err)
	}
	return &Store{root: dir}, nil
}

// Root is the directory models live in.
func (s *Store) Root() string { return s.root }

// List returns every model, newest first.
func (s *Store) List() ([]Model, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", s.root, err)
	}

	var out []Model
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue // vanished between ReadDir and Info; not worth failing over
		}
		out = append(out, Model{
			Name:     nameOf(e.Name()),
			Path:     filepath.Join(s.root, e.Name()),
			Size:     info.Size(),
			Modified: info.ModTime(),
		})
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Modified.After(out[j].Modified) })
	return out, nil
}

// Resolve turns a user-supplied name into a path.
//
// Accepts, in order of preference: an exact name, an exact filename, a path
// that exists as given, or a unique case-insensitive prefix. The prefix rule is
// what makes `kinfer run qwen` usable; it deliberately fails when ambiguous
// rather than picking one, because silently loading the wrong 7B model is a
// worse outcome than an error message.
func (s *Store) Resolve(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("no model specified")
	}

	// An explicit path wins — it lets people run a model from anywhere.
	if strings.ContainsRune(name, os.PathSeparator) || strings.HasSuffix(name, ".gguf") {
		if _, err := os.Stat(name); err == nil {
			return name, nil
		}
	}

	models, err := s.List()
	if err != nil {
		return "", err
	}

	lower := strings.ToLower(name)
	for _, m := range models {
		if strings.EqualFold(m.Name, name) || strings.EqualFold(filepath.Base(m.Path), name) {
			return m.Path, nil
		}
	}

	var matches []Model
	for _, m := range models {
		if strings.HasPrefix(strings.ToLower(m.Name), lower) {
			matches = append(matches, m)
		}
	}
	switch len(matches) {
	case 1:
		return matches[0].Path, nil
	case 0:
		if len(models) == 0 {
			return "", fmt.Errorf("no models installed — try `kinfer pull <repo>`")
		}
		return "", fmt.Errorf("no model matches %q (have: %s)", name, strings.Join(names(models), ", "))
	default:
		return "", fmt.Errorf("%q is ambiguous: %s", name, strings.Join(names(matches), ", "))
	}
}

// Remove deletes a model.
func (s *Store) Remove(name string) error {
	path, err := s.Resolve(name)
	if err != nil {
		return err
	}
	// Refuse to delete outside the store, so a path-shaped argument cannot turn
	// `kinfer rm` into a general-purpose file remover.
	rel, err := filepath.Rel(s.root, path)
	if err != nil || strings.HasPrefix(rel, "..") {
		return fmt.Errorf("%s is outside the model directory; delete it yourself", path)
	}
	return os.Remove(path)
}

func nameOf(filename string) string {
	return strings.TrimSuffix(filename, filepath.Ext(filename))
}

func names(models []Model) []string {
	out := make([]string, len(models))
	for i, m := range models {
		out[i] = m.Name
	}
	return out
}

// HumanSize formats a byte count for display.
func HumanSize(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}
