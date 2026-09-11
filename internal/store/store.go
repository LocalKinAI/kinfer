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
	"regexp"
	"sort"
	"strings"
	"time"
)

// shardSuffix matches the "-00002-of-00005" a split GGUF carries.
//
// Anything past roughly 50 GB arrives in pieces, because that is the largest
// file Hugging Face will serve. llama.cpp handles them — given the first piece
// it finds the rest by name. What it cannot do is stop a directory of pieces
// from looking like several separate models, which is this package's job.
var shardSuffix = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

// shardOf reports a file's place in a split model: the name the whole set
// should be known by, and which piece this is. An index of 0 means the file is
// not a shard.
func shardOf(filename string) (base string, index int) {
	m := shardSuffix.FindStringSubmatch(filename)
	if m == nil {
		return "", 0
	}
	n := 0
	for _, c := range m[1] {
		n = n*10 + int(c-'0')
	}
	return filename[:len(filename)-len(m[0])], n
}

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

	// Source is where the file lives. A borrowed model is read in place and
	// never written to.
	Source Source
}

// Store is a directory of models.
type Store struct {
	root string

	// borrow says whether to also offer the models Ollama has on this machine.
	// Only the default store does. OpenAt names one directory and means it —
	// reaching into another program's store from there would surprise a caller
	// who asked about a path, and would make a test depend on whatever the
	// machine running it happens to have installed.
	borrow bool
}

// Open returns the store at ~/.kinfer/models, creating it if needed.
func Open() (*Store, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("locate home directory: %w", err)
	}
	s, err := OpenAt(filepath.Join(home, ".kinfer", "models"))
	if err != nil {
		return nil, err
	}
	s.borrow = true
	return s, nil
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

// Borrow turns on offering the models Ollama has on this machine, which Open
// does and OpenAt does not. Exported so a test can build a store over a fixture
// directory and still exercise the borrowing path — the alternative is a store
// that only ever sees its own files, which is not the store anything runs
// against.
func (s *Store) Borrow(on bool) { s.borrow = on }

// List returns every model, newest first.
func (s *Store) List() ([]Model, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", s.root, err)
	}

	// A split model is one model. Gather its pieces before building the list,
	// or it appears once per piece, each showing a fraction of its true size,
	// and asking for it by name is ambiguous.
	shards := map[string]*Model{}

	var out []Model
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
			continue
		}
		path := filepath.Join(s.root, e.Name())

		// os.Stat, not DirEntry.Info: the latter does not follow symlinks, and
		// a symlinked model would report the size of the link itself — a
		// hundred bytes where the model is twenty gigabytes. Linking a model
		// into the store is a reasonable thing to do; it is how one file gets
		// shared with another runtime instead of copied.
		info, err := os.Stat(path)
		if err != nil {
			continue // vanished, or a link with nothing at the end of it
		}

		if base, idx := shardOf(e.Name()); idx > 0 {
			m := shards[base]
			if m == nil {
				m = &Model{Name: base, Source: FromKinfer}
				shards[base] = m
			}
			m.Size += info.Size()
			// The first piece is the handle: llama.cpp opens it and finds the
			// rest itself.
			if idx == 1 {
				m.Path = path
			}
			if info.ModTime().After(m.Modified) {
				m.Modified = info.ModTime()
			}
			continue
		}

		out = append(out, Model{
			Name:     nameOf(e.Name()),
			Path:     path,
			Size:     info.Size(),
			Modified: info.ModTime(),
			Source:   FromKinfer,
		})
	}

	for _, m := range shards {
		// A set missing its first piece cannot be loaded; listing it would only
		// move the failure to a more confusing place.
		if m.Path != "" {
			out = append(out, *m)
		}
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Modified.After(out[j].Modified) })
	return out, nil
}

// All is every model kinfer can load: its own, plus the ones Ollama already has
// on this machine. A name present in both belongs to kinfer's copy — that is
// the one someone put there deliberately.
func (s *Store) All() ([]Model, error) {
	own, err := s.List()
	if err != nil {
		return nil, err
	}
	seen := make(map[string]bool, len(own))
	for _, m := range own {
		seen[strings.ToLower(m.Name)] = true
	}
	if s.borrow {
		for _, m := range OllamaModels() {
			if !seen[strings.ToLower(m.Name)] {
				own = append(own, m)
			}
		}
	}
	return own, nil
}

// Resolve turns a user-supplied name into a path.
//
// Accepts, in order of preference: an exact name, an exact filename, a path
// that exists as given, or a unique case-insensitive prefix. The prefix rule is
// what makes `kinfer run qwen` usable; it deliberately fails when ambiguous
// rather than picking one, because silently loading the wrong 7B model is a
// worse outcome than an error message.
// Lookup returns the model a name refers to.
//
// Resolve answers "where is its file", which a remote model does not have.
// Anything that needs to know what kind of thing a name is — before deciding
// whether to load it or forward it — asks here.
func (s *Store) Lookup(name string) (Model, error) {
	models, err := s.All()
	if err != nil {
		return Model{}, err
	}
	lower := strings.ToLower(name)
	for _, m := range models {
		if strings.ToLower(m.Name) == lower {
			return m, nil
		}
	}
	// Ollama's own clients accept a bare name for a :latest tag.
	for _, m := range models {
		if strings.ToLower(m.Name) == lower+":latest" {
			return m, nil
		}
	}
	return Model{}, fmt.Errorf("no model named %q", name)
}

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

	models, err := s.All()
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
		// A borrowed model must not be deletable by a kinfer command. That
		// store belongs to Ollama and is not kinfer's to damage.
		sep := string(filepath.Separator)
		if strings.Contains(path, sep+"blobs"+sep) {
			return fmt.Errorf("%s belongs to Ollama; remove it with `ollama rm %s` instead", name, name)
		}
		return fmt.Errorf("%s is outside the model directory; delete it yourself", path)
	}

	// A split model is deleted whole. Removing only the piece that was resolved
	// would leave tens of gigabytes behind that nothing can load.
	files, err := s.Shards(path)
	if err != nil {
		return err
	}
	for _, f := range files {
		if err := os.Remove(f); err != nil {
			return err
		}
	}
	return nil
}

// Shards lists every file a model is made of — its own path, unless it is
// split, in which case all the pieces in name order.
func (s *Store) Shards(path string) ([]string, error) {
	base, idx := shardOf(filepath.Base(path))
	if idx == 0 {
		return []string{path}, nil
	}
	dir := filepath.Dir(path)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if b, i := shardOf(e.Name()); i > 0 && b == base {
			out = append(out, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(out)
	return out, nil
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
