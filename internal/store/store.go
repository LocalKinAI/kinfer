// Package store manages the local model directory.
//
// Models are kept as plain files under ~/.kinfer/models with the names they had
// on Hugging Face. That is a deliberate difference from Ollama, which stores
// blobs named by digest: when something goes wrong at 2am you want `ls` to tell
// you what you have, and you want to be able to copy a model to another machine
// with `scp`.
package store

import (
	"encoding/json"
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
// should be known by, which piece this is, and of how many. An index of 0
// means the file is not a shard.
func shardOf(filename string) (base string, index, of int) {
	m := shardSuffix.FindStringSubmatch(filename)
	if m == nil {
		return "", 0, 0
	}
	number := func(digits string) int {
		n := 0
		for _, c := range digits {
			n = n*10 + int(c-'0')
		}
		return n
	}
	return filename[:len(filename)-len(m[0])], number(m[1]), number(m[2])
}

// Model is one GGUF file on disk, or one directory of safetensors that kinfer
// serves on MLX.
type Model struct {
	// Name is how the user refers to it — the filename without .gguf, or the
	// directory's name.
	Name string

	// Path is the absolute location: the file, the first piece of a split
	// model, or the directory.
	Path string

	// Format is "gguf", or "safetensors" for a directory.
	Format string

	// EditOf names the model an edited one was made from (kinlab edit), or
	// is empty.
	EditOf string

	// Size in bytes.
	Size int64

	// Modified is the file's mtime.
	Modified time.Time

	// Source is where the file lives. A borrowed model is read in place and
	// never written to.
	Source Source

	// Projector is a vision model's projector, the file it reads pictures
	// with: a layer of its own in Ollama's store, and NAME.mmproj.gguf beside
	// one of kinfer's. Empty for a model that reads only text.
	Projector string
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

// List returns every model, newest first.
func (s *Store) List() ([]Model, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", s.root, err)
	}

	// A split model is one model. Gather its pieces before building the list,
	// or it appears once per piece, each showing a fraction of its true size,
	// and asking for it by name is ambiguous.
	type set struct {
		Model
		have, of int // pieces here, and of how many
	}
	shards := map[string]*set{}

	var out []Model
	for _, e := range entries {
		path := filepath.Join(s.root, e.Name())
		if fi, err := os.Stat(path); err == nil && fi.IsDir() {
			if d := readModelDir(path); d.complete() {
				out = append(out, Model{Name: e.Name(), Path: path, Format: "safetensors", EditOf: editOf(path),
					Size: d.size, Modified: d.modified, Source: FromKinfer})
			}
			continue
		}
		if !strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") || isProjector(e.Name()) {
			continue
		}

		// os.Stat, not DirEntry.Info: the latter does not follow symlinks, and
		// a symlinked model would report the size of the link itself — a
		// hundred bytes where the model is twenty gigabytes. Linking a model
		// into the store is a reasonable thing to do; it is how one file gets
		// shared with another runtime instead of copied.
		info, err := os.Stat(path)
		if err != nil {
			continue // vanished, or a link with nothing at the end of it
		}

		if base, idx, of := shardOf(e.Name()); idx > 0 {
			m := shards[base]
			if m == nil {
				m = &set{Model: Model{Name: base, Format: "gguf", Source: FromKinfer}, of: of}
				shards[base] = m
			}
			m.have++
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
			Name:      nameOf(e.Name()),
			Path:      path,
			Format:    "gguf",
			Size:      info.Size(),
			Modified:  info.ModTime(),
			Source:    FromKinfer,
			Projector: projectorBeside(path),
		})
	}

	for _, m := range shards {
		// A set missing a piece cannot be loaded — llama.cpp opens the first
		// and looks for the rest beside it — so listing it would only move the
		// failure to a more confusing place. Unfinished shows it instead.
		if m.Path != "" && m.have == m.of {
			out = append(out, m.Model)
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
	// A name is the model of exactly that name before one that has it but for
	// case: an MLX directory and a GGUF of the same weights are often named so
	// (Qwen3.5-0.8B-bf16, Qwen3.5-0.8B-BF16).
	for _, m := range models {
		if m.Name == name {
			return m, nil
		}
	}
	lower := strings.ToLower(name)
	var folded []Model
	for _, m := range models {
		if strings.ToLower(m.Name) == lower {
			folded = append(folded, m)
		}
	}
	switch len(folded) {
	case 0:
	case 1:
		return folded[0], nil
	default:
		return Model{}, fmt.Errorf("%q is ambiguous: %s", name, strings.Join(names(folded), ", "))
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

	// Exactly the name first, then the name but for case, as Lookup does; a
	// pull of exactly the name, unfinished, is that model too.
	for _, m := range models {
		if m.Name == name || filepath.Base(m.Path) == name {
			return m.Path, nil
		}
	}
	if u, ok := s.unfinished(name); ok && u.Name == name {
		return "", fmt.Errorf("%s has not finished arriving — the kinfer pull that fetched it goes on when run again", u.Name)
	}
	lower := strings.ToLower(name)
	var folded []Model
	for _, m := range models {
		if strings.EqualFold(m.Name, name) || strings.EqualFold(filepath.Base(m.Path), name) {
			folded = append(folded, m)
		}
	}
	switch len(folded) {
	case 0:
	case 1:
		return folded[0].Path, nil
	default:
		return "", fmt.Errorf("%q is ambiguous: %s", name, strings.Join(names(folded), ", "))
	}
	// Ollama's clients send a bare name for its :latest tag. It means that
	// model, before any other the name begins: "qwen3.8" on the box is
	// qwen3.8:latest, not Qwen3.8-Flash-Next or qwen3.8-27b.
	for _, m := range models {
		if strings.EqualFold(m.Name, name+":latest") {
			return m.Path, nil
		}
	}

	// The exact name of a model a pull has not finished means that model, not
	// another the name is the start of.
	if u, ok := s.unfinished(name); ok {
		return "", fmt.Errorf("%s has not finished arriving — the kinfer pull that fetched it goes on when run again", u.Name)
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

// Remove deletes a model, and whatever a pull of it left unfinished. The name
// of an unfinished model alone removes what is here of it.
func (s *Store) Remove(name string) error {
	if u, ok := s.unfinished(name); ok {
		if m, err := s.Lookup(name); err != nil || (u.Name == name && m.Name != name) {
			return removeAll(u.Files)
		}
	}
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

	// A directory goes whole, with whatever else a download left in it.
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		return os.RemoveAll(path)
	}

	// A split model is deleted whole. Removing only the piece that was resolved
	// would leave tens of gigabytes behind that nothing can load.
	files, err := s.Shards(path)
	if err != nil {
		return err
	}
	if err := removeAll(files); err != nil {
		return err
	}
	model, idx, _ := shardOf(filepath.Base(path))
	if idx == 0 {
		model = nameOf(filepath.Base(path))
	}
	// Its projector goes with it — the one kinfer keeps beside it, never a
	// file of Ollama's.
	if p := projectorBeside(filepath.Join(filepath.Dir(path), model+".gguf")); p != "" {
		if err := os.Remove(p); err != nil {
			return err
		}
	}
	if u, ok := s.unfinished(model); ok {
		return removeAll(u.Files)
	}
	return nil
}

func removeAll(files []string) error {
	for _, f := range files {
		remove := os.Remove
		if fi, err := os.Lstat(f); err == nil && fi.IsDir() {
			remove = os.RemoveAll // a directory model's, found in the store
		}
		if err := remove(f); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

// Shards lists every file a model is made of — its own path, unless it is
// split, in which case all the pieces in name order.
func (s *Store) Shards(path string) ([]string, error) {
	base, idx, _ := shardOf(filepath.Base(path))
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
		if b, i, _ := shardOf(e.Name()); i > 0 && b == base {
			out = append(out, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(out)
	return out, nil
}

// Unfinished is a model a pull has not finished: a file still arriving as
// name.part, or a split model some of whose pieces are not here. It cannot be
// loaded, so List leaves it out; it is listed here so that it is finished or
// removed rather than forgotten, holding tens of gigabytes.
type Unfinished struct {
	Name     string
	Files    []string // all of it on disk, the pull's records with it
	Size     int64    // bytes of it on disk
	Total    int64    // its size once whole; 0 when nothing here says
	Modified time.Time
}

// Unfinished lists the unfinished models in the store, newest first.
func (s *Store) Unfinished() ([]Unfinished, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", s.root, err)
	}
	type model struct {
		Unfinished
		pieces    []string // its pieces here whole
		piecesLen int64
		have, of  int // pieces here whole, and of how many
		arriving  int // files of it still name.part
		unsized   bool
	}
	models := map[string]*model{}
	var out []Unfinished
	get := func(name string, of int) *model {
		m := models[name]
		if m == nil {
			m = &model{Unfinished: Unfinished{Name: name}, of: of}
			models[name] = m
		}
		return m
	}

	for _, e := range entries {
		path := filepath.Join(s.root, e.Name())
		info, err := os.Stat(path)
		if err != nil {
			continue
		}
		if info.IsDir() {
			// A directory a pull of an MLX model is filling, or stopped
			// filling. How big it will be, nothing here says until the last
			// file has started.
			if d := readModelDir(path); d.unfinished() {
				out = append(out, Unfinished{Name: e.Name(), Files: []string{path}, Size: d.size, Modified: d.modified})
			}
			continue
		}
		if final, ok := strings.CutSuffix(e.Name(), ".part"); ok && strings.HasSuffix(strings.ToLower(final), ".gguf") {
			name, idx, of := shardOf(final)
			if idx == 0 {
				name, of = nameOf(final), 1
			}
			m := get(name, of)
			m.arriving++
			m.Files = append(m.Files, path)
			for _, kept := range []string{path + ".json", path + ".json.tmp"} {
				if _, err := os.Lstat(kept); err == nil {
					m.Files = append(m.Files, kept)
				}
			}
			if done, size, ok := readRecord(path + ".json"); ok {
				m.Size += done
				m.Total += size
			} else {
				m.unsized = true
			}
			if info.ModTime().After(m.Modified) {
				m.Modified = info.ModTime()
			}
			continue
		}
		if name, idx, of := shardOf(e.Name()); idx > 0 {
			m := get(name, of)
			m.have++
			m.pieces = append(m.pieces, path)
			m.piecesLen += info.Size()
			if info.ModTime().After(m.Modified) {
				m.Modified = info.ModTime()
			}
		}
	}

	for _, m := range models {
		whole := m.have == m.of
		if whole && m.arriving == 0 {
			continue // a model, in List
		}
		if !whole {
			m.Files = append(m.Files, m.pieces...)
			m.Size += m.piecesLen
			m.Total += m.piecesLen
		}
		// A piece neither here nor arriving has a size nothing here says.
		if m.unsized || (!whole && m.have+m.arriving < m.of) {
			m.Total = 0
		}
		sort.Strings(m.Files)
		out = append(out, m.Unfinished)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Modified.After(out[j].Modified) })
	return out, nil
}

// unfinished finds an unfinished model by its name, exactly that name first.
func (s *Store) unfinished(name string) (Unfinished, bool) {
	all, _ := s.Unfinished()
	for _, u := range all {
		if u.Name == name {
			return u, true
		}
	}
	for _, u := range all {
		if strings.EqualFold(u.Name, name) {
			return u, true
		}
	}
	return Unfinished{}, false
}

// readRecord reads what kinfer pull keeps beside a name.part as it fills it
// (hub's record): the file's size, and how much of each piece has arrived.
func readRecord(path string) (done, size int64, ok bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, false
	}
	var rec struct {
		Size int64   `json:"size"`
		Done []int64 `json:"done"`
	}
	if json.Unmarshal(b, &rec) != nil {
		return 0, 0, false
	}
	for _, n := range rec.Done {
		done += n
	}
	return done, rec.Size, true
}

// modelDir is what a directory in the store holds of a model kinfer serves on
// MLX, laid out as Hugging Face publishes one: config.json, tokenizer.json,
// and the weights in safetensors, one file or the pieces
// model.safetensors.index.json lists.
type modelDir struct {
	config, tokenizer bool
	weights           int   // safetensors files here
	missing           int   // pieces the index names that are not here
	arriving          int   // files still name.part
	size              int64 // bytes here; of a file arriving, what has arrived
	modified          time.Time
}

func (d modelDir) complete() bool {
	return d.config && d.tokenizer && d.weights > 0 && d.missing == 0 && d.arriving == 0
}

// unfinished is a directory a pull is filling or stopped filling; any other
// directory someone keeps in the store is none of the store's business.
func (d modelDir) unfinished() bool { return !d.complete() && (d.arriving > 0 || d.weights > 0) }

func readModelDir(dir string) modelDir {
	var d modelDir
	entries, err := os.ReadDir(dir)
	if err != nil {
		return d
	}
	for _, e := range entries {
		path := filepath.Join(dir, e.Name())
		info, err := os.Stat(path)
		if err != nil || info.IsDir() {
			continue
		}
		name := e.Name()
		switch {
		case strings.HasSuffix(name, ".part"):
			d.arriving++
			if done, _, ok := readRecord(path + ".json"); ok {
				d.size += done
			}
		case strings.HasSuffix(name, ".part.json"), strings.HasSuffix(name, ".part.json.tmp"):
			continue // the pull's records of a file arriving
		default:
			d.size += info.Size()
			switch {
			case name == "config.json":
				d.config = true
			case name == "tokenizer.json":
				d.tokenizer = true
			case strings.HasSuffix(name, ".safetensors"):
				d.weights++
			}
		}
		if info.ModTime().After(d.modified) {
			d.modified = info.ModTime()
		}
	}
	// One piece missing and the model cannot load.
	if b, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json")); err == nil {
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if json.Unmarshal(b, &index) == nil {
			seen := map[string]bool{}
			for _, f := range index.WeightMap {
				if seen[f] {
					continue
				}
				seen[f] = true
				if fi, err := os.Stat(filepath.Join(dir, f)); err != nil || fi.IsDir() {
					d.missing++
				}
			}
		}
	}
	return d
}

// editOf is the base an edited model's edit.json names, or "".
func editOf(dir string) string {
	b, err := os.ReadFile(filepath.Join(dir, "edit.json"))
	if err != nil {
		return ""
	}
	var rec struct {
		Base string `json:"base"`
	}
	json.Unmarshal(b, &rec)
	return rec.Base
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

// isProjector is a vision model's projector file, which is not a model:
// mmproj-F16.gguf as Hugging Face repos name it, NAME.mmproj.gguf as kinfer
// keeps it.
func isProjector(filename string) bool {
	return strings.Contains(strings.ToLower(filename), "mmproj")
}

// projectorBeside is NAME.mmproj.gguf next to NAME.gguf, if it is there.
func projectorBeside(model string) string {
	p := strings.TrimSuffix(model, filepath.Ext(model)) + ".mmproj.gguf"
	if _, err := os.Stat(p); err != nil {
		return ""
	}
	return p
}

// ProjectorFor is the projector of the model at path, or "" for one that
// reads only text: what Ollama's manifest names for its blob, or the file
// beside one of kinfer's own (and the first piece's, for a split model).
func (s *Store) ProjectorFor(path string) string {
	// A model linked into the store from Ollama's has its projector there:
	// the same file, by another name.
	info, statErr := os.Stat(path)
	if all, err := s.All(); err == nil {
		for _, m := range all {
			if m.Projector == "" {
				continue
			}
			if m.Path == path {
				return m.Projector
			}
			if mi, err := os.Stat(m.Path); err == nil && statErr == nil && os.SameFile(info, mi) {
				return m.Projector
			}
		}
	}
	if base, idx, _ := shardOf(filepath.Base(path)); idx > 0 {
		return projectorBeside(filepath.Join(filepath.Dir(path), base+".gguf"))
	}
	return projectorBeside(path)
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
