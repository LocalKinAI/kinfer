package store

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"
)

// Anyone installing kinfer is likely already running Ollama, with tens of
// gigabytes of models on disk. Making them download the same weights again is a
// poor welcome, and for a runtime whose whole purpose is to be the second leg
// when the first one fails, duplicating the first leg's storage is worse than
// poor — it is the reason there is no room for the fallback.
//
// So Ollama's models are discovered and read in place. They are never written
// to, never moved, and never deleted: that store belongs to Ollama, and kinfer
// borrowing from it must not be able to damage it.

// Source says where a model came from.
type Source string

const (
	// FromKinfer is a file in kinfer's own directory.
	FromKinfer Source = "kinfer"

	// FromOllama is a blob in Ollama's store, read where it lies.
	FromOllama Source = "ollama"
)

// OllamaModels lists the local models Ollama has, as kinfer can use them.
//
// Cloud models are skipped. Ollama lists them like any other, but they hold no
// weights — the name resolves to hosted inference — and kinfer serves local
// models only. That is a boundary, not a gap: for one day kinfer forwarded
// them, and a fleet whose cloud quota ran out was answered by a local model
// that could not hold its context, under a health check that could not tell.
// Cloud belongs to Ollama; kinfer stays on the other side of the line.
func OllamaModels() []Model {
	root := ollamaRoot()
	if root == "" {
		return nil
	}
	manifests := filepath.Join(root, "manifests")

	var out []Model
	_ = filepath.WalkDir(manifests, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil //nolint:nilerr // a directory we cannot read is not fatal
		}
		m, ok := readOllamaManifest(root, manifests, path)
		if ok {
			out = append(out, m)
		}
		return nil
	})

	sort.Slice(out, func(i, j int) bool { return out[i].Modified.After(out[j].Modified) })
	return out
}

func readOllamaManifest(root, manifests, path string) (Model, bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Model{}, false
	}
	var mf struct {
		Layers []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
			Size      int64  `json:"size"`
			Name      string `json:"name"`
		} `json:"layers"`
	}
	if json.Unmarshal(raw, &mf) != nil {
		return Model{}, false
	}

	// manifests/<registry>/<namespace>/<name>/<tag> is Ollama's own name for it,
	// and the name someone will type.
	rel, err := filepath.Rel(manifests, path)
	if err != nil {
		return Model{}, false
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	if len(parts) < 2 {
		return Model{}, false
	}
	name := parts[len(parts)-2] + ":" + parts[len(parts)-1]
	if ns := parts[len(parts)-3 : len(parts)-2]; len(ns) == 1 && ns[0] != "library" {
		name = ns[0] + "/" + name
	}

	// The weights are the layer Ollama calls the model. A cloud entry has none,
	// which is exactly how it is told apart — unless it is an MLX model, whose
	// weights are a layer for each tensor.
	var blob, projector string
	var size, tensorBytes int64
	var tensors, files []namedBlob
	for _, l := range mf.Layers {
		path := filepath.Join(root, "blobs", strings.ReplaceAll(l.Digest, ":", "-"))
		switch {
		case strings.HasSuffix(l.MediaType, ".image.model") && blob == "":
			blob, size = path, l.Size
		case strings.HasSuffix(l.MediaType, ".image.projector"):
			// A vision model's projector, the mmproj another runtime reads.
			if _, err := os.Stat(path); err == nil {
				projector = path
			}
		case strings.HasSuffix(l.MediaType, ".image.tensor"):
			tensors = append(tensors, namedBlob{l.Name, path})
			tensorBytes += l.Size
		case strings.HasSuffix(l.MediaType, ".image.json"):
			files = append(files, namedBlob{l.Name, path})
		}
	}
	if blob == "" && len(tensors) > 0 {
		dir, err := ollamaMLXDir(name, raw, tensors, files)
		info, statErr := os.Stat(path)
		if err != nil || statErr != nil {
			return Model{}, false
		}
		return Model{Name: name, Path: dir, Format: "safetensors", Size: tensorBytes, Modified: info.ModTime(), Source: FromOllama}, true
	}
	// No weights layer is a cloud entry; a zero-byte one is a broken manifest.
	// Neither is anything kinfer can load.
	if blob == "" || size == 0 {
		return Model{}, false
	}
	info, err := os.Stat(blob)
	if err != nil {
		return Model{}, false // manifest without its blob
	}

	return Model{
		Name:      name,
		Path:      blob,
		Format:    "gguf",
		Size:      info.Size(),
		Modified:  info.ModTime(),
		Source:    FromOllama,
		Projector: projector,
	}, true
}

// namedBlob is a layer of an Ollama manifest that has a name of its own.
type namedBlob struct{ name, path string }

// OllamaMLXRoot is where kinfer keeps a directory for each of Ollama's MLX
// models; "" when the machine has no cache directory.
func OllamaMLXRoot() string {
	cache, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(cache, "kinfer", "ollama-mlx")
}

// ollamaMLXDir returns a directory the MLX engine can read for one of
// Ollama's MLX models, making it first if the manifest is new.
//
// Since 0.40 Ollama keeps an MLX model as no single file: its manifest names a
// blob for each tensor — a safetensors file holding the tensor and, for a
// quantized weight, its scales — and blobs for config.json, tokenizer.json and
// the rest by their names. The directory is symbolic links to those blobs
// under the names they stand for, and the chat template, which Ollama keeps
// in tokenizer_config.json, written out as chat_template.jinja. Nothing is
// copied and nothing of Ollama's is touched. A stamp of the manifest says
// whether the directory is still its own.
func ollamaMLXDir(name string, manifest []byte, tensors, files []namedBlob) (string, error) {
	root := OllamaMLXRoot()
	if root == "" {
		return "", fmt.Errorf("no cache directory for %s", name)
	}
	dir := filepath.Join(root, strings.NewReplacer(":", "_", "/", "_").Replace(name))
	sum := sha256.Sum256(manifest)
	stamp := hex.EncodeToString(sum[:])
	if b, err := os.ReadFile(filepath.Join(dir, ".manifest")); err == nil && string(b) == stamp {
		return dir, nil
	}

	tmp := fmt.Sprintf("%s.%d", dir, os.Getpid())
	os.RemoveAll(tmp)
	if err := os.MkdirAll(tmp, 0o755); err != nil {
		return "", err
	}
	defer os.RemoveAll(tmp)
	link := func(file, target string) error {
		if file == "" || strings.ContainsAny(file, `/\`) || strings.HasPrefix(file, ".") {
			return fmt.Errorf("%s has a layer named %q", name, file)
		}
		return os.Symlink(target, filepath.Join(tmp, file))
	}
	for _, t := range tensors {
		if err := link(t.name+".safetensors", t.path); err != nil {
			return "", err
		}
	}
	hasTemplate := slices.ContainsFunc(files, func(f namedBlob) bool { return f.name == "chat_template.jinja" })
	for _, f := range files {
		if err := link(f.name, f.path); err != nil {
			return "", err
		}
		if f.name == "tokenizer_config.json" && !hasTemplate {
			if t := chatTemplate(f.path); t != "" {
				if err := os.WriteFile(filepath.Join(tmp, "chat_template.jinja"), []byte(t), 0o644); err != nil {
					return "", err
				}
			}
		}
	}
	if err := os.WriteFile(filepath.Join(tmp, ".manifest"), []byte(stamp), 0o644); err != nil {
		return "", err
	}
	os.RemoveAll(dir)
	if err := os.Rename(tmp, dir); err != nil {
		// Another kinfer made it at the same moment; theirs will do.
		if b, rerr := os.ReadFile(filepath.Join(dir, ".manifest")); rerr == nil && string(b) == stamp {
			return dir, nil
		}
		return "", err
	}
	return dir, nil
}

// chatTemplate is the chat template a tokenizer_config.json carries: one
// string, or the one named default of several.
func chatTemplate(path string) string {
	raw, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	var tc struct {
		ChatTemplate json.RawMessage `json:"chat_template"`
	}
	if json.Unmarshal(raw, &tc) != nil {
		return ""
	}
	var one string
	if json.Unmarshal(tc.ChatTemplate, &one) == nil {
		return one
	}
	var named []struct{ Name, Template string }
	json.Unmarshal(tc.ChatTemplate, &named)
	for _, t := range named {
		if t.Name == "default" {
			return t.Template
		}
	}
	return ""
}

// ollamaRoot finds Ollama's model store.
//
// ~/.ollama/models is the default and not much else. A large collection
// routinely lives on another volume — the machine this was written against
// keeps 40 GB at /Volumes/Data/.ollama/models while ~/.ollama/models holds only
// a few kilobytes of cloud manifests, which look convincingly like an empty
// install. So the candidates are checked for blobs, not merely for existing.
func ollamaRoot() string {
	var candidates []string
	if p := os.Getenv("OLLAMA_MODELS"); p != "" {
		candidates = append(candidates, p)
	}
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates, filepath.Join(home, ".ollama", "models"))
	}
	for _, v := range volumeRoots() {
		candidates = append(candidates, filepath.Join(v, ".ollama", "models"))
	}
	// Linux packages run the daemon as its own user.
	candidates = append(candidates, "/usr/share/ollama/.ollama/models")

	var fallback string
	for _, c := range candidates {
		if !hasBlobs(c) {
			continue
		}
		// A store with manifests that name no weights is all cloud entries:
		// real, but nothing kinfer can load. Keep looking, and settle for it
		// only if nothing better turns up.
		if countBlobs(c) > 0 {
			return c
		}
		if fallback == "" {
			fallback = c
		}
	}
	return fallback
}

func hasBlobs(root string) bool {
	fi, err := os.Stat(filepath.Join(root, "manifests"))
	return err == nil && fi.IsDir()
}

// countBlobs reports how many weight files a store actually holds.
func countBlobs(root string) int {
	entries, err := os.ReadDir(filepath.Join(root, "blobs"))
	if err != nil {
		return 0
	}
	n := 0
	for _, e := range entries {
		if info, err := e.Info(); err == nil && info.Size() > 1<<20 {
			n++
		}
	}
	return n
}

// volumeRoots lists mounted volumes, where a large model collection often ends
// up because the boot disk cannot hold it.
func volumeRoots() []string {
	entries, err := os.ReadDir("/Volumes")
	if err != nil {
		return nil
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		out = append(out, filepath.Join("/Volumes", e.Name()))
	}
	return out
}

var _ = time.Time{}
