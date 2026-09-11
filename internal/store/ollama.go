package store

import (
	"encoding/json"
	"os"
	"path/filepath"
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

	// FromCloud is a model Ollama knows the name of but holds no weights for:
	// the name resolves to hosted inference. kinfer cannot load one, and routes
	// requests for it instead.
	//
	// Recognising these is not guesswork from the ":cloud" suffix — a cloud
	// manifest simply has no weights layer, which is exact and is how Ollama
	// itself tells them apart.
	FromCloud Source = "cloud"
)

// Remote reports that this model is served somewhere else. There is no file to
// open and no slot to occupy; a request for it is forwarded.
func (m Model) Remote() bool { return m.Source == FromCloud }

// OllamaModels lists the models Ollama has, as kinfer can use them.
//
// Cloud models are included and marked FromCloud rather than dropped. They were
// dropped because kinfer could only load files and offering one as something to
// load would fail at a confusing distance from the cause. That reason expired
// when kinfer learned to forward them: the model that took a fleet down for
// half a day was a cloud model, and a runtime that is not in that request's path
// cannot apply a single one of its protections to it.
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
		} `json:"layers"`
	}
	if json.Unmarshal(raw, &mf) != nil {
		return Model{}, false
	}

	// The weights are the layer Ollama calls the model. A cloud entry has none,
	// which is exactly how it is told apart.
	var blob string
	var size int64
	for _, l := range mf.Layers {
		if strings.HasSuffix(l.MediaType, ".image.model") {
			blob = filepath.Join(root, "blobs", strings.ReplaceAll(l.Digest, ":", "-"))
			size = l.Size
			break
		}
	}
	// No weights layer at all is a cloud entry. A weights layer of zero bytes is
	// a broken manifest, and routing that upstream would send a local model's
	// corruption to a hosted endpoint that never heard of it.
	cloud := blob == ""
	var info os.FileInfo
	if !cloud {
		if size == 0 {
			return Model{}, false
		}
		var err error
		if info, err = os.Stat(blob); err != nil {
			return Model{}, false // manifest without its blob
		}
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

	if cloud {
		// No path, no size: there is no file. Reporting a size would invite a
		// caller to reason about memory for something that occupies none here.
		return Model{Name: name, Source: FromCloud}, true
	}
	return Model{
		Name:     name,
		Path:     blob,
		Size:     info.Size(),
		Modified: info.ModTime(),
		Source:   FromOllama,
	}, true
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
