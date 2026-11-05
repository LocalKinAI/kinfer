// Package hub downloads models from Hugging Face: GGUF files, and the
// safetensors directories kinfer serves on MLX.
//
// No registry sits in between: a Hugging Face repo id is the model name, and
// the file lands on disk under the name it has upstream. There is nothing to
// mirror, nothing to wait for, and nothing to reverse-engineer when you want to
// know what a file actually is.
package hub

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const defaultEndpoint = "https://huggingface.co"

// endpoint returns the Hugging Face host to talk to.
//
// Hard-coding huggingface.co breaks kinfer wherever it is slow or unreachable —
// which includes mainland China, where every delivered machine would otherwise
// need a VPN just to fetch a model. HF_ENDPOINT is the variable the official
// Python client already honours, so mirrors document it and users may already
// have it set:
//
//	export HF_ENDPOINT=https://hf-mirror.com
//
// KINFER_HF_ENDPOINT wins when both are set, so kinfer can be pointed elsewhere
// without disturbing other tools on the same machine.
func endpoint() string {
	for _, key := range []string{"KINFER_HF_ENDPOINT", "HF_ENDPOINT"} {
		if v := strings.TrimSpace(os.Getenv(key)); v != "" {
			return strings.TrimRight(v, "/")
		}
	}
	return defaultEndpoint
}

// Endpoint reports the host in use, for display.
func Endpoint() string { return endpoint() }

// Ref identifies a model to fetch.
type Ref struct {
	// Repo is "owner/name", e.g. "Qwen/Qwen2.5-0.5B-Instruct-GGUF".
	Repo string

	// Quant selects a quantisation, e.g. "Q4_K_M". Empty picks a default.
	Quant string

	// File names an exact file, bypassing quant matching.
	File string
}

// ParseRef accepts the forms people actually type:
//
//	Qwen/Qwen2.5-0.5B-Instruct-GGUF
//	Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
//	hf.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
//	https://huggingface.co/Qwen/...-GGUF/blob/main/model.gguf
func ParseRef(s string) (Ref, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return Ref{}, fmt.Errorf("empty model reference")
	}

	// A full URL: pull out the repo and, if present, the exact file — with
	// its folder, as in .../resolve/main/Q2_K/model-00001-of-00003.gguf, and
	// without the ?download=true a link copied from the website carries. The
	// host may be a mirror's.
	if strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://") {
		u, err := url.Parse(s)
		if err != nil {
			return Ref{}, fmt.Errorf("cannot read a repo from %q", s)
		}
		parts := strings.Split(strings.Trim(u.Path, "/"), "/")
		if len(parts) < 2 || parts[0] == "" {
			return Ref{}, fmt.Errorf("cannot read a repo from %q", s)
		}
		ref := Ref{Repo: parts[0] + "/" + parts[1]}
		// .../blob/<revision>/<path> or .../resolve/<revision>/<path>
		if len(parts) > 4 && (parts[2] == "blob" || parts[2] == "resolve") {
			ref.File = strings.Join(parts[4:], "/")
		}
		return ref, nil
	}

	s = strings.TrimPrefix(strings.TrimPrefix(s, "hf.co/"), "huggingface.co/")

	var quant string
	if i := strings.LastIndex(s, ":"); i > 0 {
		quant, s = s[i+1:], s[:i]
	}
	if strings.Count(s, "/") != 1 {
		return Ref{}, fmt.Errorf("expected owner/repo, got %q", s)
	}
	return Ref{Repo: s, Quant: quant}, nil
}

// File is one downloadable file in a repo.
type File struct {
	Name string
	Size int64
}

// List returns the GGUF files in a repo, in name order.
func List(ctx context.Context, repo string) ([]File, error) {
	files, err := Files(ctx, repo)
	if err != nil {
		return nil, err
	}
	out := GGUFs(files)
	if len(out) == 0 {
		return nil, fmt.Errorf("%s contains no .gguf files", repo)
	}
	return out, nil
}

// GGUFs is the GGUF files among a repo's files.
func GGUFs(files []File) []File {
	var out []File
	for _, f := range files {
		if strings.HasSuffix(strings.ToLower(f.Name), ".gguf") {
			out = append(out, f)
		}
	}
	return out
}

// Files returns every file in a repo, in name order.
func Files(ctx context.Context, repo string) ([]File, error) {
	// blobs=true is what makes Hugging Face report file sizes; without it every
	// file comes back as zero bytes, and a size-based decision silently becomes
	// a decision about nothing.
	url := fmt.Sprintf("%s/api/models/%s?blobs=true", endpoint(), repo)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		// The most common cause is a blocked or slow host, and the fix is a
		// mirror — worth saying so rather than printing a bare network error.
		hint := ""
		if endpoint() == defaultEndpoint {
			hint = "\n  (unreachable? try: export HF_ENDPOINT=https://hf-mirror.com)"
		}
		return nil, fmt.Errorf("query %s via %s: %w%s", repo, endpoint(), err, hint)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("no such repo: %s", repo)
	}
	// Hugging Face answers 401 for repos that are private *and* for ones that do
	// not exist, so the message has to cover both rather than say "unauthorized".
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return nil, fmt.Errorf("cannot read %s — it is private, gated, or misspelled", repo)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("hugging face returned %s for %s", resp.Status, repo)
	}

	var payload struct {
		Siblings []struct {
			Filename string `json:"rfilename"`
			Size     int64  `json:"size"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode repo listing: %w", err)
	}

	var out []File
	for _, s := range payload.Siblings {
		out = append(out, File{Name: s.Filename, Size: s.Size})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// MLXFiles picks, from a repo's files, the ones a model kinfer serves on MLX
// is made of, and reports whether they are one: config.json, tokenizer.json
// and the weights in safetensors, with what else at the top of the repo
// describes them (the chat template, the safetensors index, the other
// configs). The weights come first and config.json last, so the directory
// they land in is no model to the store until all of it has arrived.
func MLXFiles(files []File) ([]File, bool) {
	var weights, rest []File
	var config File
	tokenizer := false
	for _, f := range files {
		if strings.Contains(f.Name, "/") {
			continue // another copy, another framework's weights, a picture
		}
		switch ext := strings.ToLower(filepath.Ext(f.Name)); {
		case f.Name == "config.json":
			config = f
		case ext == ".safetensors":
			weights = append(weights, f)
		case ext == ".json" || ext == ".jinja" || ext == ".txt":
			tokenizer = tokenizer || f.Name == "tokenizer.json"
			rest = append(rest, f)
		}
	}
	if config.Name == "" || !tokenizer || len(weights) == 0 {
		return nil, false
	}
	return append(append(weights, rest...), config), true
}

// Pick chooses which file to download.
//
// An explicit file wins; then a quantisation match; otherwise Q4_K_M, the
// quantisation with the best size-to-quality ratio for most models — and if
// that is absent, the first file, so `pull` still does something sensible.
func Pick(files []File, ref Ref) (File, error) {
	// A projector, an importance matrix, an MTP head: files beside a model,
	// not models. "F16" would otherwise pick mmproj-F16.gguf.
	var models []File
	for _, f := range files {
		if IsModel(f.Name) {
			models = append(models, f)
		}
	}
	if len(models) > 0 {
		files = models
	}
	if ref.File != "" {
		for _, f := range files {
			if strings.EqualFold(f.Name, ref.File) {
				return f, nil
			}
		}
		return File{}, fmt.Errorf("%s not found in %s", ref.File, ref.Repo)
	}

	if ref.Quant != "" {
		var matches []File
		for _, f := range files {
			if strings.Contains(strings.ToLower(f.Name), strings.ToLower(ref.Quant)) {
				matches = append(matches, f)
			}
		}
		switch len(matches) {
		case 1:
			return matches[0], nil
		case 0:
			return File{}, fmt.Errorf("no file matches quantisation %q (have: %s)", ref.Quant, join(files))
		default:
			// Multi-part models split as -00001-of-00003; take the first part so
			// the caller at least starts in the right place.
			for _, m := range matches {
				if strings.Contains(m.Name, "00001-of-") {
					return m, nil
				}
			}
			return File{}, fmt.Errorf("%q is ambiguous: %s", ref.Quant, join(matches))
		}
	}

	for _, f := range files {
		if strings.Contains(strings.ToLower(f.Name), "q4_k_m") {
			return f, nil
		}
	}
	return files[0], nil
}

// Projector is the projector a vision model's repo ships, the file it reads
// pictures with: F16 where there is one — Metal reads it as is — then BF16,
// F32, and failing those the first.
func Projector(files []File) (File, bool) {
	var found []File
	for _, f := range files {
		if strings.Contains(strings.ToLower(filepath.Base(f.Name)), "mmproj") {
			found = append(found, f)
		}
	}
	for _, kind := range []string{"f16", "bf16", "f32"} {
		for _, f := range found {
			if strings.Contains(strings.ToLower(f.Name), kind) && !(kind == "f16" && strings.Contains(strings.ToLower(f.Name), "bf16")) {
				return f, true
			}
		}
	}
	if len(found) > 0 {
		return found[0], true
	}
	return File{}, false
}

// ProjectorName is what a projector is kept as beside the model set names —
// the files of one split model, or one file — so the store finds it.
func ProjectorName(set []File) string {
	name := strings.TrimSuffix(filepath.Base(set[0].Name), ".gguf")
	if len(set) > 1 {
		name = pieceSuffix.ReplaceAllString(name, "")
	}
	return name + ".mmproj.gguf"
}

// pieceSuffix is a split model's piece number at the end of a name.
var pieceSuffix = regexp.MustCompile(`-\d{5}-of-\d{5}$`)

// pieceNumber is where a split model's piece numbers itself, as in
// "-00002-of-00005.gguf". Taking the number out leaves a name every piece of
// one set shares.
var pieceNumber = regexp.MustCompile(`-\d{5}(-of-\d{5}\.gguf)$`)

// Shards returns every piece of the split model f is a piece of, in order, or
// just f when it stands alone. llama.cpp opens a split model from its first
// piece and finds the rest beside it, so one piece on its own is no model.
func Shards(files []File, f File) []File {
	set := pieceNumber.ReplaceAllString(f.Name, "-*${1}")
	if set == f.Name {
		return []File{f}
	}
	var out []File
	for _, g := range files {
		if pieceNumber.ReplaceAllString(g.Name, "-*${1}") == set {
			out = append(out, g)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func join(files []File) string {
	names := make([]string, len(files))
	for i, f := range files {
		names[i] = f.Name
	}
	return strings.Join(names, ", ")
}
