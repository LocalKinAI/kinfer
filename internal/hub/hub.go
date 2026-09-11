// Package hub downloads GGUF models from Hugging Face.
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
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
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

	// A full URL: pull out the repo and, if present, the exact file.
	if strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://") {
		trimmed := strings.TrimPrefix(strings.TrimPrefix(s, "https://"), "http://")
		trimmed = strings.TrimPrefix(trimmed, "huggingface.co/")
		parts := strings.Split(trimmed, "/")
		if len(parts) < 2 {
			return Ref{}, fmt.Errorf("cannot read a repo from %q", s)
		}
		ref := Ref{Repo: parts[0] + "/" + parts[1]}
		// .../blob/main/file.gguf or .../resolve/main/file.gguf
		for i := 2; i < len(parts)-1; i++ {
			if parts[i] == "blob" || parts[i] == "resolve" {
				ref.File = parts[len(parts)-1]
			}
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

// Progress reports download state. Total is 0 when the server sends no length.
type Progress func(downloaded, total int64)

// List returns the GGUF files in a repo, smallest first.
func List(ctx context.Context, repo string) ([]File, error) {
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
		if strings.HasSuffix(strings.ToLower(s.Filename), ".gguf") {
			out = append(out, File{Name: s.Filename, Size: s.Size})
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s contains no .gguf files", repo)
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// Pick chooses which file to download.
//
// An explicit file wins; then a quantisation match; otherwise Q4_K_M, the
// quantisation with the best size-to-quality ratio for most models — and if
// that is absent, the first file, so `pull` still does something sensible.
func Pick(files []File, ref Ref) (File, error) {
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

// Download fetches a file into dir and returns its path.
//
// Downloads land in a .part file and are renamed on success, so an interrupted
// pull can never leave something that looks like a usable model.
func Download(ctx context.Context, repo, filename, dir string, onProgress Progress) (string, error) {
	url := fmt.Sprintf("%s/%s/resolve/main/%s", endpoint(), repo, filename)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}

	client := &http.Client{Timeout: 6 * time.Hour} // large models on slow links
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", filename, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: %s", filename, resp.Status)
	}

	final := filepath.Join(dir, filepath.Base(filename))
	part := final + ".part"

	f, err := os.Create(part)
	if err != nil {
		return "", fmt.Errorf("create %s: %w", part, err)
	}
	defer func() {
		f.Close()
		os.Remove(part) // no-op after a successful rename
	}()

	w := &progressWriter{w: f, total: resp.ContentLength, onProgress: onProgress}
	if _, err := io.Copy(w, resp.Body); err != nil {
		return "", fmt.Errorf("download %s: %w", filename, err)
	}
	if err := f.Close(); err != nil {
		return "", fmt.Errorf("finish %s: %w", part, err)
	}
	if err := os.Rename(part, final); err != nil {
		return "", fmt.Errorf("install %s: %w", final, err)
	}
	return final, nil
}

type progressWriter struct {
	w          io.Writer
	total      int64
	done       int64
	lastReport time.Time
	onProgress Progress
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.done += int64(n)

	// Throttle: a multi-GB download would otherwise repaint thousands of times
	// a second and cost more than the transfer.
	if p.onProgress != nil && time.Since(p.lastReport) > 100*time.Millisecond {
		p.onProgress(p.done, p.total)
		p.lastReport = time.Now()
	}
	return n, err
}

func join(files []File) string {
	names := make([]string, len(files))
	for i, f := range files {
		names[i] = f.Name
	}
	return strings.Join(names, ", ")
}
