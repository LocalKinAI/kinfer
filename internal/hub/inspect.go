package hub

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"sort"
	"strings"
)

// Quant is one quantisation of a model: possibly several files, and the size
// that matters is their total.
type Quant struct {
	Name  string   // "UD-Q2_K_XL", "Q4_K_M"
	Repo  string   // where it lives
	Files []string // in shard order
	Size  int64
}

var shardName = regexp.MustCompile(`-\d{5}-of-\d{5}\.gguf$`)

// auxiliary names a GGUF that sits beside a model without being one.
//
// A multimodal repo ships mmproj — the vision projector — as its own GGUF, and
// it is both the smallest file present and an entirely different architecture
// ("clip"). Reading the architecture from it, or offering it as something to
// pull, points someone at a few hundred megabytes that cannot answer a
// question. imatrix files are calibration data and not weights at all.
// mtp files are speculative-decoding draft heads — a few dozen tensors that
// ride alongside a model rather than replacing it.
var auxiliary = regexp.MustCompile(`(?i)(^|/)(mmproj|mtp-|.*imatrix.*)`)

// IsModel reports whether a GGUF in a repo is a model rather than something
// shipped alongside one.
func IsModel(filename string) bool { return !auxiliary.MatchString(filename) }

// Quants groups a repo's GGUF files by quantisation, summing split models.
//
// A model past roughly 50 GB is always split, so treating each piece as its own
// download would report a 79 GB model as three unrelated ones — and none of
// them loadable.
func Quants(ctx context.Context, repo string) ([]Quant, error) {
	files, err := List(ctx, repo)
	if err != nil {
		return nil, err
	}

	// Group by what a file is a piece OF, never by the directory it sits in.
	// A directory can hold several unrelated quantisations — one repo keeps six
	// draft heads in an MTP/ folder — and summing those reports a model that
	// does not exist, at a size nothing has.
	byBase := map[string]*Quant{}
	for _, f := range files {
		if !strings.HasSuffix(strings.ToLower(f.Name), ".gguf") || !IsModel(f.Name) {
			continue
		}
		base := shardName.ReplaceAllString(f.Name, "")
		if base == f.Name {
			base = strings.TrimSuffix(f.Name, ".gguf") // not split: it stands alone
		}

		q := byBase[base]
		if q == nil {
			q = &Quant{Name: quantName(f.Name, base), Repo: repo}
			byBase[base] = q
		}
		q.Files = append(q.Files, f.Name)
		q.Size += f.Size
	}

	out := make([]Quant, 0, len(byBase))
	for _, q := range byBase {
		sort.Strings(q.Files)
		out = append(out, *q)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Size < out[j].Size })
	return out, nil
}

// quantName is what to call a quantisation: uploaders put it either in a
// directory name or in the filename, and the one that is not the model's own
// name is the one that identifies it.
func quantName(file, base string) string {
	if i := strings.LastIndex(base, "/"); i >= 0 {
		dir, stem := base[:i], base[i+1:]
		// A directory holding one split model is named for the quantisation;
		// a directory holding several files is not.
		if strings.Contains(stem, dir) || strings.Contains(file, "-of-") {
			return dir
		}
		return stem
	}
	// Everything in one directory: the quantisation trails the model name.
	name := base
	for _, sep := range []string{"-", "."} {
		if j := strings.LastIndex(name, sep); j >= 0 && j > len(name)/2 {
			return name[j+1:]
		}
	}
	return name
}

// Search finds repos matching a query, newest and most-downloaded first, which
// is how a GGUF conversion of a safetensors-only release is located.
func Search(ctx context.Context, query string, limit int) ([]string, error) {
	url := fmt.Sprintf("%s/api/models?search=%s&limit=%d&sort=downloads&direction=-1",
		endpoint(), strings.ReplaceAll(query, " ", "%20"), limit)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search %s via %s: %w", query, endpoint(), err)
	}
	defer resp.Body.Close()

	var hits []struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&hits); err != nil {
		return nil, err
	}
	out := make([]string, 0, len(hits))
	for _, h := range hits {
		out = append(out, h.ID)
	}
	return out, nil
}

// ModelType reads a safetensors repo's declared architecture from config.json.
//
// It is the same name a GGUF conversion will carry as general.architecture, so
// it answers "will llama.cpp run this" before any conversion exists.
func ModelType(ctx context.Context, repo string) (string, error) {
	url := fmt.Sprintf("%s/%s/resolve/main/config.json", endpoint(), repo)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("no config.json in %s", repo)
	}
	var cfg struct {
		ModelType string `json:"model_type"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&cfg); err != nil {
		return "", err
	}
	return cfg.ModelType, nil
}

// Architecture reads general.architecture out of a GGUF without downloading it.
//
// The metadata sits at the front of the file, so a few megabytes settle the
// question that otherwise costs tens of gigabytes to ask.
func Architecture(ctx context.Context, repo, file string) (string, error) {
	url := fmt.Sprintf("%s/%s/resolve/main/%s", endpoint(), repo, file)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Range", "bytes=0-1048575") // 1 MB: architecture is the first key
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	head, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", err
	}
	return ggufArchitecture(head)
}

// ggufArchitecture parses just enough of a GGUF header to read the first
// metadata key, which is general.architecture in every file llama.cpp writes.
func ggufArchitecture(d []byte) (string, error) {
	if len(d) < 24 || string(d[:4]) != "GGUF" {
		return "", fmt.Errorf("not a GGUF file")
	}
	nKV := binary.LittleEndian.Uint64(d[16:24])
	if nKV == 0 {
		return "", fmt.Errorf("GGUF carries no metadata")
	}
	off := 24

	readStr := func() (string, bool) {
		if off+8 > len(d) {
			return "", false
		}
		n := int(binary.LittleEndian.Uint64(d[off:]))
		off += 8
		if n < 0 || off+n > len(d) {
			return "", false
		}
		s := string(d[off : off+n])
		off += n
		return s, true
	}

	key, ok := readStr()
	if !ok || key != "general.architecture" {
		return "", fmt.Errorf("unexpected first metadata key %q", key)
	}
	if off+4 > len(d) {
		return "", fmt.Errorf("truncated GGUF header")
	}
	if t := binary.LittleEndian.Uint32(d[off:]); t != 8 { // 8 = string
		return "", fmt.Errorf("general.architecture is not a string")
	}
	off += 4
	arch, ok := readStr()
	if !ok {
		return "", fmt.Errorf("truncated GGUF header")
	}
	return arch, nil
}
