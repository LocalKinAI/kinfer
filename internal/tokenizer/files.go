package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// LoadDir reads a model directory's tokenizer: tokenizer.json, or, for a
// release that ships only the slow tokenizer's files — Qwen3-ASR's MLX
// conversion has vocab.json, merges.txt and tokenizer_config.json — the
// byte-level BPE they describe, with Qwen2's pre-tokenizer.
func LoadDir(dir string) (*Tokenizer, error) {
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err == nil {
		return Load(filepath.Join(dir, "tokenizer.json"))
	}
	vocab, err := os.ReadFile(filepath.Join(dir, "vocab.json"))
	if err != nil {
		return nil, fmt.Errorf("%s has neither tokenizer.json nor vocab.json", dir)
	}
	mergeText, err := os.ReadFile(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, err
	}
	var merges []string
	for _, line := range strings.Split(string(mergeText), "\n") {
		line = strings.TrimRight(line, "\r")
		if line == "" || strings.HasPrefix(line, "#version") {
			continue
		}
		merges = append(merges, line)
	}
	var cfg struct {
		Added map[string]struct {
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens_decoder"`
	}
	if b, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json")); err == nil {
		if err := json.Unmarshal(b, &cfg); err != nil {
			return nil, fmt.Errorf("%s/tokenizer_config.json: %w", dir, err)
		}
	}
	type addedTok struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	}
	var added []addedTok
	for k, a := range cfg.Added {
		id, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("%s/tokenizer_config.json: added token id %q", dir, k)
		}
		added = append(added, addedTok{int32(id), a.Content, a.Special})
	}
	sort.Slice(added, func(i, j int) bool { return added[i].ID < added[j].ID })
	mj, _ := json.Marshal(merges)
	aj, _ := json.Marshal(added)
	pt, _ := json.Marshal(map[string]any{"type": "Sequence", "pretokenizers": []any{
		map[string]any{"type": "Split", "pattern": map[string]string{"Regex": qwen2Pattern}, "behavior": "Isolated"},
		map[string]any{"type": "ByteLevel", "use_regex": false},
	}})
	raw := fmt.Sprintf(`{"added_tokens":%s,"pre_tokenizer":%s,"model":{"type":"BPE","vocab":%s,"merges":%s}}`, aj, pt, vocab, mj)
	return parse([]byte(raw), filepath.Join(dir, "vocab.json"))
}
