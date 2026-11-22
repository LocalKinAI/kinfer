package server

import (
	"encoding/json"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/engine"
)

// The part of a prompt that was not read again is told in each dialect's own
// field, and each dialect's counts still add up to the whole prompt.
func TestUsageTellsWhatWasCached(t *testing.T) {
	st := engine.Stats{PromptTokens: 7000, CachedTokens: 6900, EvalTokens: 50}
	var o struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
		Details      struct {
			Cached int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	}
	b, _ := json.Marshal(usage(st))
	if json.Unmarshal(b, &o); o.PromptTokens != 7000 || o.Details.Cached != 6900 || o.TotalTokens != 7050 {
		t.Errorf("OpenAI: %s", b)
	}
	var r struct {
		Input   int `json:"input_tokens"`
		Details struct {
			Cached int `json:"cached_tokens"`
		} `json:"input_tokens_details"`
	}
	b, _ = json.Marshal(responsesUsage(st))
	if json.Unmarshal(b, &r); r.Input != 7000 || r.Details.Cached != 6900 {
		t.Errorf("Responses: %s", b)
	}
	var a struct {
		Input   int `json:"input_tokens"`
		Read    int `json:"cache_read_input_tokens"`
		Created int `json:"cache_creation_input_tokens"`
		Output  int `json:"output_tokens"`
	}
	b, _ = json.Marshal(anthropicUsage(st))
	if json.Unmarshal(b, &a); a.Input+a.Read+a.Created != 7000 || a.Read != 6900 || a.Output != 50 {
		t.Errorf("Anthropic: %s", b)
	}
	b, _ = json.Marshal(anthropicUsage(engine.Stats{PromptTokens: 10}))
	if json.Unmarshal(b, &a); a.Input != 10 || a.Read != 0 {
		t.Errorf("Anthropic, nothing cached: %s", b)
	}
}
