package engine

import (
	"testing"

	"github.com/LocalKinAI/kinfer/internal/llama"
)

func TestWarmOnlyForASplitWithoutHostExperts(t *testing.T) {
	for _, c := range []struct {
		split, cache, want int
	}{
		{0, 0, 0},                  // one machine
		{1, 0, llama.WarmWindowMs}, // a split, all on GPUs
		{2, 0, llama.WarmWindowMs},
		{1, 1, 0}, // a split beside a model with host experts
		{0, 1, 0}, // host experts on one machine
	} {
		if got := warmWindow(c.split, c.cache); got != c.want {
			t.Errorf("warmWindow(%d, %d) = %d, want %d", c.split, c.cache, got, c.want)
		}
	}
}
