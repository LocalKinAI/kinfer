package hardware

import "testing"

const gb = int64(1) << 30

func mac(ramGB int64) Info {
	return Info{OS: "darwin", Arch: "arm64", Cores: 10,
		RAM: ramGB * gb, Metal: true, Unified: true, GPU: "Apple GPU"}
}

// The estimates must land near real files, or every recommendation is wrong.
func TestEstimate_MatchesRealFiles(t *testing.T) {
	cases := []struct {
		params    float64
		quant     string
		wantGB    float64
		tolerance float64
	}{
		// Llama-3-8B Q4_K_M is ~4.9 GB; Qwen2.5-7B Q4_K_M ~4.7 GB.
		{7, "Q4_K_M", 4.55, 0.7},
		{14, "Q4_K_M", 9.1, 1.2},
		// Q8_0 is roughly twice Q4_K_M.
		{7, "Q8_0", 8.0, 1.2},
	}
	for _, c := range cases {
		got := float64(estimate(c.params, c.quant)) / float64(gb)
		if got < c.wantGB-c.tolerance || got > c.wantGB+c.tolerance {
			t.Errorf("estimate(%.0fB, %s) = %.1f GB, want ~%.1f", c.params, c.quant, got, c.wantGB)
		}
	}
}

func TestRecommend_ScalesWithMemory(t *testing.T) {
	cases := []struct {
		ramGB       int64
		wantAtMost  float64
		wantAtLeast float64
	}{
		{8, 3, 1.5},   // 8 GB: a 3B at most
		{16, 7, 3},    // 16 GB: 7B territory
		{32, 14, 7},   // 32 GB: 14B
		{64, 32, 14},  // 64 GB: 32B
		{128, 72, 32}, // 128 GB: the big ones
	}
	for _, c := range cases {
		r := Recommend(mac(c.ramGB))
		if r.Best == nil {
			t.Errorf("%d GB: no recommendation at all", c.ramGB)
			continue
		}
		if r.Best.Params > c.wantAtMost {
			t.Errorf("%d GB recommended %.0fB — too large (max %.0fB)", c.ramGB, r.Best.Params, c.wantAtMost)
		}
		if r.Best.Params < c.wantAtLeast {
			t.Errorf("%d GB recommended only %.0fB — too timid (expected >= %.0fB)", c.ramGB, r.Best.Params, c.wantAtLeast)
		}
	}
}

// The recommendation must actually fit in the stated budget.
func TestRecommend_BestFitsBudget(t *testing.T) {
	for _, ramGB := range []int64{8, 16, 32, 64, 128} {
		r := Recommend(mac(ramGB))
		if r.Best == nil {
			continue
		}
		if r.Best.EstBytes > r.Budget {
			t.Errorf("%d GB: recommended %.1f GB but budget is %.1f GB",
				ramGB, float64(r.Best.EstBytes)/float64(gb), float64(r.Budget)/float64(gb))
		}
	}
}

// Measured on an M-series Mac: a 0.5B runs faster on CPU than on Metal.
func TestBackend_SmallModelsPreferCPU(t *testing.T) {
	small := &Candidate{Params: 0.5}
	if b, _ := pickBackend(mac(8), small); b != "cpu" {
		t.Errorf("0.5B on Metal hardware should pick cpu, got %q", b)
	}

	large := &Candidate{Params: 14}
	if b, _ := pickBackend(mac(64), large); b != "metal" {
		t.Errorf("14B should pick metal, got %q", b)
	}

	if b, _ := pickBackend(Info{OS: "linux"}, large); b != "cpu" {
		t.Errorf("no GPU should pick cpu, got %q", b)
	}
	if b, _ := pickBackend(Info{OS: "linux", CUDA: true}, large); b != "cuda" {
		t.Errorf("CUDA machine should pick cuda, got %q", b)
	}
}

// Unknown memory must degrade gracefully, not recommend nonsense.
func TestRecommend_UnknownMemory(t *testing.T) {
	r := Recommend(Info{OS: "linux", Arch: "amd64", Cores: 4})
	if r.Best != nil {
		t.Error("with unknown RAM there should be no recommendation")
	}
	if r.Backend != "cpu" {
		t.Errorf("unknown hardware should default to cpu, got %q", r.Backend)
	}
	if len(r.Notes) == 0 {
		t.Error("unknown RAM should be called out in the notes")
	}
}

// A tiny machine should still say something rather than nothing.
func TestRecommend_TinyMachine(t *testing.T) {
	r := Recommend(mac(4))
	if r.Best == nil {
		t.Fatal("4 GB should still get a suggestion")
	}
	if r.Best.Params > 1.5 {
		t.Errorf("4 GB recommended %.1fB — unrealistic", r.Best.Params)
	}
}

func TestFits(t *testing.T) {
	r := Recommend(mac(16)) // budget ~9.6 GB
	cases := map[int64]string{
		1 * gb:  "comfortable",
		7 * gb:  "tight",
		20 * gb: "too large",
	}
	for size, want := range cases {
		if got := r.Fits(size); got != want {
			t.Errorf("Fits(%d GB) = %q, want %q", size/gb, got, want)
		}
	}
}
