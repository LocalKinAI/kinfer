package hardware

import "fmt"

// Q4_K_M weighs roughly this much per billion parameters.
//
// Calibrated against real files rather than the nominal 4.83 bits/parameter:
// Llama-3-8B Q4_K_M is ~4.9 GB (0.61 GB/B) and Qwen2.5-7B is ~4.7 GB
// (0.67 GB/B). Small models run heavier because embeddings dominate — the
// measured Qwen2.5-0.5B is 0.47 GB, or 0.94 GB/B — but they are never the
// constraint, so the figure is tuned for the sizes that actually strain a
// machine.
const gbPerBillionQ4 = 0.65

// quantScale converts a Q4_K_M size estimate to another quantisation.
var quantScale = map[string]float64{
	"Q3_K_M": 0.80,
	"Q4_K_M": 1.00,
	"Q5_K_M": 1.17,
	"Q6_K":   1.37,
	"Q8_0":   1.76,
}

// Candidate is a model size the machine could run.
type Candidate struct {
	// Params in billions.
	Params float64

	// Quant is the quantisation, e.g. "Q4_K_M".
	Quant string

	// EstBytes is the estimated file size.
	EstBytes int64

	// Comfort is "comfortable", "tight", or "will swap".
	Comfort string

	// Example is a concrete Hugging Face reference for `kinfer pull`.
	Example string
}

// Recommendation is the advice for one machine.
type Recommendation struct {
	Machine Info

	// Budget is how many bytes may go to model weights.
	Budget int64

	// Best is the largest comfortable model, or nil if nothing fits.
	Best *Candidate

	// Options are other reasonable choices, largest first.
	Options []Candidate

	// Backend is "metal", "cuda", or "cpu".
	Backend string

	// BackendNote explains the backend choice.
	BackendNote string

	// Notes are caveats worth printing.
	Notes []string
}

// knownModels are sizes with a well-known GGUF repo, so the advice ends in a
// command rather than a number.
var knownModels = []struct {
	params  float64
	example string
}{
	{0.5, "Qwen/Qwen2.5-0.5B-Instruct-GGUF"},
	{1.5, "Qwen/Qwen2.5-1.5B-Instruct-GGUF"},
	{3, "Qwen/Qwen2.5-3B-Instruct-GGUF"},
	{7, "Qwen/Qwen2.5-7B-Instruct-GGUF"},
	{14, "Qwen/Qwen2.5-14B-Instruct-GGUF"},
	{32, "Qwen/Qwen2.5-32B-Instruct-GGUF"},
	{72, "Qwen/Qwen2.5-72B-Instruct-GGUF"},
}

// Recommend suggests what to run on this machine.
func Recommend(info Info) Recommendation {
	r := Recommendation{Machine: info}

	if info.RAM == 0 {
		r.Notes = append(r.Notes, "could not read installed memory — sizes below are guesses")
		r.Backend = "cpu"
		r.BackendNote = "unknown hardware, so the safe default"
		return r
	}

	// Reserve 40% of RAM. The OS, a browser, and an editor routinely hold
	// several gigabytes, and on Apple Silicon the GPU draws from this same pool.
	// Overshooting does not fail cleanly — macOS swaps and the machine crawls.
	r.Budget = int64(float64(info.RAM) * 0.6)

	for i := len(knownModels) - 1; i >= 0; i-- {
		m := knownModels[i]
		est := estimate(m.params, "Q4_K_M")

		comfort := ""
		switch {
		case est <= r.Budget*7/10:
			comfort = "comfortable"
		case est <= r.Budget:
			comfort = "tight"
		default:
			continue // beyond this machine
		}

		c := Candidate{
			Params:   m.params,
			Quant:    "Q4_K_M",
			EstBytes: est,
			Comfort:  comfort,
			Example:  fmt.Sprintf("%s:Q4_K_M", m.example),
		}
		if r.Best == nil && comfort == "comfortable" {
			r.Best = &c
		}
		r.Options = append(r.Options, c)
	}

	// Nothing was comfortable — take the largest that at least fits.
	if r.Best == nil && len(r.Options) > 0 {
		r.Best = &r.Options[0]
	}

	// A machine with room to spare gets better quality at the same size instead
	// of a bigger model that leaves nothing for context.
	if r.Best != nil {
		for _, q := range []string{"Q8_0", "Q6_K", "Q5_K_M"} {
			if est := estimate(r.Best.Params, q); est <= r.Budget*7/10 {
				r.Notes = append(r.Notes, fmt.Sprintf(
					"%s also fits at %s (~%.1f GB) — better quality at the same size",
					r.Best.Label(), q, float64(est)/(1<<30)))
				break
			}
		}
	}

	r.Backend, r.BackendNote = pickBackend(info, r.Best)

	if info.Unified {
		r.Notes = append(r.Notes,
			"unified memory: the GPU draws from the same pool, so model size and everything else compete")
	}
	r.Notes = append(r.Notes,
		"estimates are weights only — a large context adds hundreds of MB of KV cache")

	return r
}

// pickBackend chooses CPU or GPU.
//
// The small-model case is measured, not assumed: on an M-series Mac,
// Qwen2.5-0.5B ran at 123.7 tok/s on CPU versus 114.5 tok/s on Metal, because
// transfer overhead outweighs the compute win when the weights are tiny. Above
// a few billion parameters that reverses decisively.
func pickBackend(info Info, best *Candidate) (string, string) {
	switch {
	case info.Metal && best != nil && best.Params <= 1.5:
		return "cpu", "models this small run faster on CPU (measured: 123.7 vs 114.5 tok/s on a 0.5B)"
	case info.Metal:
		return "metal", "offload every layer with -ngl 99"
	case info.CUDA:
		return "cuda", "offload every layer with -ngl 99"
	default:
		return "cpu", "no GPU detected"
	}
}

// estimate returns the approximate file size for a model.
func estimate(params float64, quant string) int64 {
	scale, ok := quantScale[quant]
	if !ok {
		scale = 1.0
	}
	return int64(params * gbPerBillionQ4 * scale * float64(1<<30))
}

// Label renders a parameter count the way people say it: "0.5B", "1.5B", "7B".
// Plain %.0f would round 0.5B to "0B" and 1.5B to "2B".
func (c Candidate) Label() string {
	if c.Params < 10 && c.Params != float64(int(c.Params)) {
		return fmt.Sprintf("%.1fB", c.Params)
	}
	return fmt.Sprintf("%.0fB", c.Params)
}

// Fits reports whether a file of the given size runs comfortably.
func (r Recommendation) Fits(size int64) string {
	switch {
	case r.Budget == 0:
		return "unknown"
	case size <= r.Budget*7/10:
		return "comfortable"
	case size <= r.Budget:
		return "tight"
	default:
		return "too large"
	}
}
