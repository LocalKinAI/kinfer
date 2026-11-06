package mlxengine

import (
	"math"
	"os"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/sampling"
)

// The penalties are llama.cpp's: a seen token's logit divided by the repeat
// penalty when positive, multiplied when negative, then lowered by the
// presence penalty, once however often it was seen.
func TestPenalizeIsLlamaCpps(t *testing.T) {
	lib := os.Getenv("KINFER_MLX_LIB")
	if lib == "" {
		t.Skip("KINFER_MLX_LIB not set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	lg := mlx.FromFloat32([]float32{5, 3, -2, 1, 5, 3, -2, 1, 5, 3, -2, 1}, 3, 4)
	defer lg.Free()
	params := []sampling.Params{
		{RepeatPenalty: 2, PresencePenalty: 1, RepeatLastN: -1},
		{RepeatPenalty: 1, PresencePenalty: 1.5, RepeatLastN: 2}, // sees only its last two tokens
		{RepeatPenalty: 1, RepeatLastN: 64},                      // no penalty at all
	}
	hist := [][]int32{{0, 2, 2}, {0, 2, 3}, {0, 1}}
	s := mlx.NewScope()
	got := mlx.Astype(penalize(lg, params, hist), mlx.Float32).Float32s()
	s.Close()
	want := []float32{
		5/2.0 - 1, 3, -2*2 - 1, 1,
		5, 3, -2 - 1.5, 1 - 1.5,
		5, 3, -2, 1,
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("penalized logits %v, want %v", got, want)
		}
	}
}
