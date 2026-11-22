package mlxasr

import (
	"math"
	"testing"
)

// A long recording is cut near every twenty minutes, at its quietest, and the
// pieces are all of it, in order.
func TestPiecesCutWhereItIsQuiet(t *testing.T) {
	const sr = sampleRate
	n := 50 * 60 * sr // 50 minutes
	pcm := make([]float32, n)
	for i := range pcm {
		pcm[i] = float32(0.3 * math.Sin(float64(i)*0.05))
	}
	// A pause 3 s before the first 20-minute mark, and one 2 s after where the
	// second piece's would then fall.
	quiet := func(at int) {
		for i := at; i < at+sr/2; i++ {
			pcm[i] = 0
		}
	}
	first := 1200*sr - 3*sr
	quiet(first)
	second := first + sr/4 + 1200*sr + 2*sr
	quiet(second)

	parts := pieces(pcm)
	if len(parts) != 3 {
		t.Fatalf("%d pieces of 50 minutes, want 3", len(parts))
	}
	total, at := 0, []int{}
	for _, p := range parts {
		total += len(p)
		at = append(at, total)
	}
	if total != n {
		t.Errorf("the pieces are %d samples of %d", total, n)
	}
	if at[0] < first || at[0] > first+sr/2 {
		t.Errorf("the first cut is at %.2f s, the pause at %.2f-%.2f s", float64(at[0])/sr, float64(first)/sr, float64(first+sr/2)/sr)
	}
	if at[1] < second || at[1] > second+sr/2 {
		t.Errorf("the second cut is at %.2f s, the pause at %.2f-%.2f s", float64(at[1])/sr, float64(second)/sr, float64(second+sr/2)/sr)
	}
	// Twenty minutes and under: one piece, the recording itself.
	if p := pieces(pcm[:1200*sr]); len(p) != 1 || len(p[0]) != 1200*sr {
		t.Errorf("twenty minutes came out as %d pieces", len(p))
	}
}

func TestPiecesJoin(t *testing.T) {
	for _, c := range []struct{ before, want string }{
		{"", ""}, {"He said so.", " "}, {"他这样说。", ""}, {"そうです", ""}, {"mixed 中文", ""}, {"end with space ", " "},
	} {
		if got := joint(c.before); got != c.want {
			t.Errorf("after %q: %q, want %q", c.before, got, c.want)
		}
	}
	if got := afterOpener("language English<asr_text>Hello."); got != "Hello." {
		t.Errorf("afterOpener: %q", got)
	}
	if got := afterOpener("Hello."); got != "Hello." {
		t.Errorf("afterOpener without one: %q", got)
	}
}
