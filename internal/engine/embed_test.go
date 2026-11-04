package engine

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/llama"
)

// A text cut to fit keeps its closing marker, which the model was taught to
// read at a text's end; one that had none is cut plainly.
func TestCutKeepsTheClosingMarker(t *testing.T) {
	toks := []llama.Token{101, 7, 8, 9, 10, 102}
	if got := cut(toks, 4, true); !slices.Equal(got, []llama.Token{101, 7, 8, 102}) {
		t.Errorf("cut with a closing marker = %v", got)
	}
	if got := cut(toks, 4, false); !slices.Equal(got, []llama.Token{101, 7, 8, 9}) {
		t.Errorf("cut without one = %v", got)
	}
	if toks[3] != 9 {
		t.Error("cut wrote into the caller's tokens")
	}
}

// Texts go into passes in order, as many as fit a pass's tokens and its
// sequences, and none is split.
func TestPasses(t *testing.T) {
	texts := [][]llama.Token{make([]llama.Token, 5), make([]llama.Token, 4), make([]llama.Token, 3), make([]llama.Token, 9), {1}, {1}, {1}}
	got := passes(texts, 9, 3)
	want := [][2]int{{0, 2}, {2, 3}, {3, 4}, {4, 7}}
	if !slices.Equal(got, want) {
		t.Errorf("passes = %v, want %v", got, want)
	}
	// Short texts fill a pass's sequences before its tokens.
	ones := [][]llama.Token{{1}, {1}, {1}, {1}, {1}}
	if got := passes(ones, 9, 3); !slices.Equal(got, [][2]int{{0, 3}, {3, 5}}) {
		t.Errorf("five one-token texts, three to a pass: %v", got)
	}
	if got := passes(nil, 9, 3); len(got) != 0 {
		t.Errorf("no texts, passes %v", got)
	}
}

func TestNormalizeAndShorten(t *testing.T) {
	v := Normalize([]float32{3, 4, 0, 0})
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Errorf("Normalize(3, 4, 0, 0) = %v", v)
	}
	if z := Normalize([]float32{0, 0}); z[0] != 0 || z[1] != 0 {
		t.Errorf("a zero vector became %v", z)
	}
	s := Shorten([]float32{0.6, 0.8, 0, 0}, 1)
	if len(s) != 1 || s[0] != 1 {
		t.Errorf("Shorten to 1 = %v, want (1)", s)
	}
	if s := Shorten([]float32{0.6, 0.8}, 0); len(s) != 2 {
		t.Errorf("Shorten to 0 kept %d values, want all", len(s))
	}
}

// A picture for a model with no projector is refused, not described in words
// the model may answer around.
func TestPicturesForAModelThatCannotSeeThem(t *testing.T) {
	s := &scheduler{}
	j := &job{ctx: context.Background(), frags: make(chan string, 1), done: make(chan struct{}),
		msgs: []chat.Message{{Role: "user", Content: ImageMarker + "\nwhat is it?", Images: [][]byte{{1}}}}}
	if s.prepare(j) {
		t.Fatal("prepare took a picture for a model with no projector")
	}
	if !errors.Is(j.err, ErrNoPictures) {
		t.Errorf("finished with %v, want ErrNoPictures", j.err)
	}
	var bad PictureError
	if !errors.As(j.err, &bad) {
		t.Errorf("%v is not a PictureError, so a server would call it its own fault", j.err)
	}
}
