package mlxembed

import (
	"encoding/binary"
	"math"
	"testing"
)

// A picture is resized to whole 48-pixel cells within 280 soft tokens, its
// shape kept: the sizes Ollama's processor gives, which the tests on the box
// matched token for token.
func TestVisionTargetSize(t *testing.T) {
	for _, c := range []struct{ h, w, wantH, wantW int }{
		{750, 1000, 672, 912}, // 19 x 14 = 266 soft tokens: the 270 tokens Ollama read
		{1400, 600, 1200, 480},
		{90, 120, 672, 912},
		{800, 800, 768, 768},
	} {
		h, w, err := visionTargetSize(c.h, c.w, 16, 3, 280)
		if err != nil || h != c.wantH || w != c.wantW {
			t.Errorf("%dx%d → %dx%d (%v), want %dx%d", c.w, c.h, w, h, err, c.wantW, c.wantH)
		}
		if (h/48)*(w/48) > 280 {
			t.Errorf("%dx%d: %d soft tokens, more than 280", c.w, c.h, (h/48)*(w/48))
		}
	}
}

func TestPatchifyLaysOutRowsOfPatches(t *testing.T) {
	w, h := 96, 48
	rgb := make([]uint8, w*h*3)
	rgb[(0*w+16)*3] = 255 // patch 1's first pixel, red
	in := patchify(rgb, w, h, 16, 3)
	if in.patchesW != 6 || in.patchesH != 3 || in.softTokens != 2 || in.pixelsPerPatch != 768 {
		t.Fatalf("%+v", in)
	}
	if in.pixels[768] != 1 || in.positions[2] != 1 || in.positions[3] != 0 {
		t.Errorf("patch 1: red %v at (%d, %d)", in.pixels[768], in.positions[2], in.positions[3])
	}
}

func wav16(rate int, samples []int16) []byte {
	b := make([]byte, 44+2*len(samples))
	copy(b, "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(36+2*len(samples)))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], 1)
	binary.LittleEndian.PutUint32(b[24:], uint32(rate))
	binary.LittleEndian.PutUint32(b[28:], uint32(2*rate))
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(2*len(samples)))
	for i, s := range samples {
		binary.LittleEndian.PutUint16(b[44+2*i:], uint16(s))
	}
	return b
}

// A second of a 440 Hz tone at 24 kHz: read, made 16 kHz, cut into frames of
// 10 ms, and four times fewer tokens after the tower's two strided
// convolutions.
func TestPrepareAudio(t *testing.T) {
	s := make([]int16, 24000)
	for i := range s {
		s[i] = int16(8000 * math.Sin(2*math.Pi*440*float64(i)/24000))
	}
	ps, err := prepareAudio(wav16(24000, s))
	if err != nil || len(ps) != 1 {
		t.Fatalf("%v, %d pieces", err, len(ps))
	}
	if ps[0].frames != 99 || ps[0].tokens != 25 || len(ps[0].mel) != 99*audioMelBins {
		t.Errorf("frames %d, tokens %d, mel %d", ps[0].frames, ps[0].tokens, len(ps[0].mel))
	}
	// The tone's energy is in the mel bins around 440 Hz, not at the top.
	low, high := ps[0].mel[50*audioMelBins+20], ps[0].mel[50*audioMelBins+120]
	if low <= high {
		t.Errorf("mel at 440 Hz %.2f, near 8 kHz %.2f", low, high)
	}
	if _, err := prepareAudio([]byte("RIFF\x00\x00\x00\x00WAVE")); err == nil {
		t.Error("a WAV with no chunks was read")
	}
}

// A long sound is cut into pieces of at most 30 s, at its quietest moments.
func TestSplitQuiet(t *testing.T) {
	s := make([]float32, 16000*70)
	for i := range s {
		s[i] = 0.5
	}
	for i := 16000 * 34; i < 16000*35; i++ { // a second of silence near the even share
		s[i] = 0
	}
	ps := splitQuiet(s, 16000, 30)
	if len(ps) != 3 {
		t.Fatalf("%d pieces, want 3", len(ps))
	}
	total := 0
	for _, p := range ps {
		if len(p) > 16000*30 {
			t.Errorf("a piece of %.1f s", float64(len(p))/16000)
		}
		total += len(p)
	}
	if total != len(s) {
		t.Errorf("pieces hold %d samples of %d", total, len(s))
	}
}
