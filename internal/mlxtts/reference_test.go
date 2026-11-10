package mlxtts

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
)

// readNpy reads a little-endian float32 or int32 .npy as float32s.
func readNpy(t *testing.T, path string) ([]float32, []int) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(b[1:6]) != "NUMPY" {
		t.Fatalf("%s is not .npy", path)
	}
	hl := int(binary.LittleEndian.Uint16(b[8:10]))
	header := string(b[10 : 10+hl])
	ints := strings.Contains(header, "'<i4'")
	if !ints && !strings.Contains(header, "'<f4'") {
		t.Fatalf("%s: %s, want <f4 or <i4", path, header)
	}
	var shape []int
	if m := regexp.MustCompile(`'shape': \(([^)]*)\)`).FindStringSubmatch(header); m != nil {
		for _, s := range strings.Split(m[1], ",") {
			if s = strings.TrimSpace(s); s != "" {
				n, _ := strconv.Atoi(s)
				shape = append(shape, n)
			}
		}
	}
	data := b[10+hl:]
	out := make([]float32, len(data)/4)
	for i := range out {
		u := binary.LittleEndian.Uint32(data[4*i:])
		if ints {
			out[i] = float32(int32(u))
		} else {
			out[i] = math.Float32frombits(u)
		}
	}
	return out, shape
}

func compare(a, b []float32) (maxAbs, cos float64) {
	var dot, na, nb float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		maxAbs = math.Max(maxAbs, d)
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return maxAbs, dot / math.Sqrt(na*nb)
}

func f32(a mlx.Array) []float32 { return mlx.Contiguous(mlx.Astype(a, mlx.Float32)).Float32s() }

// sameFrames is how many frames from the start a and b [frames][16] agree on.
func sameFrames(a [][]int32, b []float32) int {
	for i, f := range a {
		for g, c := range f {
			if 16*i+g >= len(b) || int32(b[16*i+g]) != c {
				return i
			}
		}
	}
	return len(a)
}

func codesOf(flat []float32) [][]int32 {
	var out [][]int32
	for i := 0; i+16 <= len(flat); i += 16 {
		f := make([]int32, 16)
		for g := range f {
			f[g] = int32(flat[i+g])
		}
		out = append(out, f)
	}
	return out
}

// With KINFER_TTS_MODEL (an MLX Qwen3-TTS CustomVoice directory) and
// KINFER_TTS_REF (tts_ref.py's output from mlx-audio), each part is checked
// against mlx-audio's: the prompt, the first step's logits, the code
// predictor's, the codes greedy and drawn with seed 7, and the audio.
func TestAgainstMLXAudio(t *testing.T) {
	dir, refs := os.Getenv("KINFER_TTS_MODEL"), os.Getenv("KINFER_TTS_REF")
	if dir == "" || refs == "" {
		t.Skip("KINFER_TTS_MODEL and KINFER_TTS_REF not set")
	}
	libs, err := mlxengine.Libraries()
	if err != nil {
		t.Fatal(err)
	}
	if err := mlx.Open(libs); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	t.Logf("loaded in %v, %.2f GB, voices %v", time.Since(t0).Round(time.Millisecond), float64(m.Bytes())/1e9, m.Voices())

	cases, _ := filepath.Glob(filepath.Join(refs, "*", "ref.json"))
	for _, rj := range cases {
		rd := filepath.Dir(rj)
		name := filepath.Base(rd)
		var ref struct {
			Text, Voice, Language string
			IDs                   []int32 `json:"ids"`
			Seed7Seconds          float64 `json:"seed7_seconds"`
			Seed7Took             float64 `json:"seed7_took"`
		}
		b, _ := os.ReadFile(rj)
		json.Unmarshal(b, &ref)
		req := Request{Text: ref.Text, Voice: ref.Voice, Language: ref.Language}

		ids := m.tok.Encode("<|im_start|>assistant\n"+ref.Text+"<|im_end|>\n<|im_start|>assistant\n", true)
		if len(ids) != len(ref.IDs) {
			t.Errorf("%s: %d ids, mlx-audio %d", name, len(ids), len(ref.IDs))
		} else {
			for i := range ids {
				if ids[i] != ref.IDs[i] {
					t.Errorf("%s: id %d is %d, mlx-audio %d", name, i, ids[i], ref.IDs[i])
					break
				}
			}
		}

		func() {
			sc := mlx.NewScope()
			defer sc.Close()
			prompt, trailing, pad, err := m.inputs(req.withDefaults())
			if err != nil {
				t.Fatal(err)
			}
			for _, x := range []struct {
				name string
				a    mlx.Array
			}{{"input_embeds", prompt}, {"trailing", trailing}, {"pad", pad}} {
				want, shape := readNpy(t, filepath.Join(rd, x.name+".npy"))
				got := f32(x.a)
				if len(got) != len(want) {
					t.Errorf("%s: %s %v, mlx-audio %v", name, x.name, x.a.Shape(), shape)
					continue
				}
				d, c := compare(got, want)
				t.Logf("%s: %s %v max |Δ| %.3g cos %.7f", name, x.name, x.a.Shape(), d, c)
			}
			cache := newCache(m.cfg.Talker.Layers)
			logits, hidden := m.talker.step(prompt, prompt.Shape()[1], cache)
			want, _ := readNpy(t, filepath.Join(rd, "logits0.npy"))
			d, c := compare(f32(logits), want)
			wantH, _ := readNpy(t, filepath.Join(rd, "hidden0.npy"))
			dh, ch := compare(f32(hidden), wantH)
			t.Logf("%s: first logits max |Δ| %.3g cos %.7f; hidden max |Δ| %.3g cos %.7f", name, d, c, dh, ch)
			tc := m.cfg.Talker
			sf := newSampling(0, 50, 1.05, tc.CodecEOS, tc.Vocab, logits.Dtype())
			defer sf.free()
			first := sf.draw(sf.penalize(logits, mlx.Full(mlx.Int32, -1, recentCodes)))
			pc := newCache(tc.CodePredictor.Layers)
			cl := m.talker.predict(mlx.Concatenate(1, hidden, m.talker.codec(first)), 2, pc, 0)
			wantC, _ := readNpy(t, filepath.Join(rd, "cp_logits0.npy"))
			dc, cc := compare(f32(cl), wantC)
			t.Logf("%s: code predictor's first logits max |Δ| %.3g cos %.7f", name, dc, cc)
		}()

		// The decoder alone, on mlx-audio's greedy codes.
		gc, _ := readNpy(t, filepath.Join(rd, "greedy_codes.npy"))
		ga, _ := readNpy(t, filepath.Join(rd, "greedy_audio.npy"))
		pcm := m.decode(codesOf(gc))
		if len(pcm) != len(ga) {
			t.Errorf("%s: decoded %d samples, mlx-audio %d", name, len(pcm), len(ga))
		} else {
			d, c := compare(pcm, ga)
			t.Logf("%s: decoder on mlx-audio's codes: %d samples max |Δ| %.3g cos %.7f", name, len(pcm), d, c)
		}
		// Streamed in pieces of 7 frames, as the Python service streams.
		st := newCodecState(m.codec.cfg.Layers)
		var streamed []float32
		cs := codesOf(gc)
		for i := 0; i < len(cs); i += 7 {
			streamed = append(streamed, m.decodePiece(cs[i:min(i+7, len(cs))], st)...)
		}
		st.free()
		if n := min(len(streamed), len(pcm)); n > 0 {
			d, c := compare(streamed[:n], pcm[:n])
			t.Logf("%s: streamed 7 frames a piece vs whole: %d/%d samples max |Δ| %.3g cos %.7f", name, len(streamed), len(pcm), d, c)
		}

		// Greedy codes.
		var got []float32
		res, err := m.Speak(context.Background(), Request{Text: ref.Text, Voice: ref.Voice, Language: ref.Language, Temperature: 0, MaxFrames: 120},
			func(p []float32) error { got = p; return nil })
		if err != nil {
			t.Fatal(err)
		}
		// Greedy codes part after a few frames: kinfer's MLX is not the
		// Python service's, and a near tie in the code predictor goes the
		// other way now and then (TestTeacherForced measures how close the
		// logits are). Logged, not held to.
		same := sameFrames(res.Codes, gc)
		t.Logf("%s: greedy %d frames (mlx-audio %d), the first %d the same", name, res.Frames, len(gc)/16, same)

		// Drawn with seed 7, the defaults.
		sc, _ := readNpy(t, filepath.Join(rd, "seed7_codes.npy"))
		sa, _ := readNpy(t, filepath.Join(rd, "seed7_audio.npy"))
		mlx.Seed(7)
		t1 := time.Now()
		res, err = m.Speak(context.Background(), Request{Text: ref.Text, Voice: ref.Voice, Language: ref.Language, Temperature: 0.9, MaxFrames: 2048},
			func(p []float32) error { got = p; return nil })
		took := time.Since(t1)
		if err != nil {
			t.Fatal(err)
		}
		same = sameFrames(res.Codes, sc)
		secs := float64(len(got)) / sampleRate
		t.Logf("%s: seed 7: %d frames (mlx-audio %d), the first %d the same; %.2f s of audio in %v (mlx-audio %.2f s in %.2f s): prefill %v, talk %v, decode %v",
			name, res.Frames, len(sc)/16, same, secs, took.Round(time.Millisecond), ref.Seed7Seconds, ref.Seed7Took,
			res.Prefill.Round(time.Millisecond), res.Talk.Round(time.Millisecond), res.Decode.Round(time.Millisecond))
		if same == len(sc)/16 && len(got) == len(sa) {
			d, c := compare(got, sa)
			t.Logf("%s: seed 7 audio max |Δ| %.3g cos %.7f", name, d, c)
		}
	}
}

// Fed mlx-audio's greedy codes frame by frame, so that a near tie drawn the
// other way cannot carry on, each frame's logits are checked against
// mlx-audio's: the talker's and the code predictor's fifteen.
func TestTeacherForced(t *testing.T) {
	dir, refs := os.Getenv("KINFER_TTS_MODEL"), os.Getenv("KINFER_TTS_REF")
	if dir == "" || refs == "" {
		t.Skip("KINFER_TTS_MODEL and KINFER_TTS_REF not set")
	}
	libs, err := mlxengine.Libraries()
	if err != nil {
		t.Fatal(err)
	}
	if err := mlx.Open(libs); err != nil {
		t.Fatal(err)
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	tc := m.cfg.Talker
	cases, _ := filepath.Glob(filepath.Join(refs, "*", "ref.json"))
	for _, rj := range cases {
		rd := filepath.Dir(rj)
		name := filepath.Base(rd)
		var ref struct{ Text, Voice, Language string }
		b, _ := os.ReadFile(rj)
		json.Unmarshal(b, &ref)
		gc, _ := readNpy(t, filepath.Join(rd, "greedy_codes.npy"))
		codes := codesOf(gc)
		wantL, _ := readNpy(t, filepath.Join(rd, "greedy_logits.npy"))
		wantC, _ := readNpy(t, filepath.Join(rd, "greedy_cp_logits.npy"))
		V, CV := tc.Vocab, tc.CodePredictor.Vocab

		outer := mlx.NewScope()
		sf := newSampling(0, 50, 1.05, tc.CodecEOS, tc.Vocab, m.talker.main.dt)
		prompt, trailing, pad, err := m.inputs(Request{Text: ref.Text, Voice: ref.Voice, Language: ref.Language}.withDefaults())
		if err != nil {
			t.Fatal(err)
		}
		cache := newCache(tc.Layers)
		in := prompt
		worstT, worstC := 1.0, 1.0
		var maxT, maxC float64
		flipsT, flipsC := 0, 0
		var recent []int32
		for f := 0; f < len(codes); f++ {
			sc := mlx.NewScope()
			old := cache.arrays()
			L := 1
			if f == 0 {
				L = prompt.Shape()[1]
			}
			logits, hidden := m.talker.step(in, L, cache)
			got := f32(logits)
			d, c := compare(got, wantL[f*V:(f+1)*V])
			worstT, maxT = math.Min(worstT, c), math.Max(maxT, d)
			frameT := d
			frameC := 0.0
			win := make([]int32, recentCodes)
			for j := range win {
				win[j] = -1
			}
			for j, c := range recent[max(0, len(recent)-recentCodes):] {
				win[recentCodes-min(len(recent), recentCodes)+j] = c
			}
			first := sf.draw(sf.penalize(logits, mlx.FromInt32(win, recentCodes)))
			if int32(mlx.Astype(first, mlx.Int32).Int32s()[0]) != codes[f][0] {
				flipsT++
			}
			ref0 := mlx.FromInt32([]int32{codes[f][0]}, 1, 1)
			pc := newCache(tc.CodePredictor.Layers)
			for g := 0; g < tc.CodePredictor.Groups-1; g++ {
				x, n := mlx.Array(0), 1
				if g == 0 {
					x, n = mlx.Concatenate(1, hidden, m.talker.codec(ref0)), 2
				} else {
					x = rows(m.talker.cpEmbed[g-1], mlx.FromInt32([]int32{codes[f][g]}, 1, 1))
				}
				cl := m.talker.predict(x, n, pc, g)
				off := (f*(tc.CodePredictor.Groups-1) + g) * CV
				d, c := compare(f32(cl), wantC[off:off+CV])
				worstC, maxC = math.Min(worstC, c), math.Max(maxC, d)
				if d > 0 && frameC == 0 && os.Getenv("KINFER_TTS_DETAIL") != "" {
					t.Logf("%s: frame %d: the predictor's first difference at group %d (%.3g)", name, f, g, d)
				}
				frameC = math.Max(frameC, d)
				if mlx.Astype(mlx.Argmax(cl, -1, false), mlx.Int32).Int32s()[0] != codes[f][g+1] {
					flipsC++
				}
			}
			if os.Getenv("KINFER_TTS_DETAIL") != "" && (frameT > 0 || frameC > 0) {
				t.Logf("%s: frame %d talker max |Δ| %.3g, predictor %.3g", name, f, frameT, frameC)
			}
			var text mlx.Array
			if f < trailing.Shape()[1] {
				text = mlx.Slice(trailing, []int{0, f, 0}, []int{1, f + 1, tc.Hidden})
			} else {
				text = pad
			}
			next := m.talker.codec(ref0)
			for g := 1; g < 16; g++ {
				next = mlx.Add(next, rows(m.talker.cpEmbed[g-1], mlx.FromInt32([]int32{codes[f][g]}, 1, 1)))
			}
			next = mlx.Add(text, next)
			mlx.Eval(next)
			for _, a := range old {
				a.Free()
			}
			sc.Escape(cache.arrays()...)
			sc.Escape(next)
			sc.Close()
			if in != prompt {
				in.Free()
			}
			in = next
			recent = append(recent, codes[f][0])
		}
		cache.free()
		sf.free()
		outer.Close()
		t.Logf("%s: %d frames teacher-forced: talker logits worst cos %.7f max |Δ| %.3g, %d first codes drawn otherwise; code predictor worst cos %.7f max |Δ| %.3g, %d of %d codes otherwise",
			name, len(codes), worstT, maxT, flipsT, worstC, maxC, flipsC, len(codes)*15)
		// One bfloat16 step on a logit, now and then, is two MLX builds'
		// rounding; anything more is a mistake.
		if worstT < 0.9999 || flipsT > 0 || worstC < 0.999 {
			t.Errorf("%s: the logits are further from mlx-audio's than rounding", name)
		}
	}
}
