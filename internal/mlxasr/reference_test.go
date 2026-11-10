package mlxasr

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
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

// readNpy reads a little-endian float32 .npy file.
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
	if !strings.Contains(header, "'<f4'") {
		t.Fatalf("%s: %s, want <f4", path, header)
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
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
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

// With KINFER_ASR_MODEL (an MLX Qwen3-ASR directory) and KINFER_ASR_REF (the
// directories ref_dump.py wrote from mlx-audio), each step is checked against
// mlx-audio's: the features, the encoder's output, the prompt, the transcript.
func TestAgainstMLXAudio(t *testing.T) {
	dir, refs := os.Getenv("KINFER_ASR_MODEL"), os.Getenv("KINFER_ASR_REF")
	if dir == "" || refs == "" {
		t.Skip("KINFER_ASR_MODEL and KINFER_ASR_REF not set")
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
	t.Logf("loaded in %v, %.2f GB", time.Since(t0).Round(time.Millisecond), float64(m.Bytes())/1e9)

	cases, _ := filepath.Glob(filepath.Join(refs, "*", "ref.json"))
	for _, rj := range cases {
		rd := filepath.Dir(rj)
		var ref struct {
			NAudio    int     `json:"n_audio_tokens"`
			PromptIDs []int32 `json:"prompt_ids"`
			Text      string  `json:"text"`
		}
		b, _ := os.ReadFile(rj)
		json.Unmarshal(b, &ref)
		pcm, _ := readNpy(t, filepath.Join(rd, "pcm.npy"))
		name := filepath.Base(rd)

		func() {
			s := mlx.NewScope()
			defer s.Close()
			feats, T, err := logMel(pcm, m.cfg.Thinker.Audio.Mels)
			if err != nil {
				t.Fatal(err)
			}
			want, wshape := readNpy(t, filepath.Join(rd, "features.npy"))
			got := mlx.Contiguous(mlx.Astype(feats, mlx.Float32)).Float32s()
			if len(got) != len(want) {
				t.Errorf("%s: features %d values (T %d), mlx-audio's %v", name, len(got), T, wshape)
				return
			}
			d, c := compare(got, want)
			t.Logf("%s: features [%d, %d] max |Δ| %.2g, cosine %.8f", name, m.cfg.Thinker.Audio.Mels, T, d, c)

			af := m.enc.forward(feats, T)
			wantAF, ashape := readNpy(t, filepath.Join(rd, "audio_features.npy"))
			gotAF := mlx.Astype(af, mlx.Float32).Float32s()
			if len(gotAF) != len(wantAF) {
				t.Errorf("%s: encoder %v, mlx-audio's %v", name, af.Shape(), ashape)
				return
			}
			d, c = compare(gotAF, wantAF)
			t.Logf("%s: encoder %v max |Δ| %.3g, cosine %.8f", name, af.Shape(), d, c)

			// The encoder fed the reference features, to tell its error from theirs.
			wf := mlx.FromFloat32(want, m.cfg.Thinker.Audio.Mels, len(want)/m.cfg.Thinker.Audio.Mels)
			gotAF2 := mlx.Astype(m.enc.forward(wf, len(want)/m.cfg.Thinker.Audio.Mels), mlx.Float32).Float32s()
			d, c = compare(gotAF2, wantAF)
			t.Logf("%s: encoder on mlx-audio's features max |Δ| %.3g, cosine %.8f", name, d, c)
		}()

		pre, post := m.prompt("", "", ref.NAudio)
		ids := append(append(append([]int32{}, pre...), make([]int32, ref.NAudio)...), post...)
		for i := range ids[len(pre) : len(pre)+ref.NAudio] {
			ids[len(pre)+i] = m.cfg.Thinker.AudioToken
		}
		if fmt.Sprint(ids) != fmt.Sprint(ref.PromptIDs) {
			t.Errorf("%s: prompt\n got %v\nwant %v", name, ids, ref.PromptIDs)
		}

		for i := 0; i < 3; i++ {
			res, err := m.Transcribe(context.Background(), pcm, "", "", 8192, nil)
			if err != nil {
				t.Fatal(err)
			}
			text := asrBody(res.Text)
			same := text == ref.Text
			t.Logf("%s run %d: encode %v prefill %v (%d tokens) decode %v (%d tokens, %.0f tok/s) same=%v\n   %q",
				name, i, res.Encode.Round(time.Millisecond), res.Prefill.Round(time.Millisecond), res.PromptTokens,
				res.Decode.Round(time.Millisecond), res.Tokens, float64(res.Tokens)/res.Decode.Seconds(), same, res.Text)
			if !same {
				t.Errorf("%s: transcript %q, mlx-audio's %q", name, text, ref.Text)
			}
			mem := mlx.MemoryUse()
			t.Logf("   memory: active %.2f GB, cache %.2f GB, peak %.2f GB", gb(mem.Active), gb(mem.Cache), gb(mem.Peak))
		}
	}
	if long := os.Getenv("KINFER_ASR_LONG"); long != "" {
		b, err := os.ReadFile(long)
		if err != nil {
			t.Fatal(err)
		}
		pcm, err := Decode(b)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 4; i++ {
			res, err := m.Transcribe(context.Background(), pcm, "", "", 8192, nil)
			if err != nil {
				t.Fatal(err)
			}
			mem := mlx.MemoryUse()
			t.Logf("%s run %d: %d audio tokens, encode %v prefill %v decode %v (%d tokens); active %.2f GB, cache %.2f GB, peak %.2f GB",
				filepath.Base(long), i, res.AudioTokens, res.Encode.Round(time.Millisecond), res.Prefill.Round(time.Millisecond),
				res.Decode.Round(time.Millisecond), res.Tokens, gb(mem.Active), gb(mem.Cache), gb(mem.Peak))
		}
	}
}

func gb(n int64) float64 { return float64(n) / 1e9 }

// asrBody is a reply without its "language X<asr_text>" opening.
func asrBody(s string) string {
	if i := strings.Index(s, "<asr_text>"); i >= 0 && strings.HasPrefix(s, "language ") {
		return strings.TrimSpace(s[i+len("<asr_text>"):])
	}
	return strings.TrimSpace(s)
}
