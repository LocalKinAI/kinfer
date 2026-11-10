package mlxasr

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
)

// With KINFER_SV_MODEL (an MLX SenseVoice directory with am.mvn beside it) and
// KINFER_SV_REF (sv_ref.py's output from mlx-audio with the same am.mvn), the
// features, the CTC head and the transcript are checked against mlx-audio's.
func TestSenseVoiceAgainstMLXAudio(t *testing.T) {
	dir, refs := os.Getenv("KINFER_SV_MODEL"), os.Getenv("KINFER_SV_REF")
	if dir == "" || refs == "" {
		t.Skip("KINFER_SV_MODEL and KINFER_SV_REF not set")
	}
	libs, err := mlxengine.Libraries()
	if err != nil {
		t.Fatal(err)
	}
	if err := mlx.Open(libs); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	s, err := LoadSenseVoice(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Free()
	t.Logf("loaded in %v, CMVN %v, %d pieces", time.Since(t0).Round(time.Millisecond), s.cmvn, len(s.pieces))

	cases, _ := filepath.Glob(filepath.Join(refs, "*", "ref.json"))
	for _, rj := range cases {
		rd := filepath.Dir(rj)
		name := filepath.Base(rd)
		var ref struct {
			Text     string `json:"mlx_text"`
			Language string `json:"mlx_language"`
			Emotion  string `json:"mlx_emotion"`
			Event    string `json:"mlx_event"`
			FunASR   string `json:"funasr_raw"`
		}
		b, _ := os.ReadFile(rj)
		json.Unmarshal(b, &ref)
		pcm, _ := readNpy(t, filepath.Join(rd, "pcm.npy"))

		func() {
			sc := mlx.NewScope()
			defer sc.Close()
			feats, err := s.features(pcm)
			if err != nil {
				t.Fatal(err)
			}
			want, _ := readNpy(t, filepath.Join(rd, "features.npy"))
			got := mlx.Contiguous(feats).Float32s()
			if len(got) != len(want) {
				t.Errorf("%s: features %v, mlx-audio's %d values", name, feats.Shape(), len(want))
				return
			}
			d, c := compare(got, want)
			lp := s.logProbs(feats, 0, true)
			head := mlx.Contiguous(mlx.Slice(lp, []int{0, 0}, []int{4, s.cfg.Vocab})).Float32s()
			wantHead, _ := readNpy(t, filepath.Join(rd, "head.npy"))
			dh, _ := compare(head, wantHead)
			ids := mlx.Astype(mlx.Argmax(lp, -1, false), mlx.Int32).Int32s()
			wantIDs, _ := readNpy(t, filepath.Join(rd, "argmax.npy"))
			same := 0
			for i := range ids {
				if i < len(wantIDs) && int32(wantIDs[i]) == ids[i] {
					same++
				}
			}
			t.Logf("%s: features %v max |Δ| %.2g cos %.7f; head log-probs max |Δ| %.2g; argmax %d/%d positions the same",
				name, feats.Shape(), d, c, dh, same, len(ids))
		}()

		h, err := s.Hear(context.Background(), pcm, "auto", true)
		if err != nil {
			t.Fatal(err)
		}
		emo := h.Emotion
		if emo == "" {
			emo = "unk"
		}
		event := "Speech"
		if len(h.Events) > 0 {
			event = h.Events[0]
		}
		ok := h.Text == strings.TrimSpace(ref.Text) && h.Language == ref.Language && emo == ref.Emotion && event == ref.Event
		t.Logf("%s: [%s %s %v] %v same as mlx-audio=%v\n   %q\n   funasr: %.90s", name, h.Language, emo, h.Events,
			h.Took.Round(time.Millisecond), ok, h.Text, ref.FunASR)
		if !ok {
			t.Errorf("%s: heard [%s %s %s] %q, mlx-audio [%s %s %s] %q", name, h.Language, emo, event, h.Text,
				ref.Language, ref.Emotion, ref.Event, ref.Text)
		}
	}
}
