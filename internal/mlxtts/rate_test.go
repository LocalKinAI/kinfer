package mlxtts

import (
	"context"
	"os"
	"sort"
	"strconv"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
)

// With KINFER_TTS_MODEL and KINFER_TTS_RATE=n: one sentence spoken with seeds
// 1..n, and how long each came out. With one text token given a frame, the
// same sentence could come out at half the speed (mlx-audio #1036).
func TestSpeakingRate(t *testing.T) {
	dir, n := os.Getenv("KINFER_TTS_MODEL"), os.Getenv("KINFER_TTS_RATE")
	if dir == "" || n == "" {
		t.Skip("KINFER_TTS_MODEL and KINFER_TTS_RATE not set")
	}
	seeds, _ := strconv.Atoi(n)
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
	voice := "serena"
	if m.Designs() {
		voice = ""
	}
	var secs []float64
	stopped := 0
	for s := 1; s <= seeds; s++ {
		sd := uint64(s)
		res, err := m.Speak(context.Background(), Request{Text: "明天九点朝阳公园东门集合，身份证三千五百元，幺三八幺二三四五六七八，取消周五改下周一。",
			Voice: voice, Language: "chinese", Instruct: os.Getenv("KINFER_TTS_INSTRUCT"), Temperature: 0.9, MaxFrames: 512, Seed: &sd}, func([]float32) error { return nil })
		if err != nil {
			t.Fatal(err)
		}
		if res.Stopped {
			stopped++
		}
		secs = append(secs, float64(res.Samples)/sampleRate)
	}
	sorted := append([]float64{}, secs...)
	sort.Float64s(sorted)
	slow := 0
	for _, s := range secs {
		if s > 1.5*sorted[len(sorted)/2] {
			slow++
		}
	}
	t.Logf("%d seeds: %.2f s shortest, %.2f s median, %.2f s longest; %d over 1.5x the median, %d cut at 512 frames", seeds, sorted[0], sorted[len(sorted)/2], sorted[len(sorted)-1], slow, stopped)
	t.Logf("each: %.2f", secs)
}
