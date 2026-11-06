package main

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
	"github.com/LocalKinAI/kinfer/internal/mlxmodel"
	"github.com/LocalKinAI/kinfer/internal/tokenizer"
)

// smokeTest reads the problems with both models and says how far their
// predictions part, position by position: the KL divergence of the model's
// next-token distribution from the other's, and how often the two put the
// same token first. Nothing is generated, so it takes seconds: an edit that
// changed nothing reads 0, and one that broke the model reads as no other
// does.
func smokeTest(path, other string, probs []problem, suffix string) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if !mlxengine.IsModelDir(path) || !mlxengine.IsModelDir(other) {
		return fmt.Errorf("a smoke test compares two models kinfer serves on MLX")
	}
	ta, _ := os.ReadFile(filepath.Join(other, "tokenizer.json"))
	tb, _ := os.ReadFile(filepath.Join(path, "tokenizer.json"))
	if !bytes.Equal(ta, tb) {
		return fmt.Errorf("%s and %s tokenize differently, so their predictions do not line up", modelName(path), modelName(other))
	}
	tok, err := tokenizer.Load(filepath.Join(path, "tokenizer.json"))
	if err != nil {
		return err
	}
	lib, err := mlxengine.Libraries()
	if err != nil {
		return err
	}
	if err := mlx.Open(lib); err != nil {
		return err
	}
	start := time.Now()
	ref, err := mlxmodel.LoadQwen35(other)
	if err != nil {
		return err
	}
	defer ref.Free()
	m, err := mlxmodel.LoadQwen35(path)
	if err != nil {
		return err
	}
	defer m.Free()

	var kls []float64
	agree := 0
	worstID, worst := "", -1.0
	for _, p := range probs {
		kl, same, err := compareLogits(ref, m, tok.Encode(p.text+suffix, false))
		if err != nil {
			return fmt.Errorf("%s: %w", p.id, err)
		}
		agree += same
		sum := 0.0
		for _, k := range kl {
			v := math.Max(0, float64(k)) // a rounding error below zero is none
			kls = append(kls, v)
			sum += v
		}
		if mean := sum / float64(len(kl)); mean > worst {
			worstID, worst = p.id, mean
		}
	}
	sorted := append([]float64(nil), kls...)
	sort.Float64s(sorted)
	at := func(q float64) float64 { return sorted[min(len(sorted)-1, int(q*float64(len(sorted))))] }
	mean := 0.0
	for _, k := range kls {
		mean += k
	}
	mean /= float64(len(kls))

	fmt.Printf("%s against %s: %d problems, %d positions, %v\n\n", modelName(path), modelName(other),
		len(probs), len(kls), time.Since(start).Round(100*time.Millisecond))
	fmt.Printf("  same most likely next token   %.2f%% of positions\n", 100*float64(agree)/float64(len(kls)))
	fmt.Printf("  KL divergence per position    mean %.4g, median %.4g, 99th percentile %.4g, most %.4g\n",
		mean, at(0.5), at(0.99), sorted[len(sorted)-1])
	if sorted[len(sorted)-1] == 0 {
		fmt.Printf("\n  The two predict exactly the same on these texts: the edit changes nothing they reach.\n")
	} else {
		fmt.Printf("  most changed problem          %s (mean KL %.4g)\n", worstID, worst)
	}
	return nil
}

// compareLogits reads ids with both models and returns, per position, the KL
// divergence of m's prediction from ref's, and at how many positions the two
// put the same token first.
func compareLogits(ref, m *mlxmodel.Qwen35, ids []int32) (kl []float32, agree int, err error) {
	defer mlx.Recover(&err)
	la, err := ref.Logits(ids)
	if err != nil {
		return nil, 0, err
	}
	defer la.Free()
	lb, err := m.Logits(ids)
	if err != nil {
		return nil, 0, err
	}
	defer lb.Free()
	s := mlx.NewScope()
	defer s.Close()
	a, b := mlx.Astype(la, mlx.Float32), mlx.Astype(lb, mlx.Float32)
	lpa := mlx.Subtract(a, mlx.Logsumexp(a, -1, true))
	lpb := mlx.Subtract(b, mlx.Logsumexp(b, -1, true))
	k := mlx.Sum(mlx.Multiply(mlx.Exp(lpa), mlx.Subtract(lpa, lpb)), -1, false)
	same := mlx.Sum(mlx.Astype(mlx.Equal(mlx.Argmax(a, -1, false), mlx.Argmax(b, -1, false)), mlx.Int32), 0, false)
	return k.Float32s(), int(same.Int32s()[0]), nil
}
