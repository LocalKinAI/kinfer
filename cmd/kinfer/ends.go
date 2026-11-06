package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
	"github.com/LocalKinAI/kinfer/internal/mlxmodel"
)

// endsChunk is how many positions' logits are held at once: a gigabyte of
// them in float32 at Qwen3.5's vocabulary.
const endsChunk = 1024

// endsTest reads the replies that ran out of tokens in the other model's full
// tier — thinking that never stopped, or an answer that never did — with both
// models, and asks how readily each would have ended them. At every position
// it takes the probability of the token that ends the part the reply is in:
// </think> while it thinks, the end of the reply after. Their sum over a
// reply is how many times the model would have ended it there, on average,
// had it read that far; the test compares the two sums reply by reply.
// Nothing is generated, so a reply of window tokens costs one read of them.
//
// The other model's sums are kept beside its results, so each edit tested
// against it costs one model's read.
func endsTest(path, other, results string, probs []problem, suffix string, window, traces int) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if !mlxengine.IsModelDir(path) || !mlxengine.IsModelDir(other) {
		return fmt.Errorf("an ends test compares two models kinfer serves on MLX")
	}
	ta, _ := os.ReadFile(filepath.Join(other, "tokenizer.json"))
	tb, _ := os.ReadFile(filepath.Join(path, "tokenizer.json"))
	if !bytes.Equal(ta, tb) {
		return fmt.Errorf("%s and %s tokenize differently, so one cannot read the other's replies", modelName(path), modelName(other))
	}
	res, err := readResults(results)
	if err != nil {
		return fmt.Errorf("the ends test reads %s's full-tier replies, and %w; run kinfer eval -tier full first", modelName(other), err)
	}
	text := map[string]string{}
	for _, p := range probs {
		text[p.id] = p.text
	}
	var cut []evalResult
	for _, r := range res {
		if r.Cut && text[r.ID] != "" {
			cut = append(cut, r)
		}
	}
	if len(cut) == 0 {
		return fmt.Errorf("none of %s's full-tier replies ran out of tokens, so there is nothing to end", modelName(other))
	}
	// The same replies every time, whatever order the results are in.
	hash := func(s string) uint64 { h := fnv.New64a(); h.Write([]byte(s)); return h.Sum64() }
	sort.Slice(cut, func(i, j int) bool { return hash(cut[i].ID) < hash(cut[j].ID) })
	if traces > 0 && traces < len(cut) {
		cut = cut[:traces]
	}
	cacheFile := fmt.Sprintf("%s.ends-%d.json", results, window)
	cache := map[string]float64{}
	if b, err := os.ReadFile(cacheFile); err == nil {
		json.Unmarshal(b, &cache)
	}

	rd, err := mlxengine.NewReader(path)
	if err != nil {
		return err
	}
	endThink, ok1 := rd.Tok.ID("</think>")
	endReply, ok2 := rd.Tok.ID("<|im_end|>")
	if !ok1 || !ok2 {
		return fmt.Errorf("%s's tokenizer has no </think> or <|im_end|>", modelName(path))
	}
	lib, err := mlxengine.Libraries()
	if err != nil {
		return err
	}
	if err := mlx.Open(lib); err != nil {
		return err
	}
	start := time.Now()
	var ref *mlxmodel.Qwen35
	for _, r := range cut {
		if _, ok := cache[r.ID]; !ok {
			if ref, err = mlxmodel.LoadQwen35(other); err != nil {
				return err
			}
			defer ref.Free()
			break
		}
	}
	m, err := mlxmodel.LoadQwen35(path)
	if err != nil {
		return err
	}
	defer m.Free()

	var sumRef, sumM, logRatio float64
	var positions, more, less int
	for _, r := range cut {
		prompt, written := rd.Reply([]chat.Message{{Role: "user", Content: text[r.ID] + suffix}}, true, r.Reply)
		written = written[:min(len(written), window)]
		seq := append(prompt, written...)
		// The token each position would end with: </think> until the reply
		// has closed its thinking, the end of the reply from then on.
		ends := make([]int32, len(seq))
		for i := range ends {
			ends[i] = endThink
		}
		for i := len(prompt); i < len(seq); i++ {
			if seq[i] == endThink {
				for j := i; j < len(seq); j++ {
					ends[j] = endReply
				}
				break
			}
		}
		// Position i predicts token i+1: the reply's positions are the
		// prompt's last and the reply's own, all but its last.
		sum := func(model *mlxmodel.Qwen35) (float64, error) {
			p, err := endProbs(model, seq, ends)
			if err != nil {
				return 0, err
			}
			s := 0.0
			for i := len(prompt) - 1; i < len(seq)-1; i++ {
				s += float64(p[i])
			}
			return s, nil
		}
		sa, ok := cache[r.ID]
		if !ok {
			if sa, err = sum(ref); err != nil {
				return err
			}
			cache[r.ID] = sa
		}
		sb, err := sum(m)
		if err != nil {
			return err
		}
		positions += len(seq) - len(prompt)
		sumRef += sa
		sumM += sb
		logRatio += math.Log(sb / sa)
		switch {
		case sb > sa:
			more++
		case sb < sa:
			less++
		}
	}

	if b, err := json.Marshal(cache); err == nil {
		os.WriteFile(cacheFile, b, 0o644)
	}
	per := func(s float64) float64 { return 1000 * s / float64(positions) }
	fmt.Printf("%s against %s: %d of the replies %s ran out of tokens on, their first %d tokens each, %d positions, %v\n\n",
		modelName(path), modelName(other), len(cut), modelName(other), window, positions, time.Since(start).Round(100*time.Millisecond))
	// The count of replies comes first: an edit that broke the model moves
	// these probabilities every which way, and a few that rise a lot can
	// carry the totals.
	fmt.Printf("  replies %s would end more readily   %d of %d; p = %.3g (sign test, exact)\n",
		modelName(path), more, len(cut), mcnemar(more, less))
	fmt.Printf("  times it would end, per 1000 tokens  %s %.3f, %s %.3f (%.2f×; %.2f× a reply, geometric mean)\n",
		modelName(other), per(sumRef), modelName(path), per(sumM), sumM/sumRef, math.Exp(logRatio/float64(len(cut))))
	return nil
}

// endProbs reads seq with m and returns, at each position, the probability
// m gives the token ends names there.
func endProbs(m *mlxmodel.Qwen35, seq, ends []int32) (out []float32, err error) {
	defer mlx.Recover(&err)
	c := m.NewCache()
	defer c.Free()
	for at := 0; at < len(seq); at += endsChunk {
		hi := min(at+endsChunk, len(seq))
		lg, err := m.ReadAll(seq[at:hi], c)
		if err != nil {
			return nil, err
		}
		s := mlx.NewScope()
		f := mlx.Astype(lg, mlx.Float32)
		lse := mlx.Logsumexp(f, -1, false)
		want := mlx.Reshape(mlx.TakeAlongAxis(f, mlx.FromInt32(ends[at:hi], hi-at, 1), 1), hi-at)
		p := mlx.Exp(mlx.Subtract(want, lse)).Float32s()
		s.Close()
		lg.Free()
		out = append(out, p...)
	}
	return out, nil
}
