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
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxengine"
	"github.com/LocalKinAI/kinfer/internal/mlxmodel"
)

// endsChunk is how many positions' logits are held at once: a gigabyte of
// them in float32 at Qwen3.5's vocabulary. endsNear is how much of a
// settled reply's thinking, before the token that ended it, is read for how
// readily a model would have ended there; a reply that thought less than
// endsThought characters — an easy problem it barely thought about — says
// nothing about that.
const (
	endsChunk   = 1024
	endsNear    = 64
	endsThought = 1000
)

// endsTest reads the replies that ran out of tokens in the other model's full
// tier — thinking that never stopped, or an answer that never did — with both
// models, and asks how readily each would have ended them. At every position
// it takes the probability of the token that ends the part the reply is in:
// </think> while it thinks, the end of the reply after. Their sum over a
// reply is how many times the model would have ended it there, on average,
// had it read that far; the test compares the two sums reply by reply.
// Nothing is generated, so a reply of window tokens costs one read of them.
//
// Ending more readily is no gain if it comes with thinking longer where the
// other model stopped in good time, which is what p50c does. So it also
// reads replies the other model got right, up to where it ended its
// thinking, and asks the same of the last stretch before that end: an edit
// that would end there less readily thinks on past it.
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
		return fmt.Errorf("the ends test reads %s's full-tier replies, and %w; run kinlab eval -tier full first", modelName(other), err)
	}
	text := map[string]string{}
	for _, p := range probs {
		text[p.id] = p.text
	}
	// The same replies every time, whatever order the results are in.
	hash := func(s string) uint64 { h := fnv.New64a(); h.Write([]byte(s)); return h.Sum64() }
	pick := func(keep func(evalResult) bool) []evalResult {
		var out []evalResult
		for _, r := range res {
			if text[r.ID] != "" && keep(r) {
				out = append(out, r)
			}
		}
		sort.Slice(out, func(i, j int) bool { return hash(out[i].ID) < hash(out[j].ID) })
		if traces > 0 && traces < len(out) {
			out = out[:traces]
		}
		return out
	}
	cut := pick(func(r evalResult) bool { return r.Cut })
	if len(cut) == 0 {
		return fmt.Errorf("none of %s's full-tier replies ran out of tokens, so there is nothing to end", modelName(other))
	}
	natural := pick(func(r evalResult) bool {
		i := strings.Index(r.Reply, "</think>")
		return r.OK && !r.Cut && i >= endsThought
	})
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

	// A trace is a reply read with its prompt, the token each position would
	// end with, and the positions summed. Position i predicts token i+1.
	type trace struct {
		key       string
		seq, ends []int32
		from, to  int
	}
	var runaway, settled []trace
	for _, r := range cut {
		prompt, written := rd.Reply([]chat.Message{{Role: "user", Content: text[r.ID] + suffix}}, true, r.Reply)
		written = written[:min(len(written), window)]
		seq := append(prompt, written...)
		// </think> until the reply has closed its thinking, the end of the
		// reply from then on.
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
		runaway = append(runaway, trace{"run:" + r.ID, seq, ends, len(prompt) - 1, len(seq) - 1})
	}
	for _, r := range natural {
		prompt, written := rd.Reply([]chat.Message{{Role: "user", Content: text[r.ID] + suffix}}, true, r.Reply)
		k := slices.Index(written, endThink)
		if k < endsNear || k >= window {
			continue
		}
		seq := append(prompt, written[:k+1]...)
		ends := make([]int32, len(seq))
		for i := range ends {
			ends[i] = endThink
		}
		// The last stretch of the thinking before the position that ended
		// it, which both models end at almost surely.
		last := len(prompt) + k - 1
		settled = append(settled, trace{"nat2:" + r.ID, seq, ends, last - endsNear, last})
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
	for _, t := range append(append([]trace(nil), runaway...), settled...) {
		if _, ok := cache[t.key]; !ok {
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

	type tally struct {
		sumRef, sumM, logRatio   float64
		positions, more, less, n int
	}
	measure := func(ts []trace) (tally, error) {
		var t tally
		for _, tr := range ts {
			sum := func(model *mlxmodel.Qwen35) (float64, error) {
				p, err := endProbs(model, tr.seq[:tr.to+1], tr.ends)
				if err != nil {
					return 0, err
				}
				s := 0.0
				for i := tr.from; i < tr.to; i++ {
					s += float64(p[i])
				}
				return s, nil
			}
			sa, ok := cache[tr.key]
			if !ok {
				var err error
				if sa, err = sum(ref); err != nil {
					return t, err
				}
				cache[tr.key] = sa
			}
			sb, err := sum(m)
			if err != nil {
				return t, err
			}
			t.n++
			t.positions += tr.to - tr.from
			t.sumRef += sa
			t.sumM += sb
			t.logRatio += math.Log(sb / sa)
			switch {
			case sb > sa:
				t.more++
			case sb < sa:
				t.less++
			}
		}
		return t, nil
	}
	run, err := measure(runaway)
	if err != nil {
		return err
	}
	nat, err := measure(settled)
	if err != nil {
		return err
	}
	if b, err := json.Marshal(cache); err == nil {
		os.WriteFile(cacheFile, b, 0o644)
	}

	a, bn := modelName(other), modelName(path)
	fmt.Printf("%s against %s, %v\n", bn, a, time.Since(start).Round(100*time.Millisecond))
	// The count of replies comes first: an edit that broke the model moves
	// these probabilities every which way, and a few that rise a lot can
	// carry the totals.
	show := func(what string, t tally) {
		fmt.Printf("\n  %s\n", what)
		fmt.Printf("    replies %s would end more readily   %d of %d; p = %.3g (sign test, exact)\n", bn, t.more, t.n, mcnemar(t.more, t.less))
		fmt.Printf("    times it would end, per 1000 tokens  %s %.3f, %s %.3f (%.2f×; %.2f× a reply, geometric mean)\n",
			a, 1000*t.sumRef/float64(t.positions), bn, 1000*t.sumM/float64(t.positions), t.sumM/t.sumRef,
			math.Exp(t.logRatio/float64(max(1, t.n))))
	}
	show(fmt.Sprintf("where %s ran out of tokens (%d replies, their first %d tokens): want more", a, run.n, window), run)
	show(fmt.Sprintf("where %s ended its thinking and was right (%d replies, the %d tokens before it ended): want no fewer", a, nat.n, endsNear), nat)
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
