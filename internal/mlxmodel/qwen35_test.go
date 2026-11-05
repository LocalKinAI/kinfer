package mlxmodel

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// reference is what ref_qwen35.py writes from mlx-lm for the same model and
// prompt.
type reference struct {
	IDs    []int32 `json:"ids"`
	Layers []struct {
		Layer  int       `json:"layer"`
		Linear bool      `json:"linear"`
		Head   []float32 `json:"head"`
		Norm   float64   `json:"norm"`
	} `json:"layers"`
	Top       []int32   `json:"logits_top"`
	TopValues []float32 `json:"logits_top_values"`
	Greedy    []int32   `json:"greedy"`
}

// Needs KINFER_MLX_LIB (the mlx-c libraries), KINFER_MLX_MODEL (a Qwen3.5
// release directory) and KINFER_MLX_REF (ref_qwen35.py's output for it).
func TestQwen35MatchesMLXLM(t *testing.T) {
	lib, dir, refPath := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_MLX_MODEL"), os.Getenv("KINFER_MLX_REF")
	if lib == "" || dir == "" || refPath == "" {
		t.Skip("KINFER_MLX_LIB, KINFER_MLX_MODEL and KINFER_MLX_REF not all set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	var ref reference
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	m, err := LoadQwen35(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()

	// Prefill, comparing each layer's last position with mlx-lm's.
	cache := m.NewCache()
	defer cache.Free()
	L := len(ref.IDs)
	logits, err := m.Forward([][]int32{ref.IDs}, cache, func(i int, h mlx.Array) {
		last := mlx.Astype(mlx.Reshape(mlx.Slice(h, []int{0, L - 1, 0}, []int{1, L, m.Config.HiddenSize}), m.Config.HiddenSize), mlx.Float32)
		v := last.Float32s()
		var norm, worst float64
		for _, x := range v {
			norm += float64(x) * float64(x)
		}
		norm = math.Sqrt(norm)
		for j, want := range ref.Layers[i].Head {
			worst = math.Max(worst, math.Abs(float64(v[j]-want)))
		}
		kind := "attn "
		if ref.Layers[i].Linear {
			kind = "delta"
		}
		t.Logf("layer %2d %s  norm %8.4f (mlx-lm %8.4f, %+.2f%%)  first 8 off by at most %.4f",
			i, kind, norm, ref.Layers[i].Norm, 100*(norm-ref.Layers[i].Norm)/ref.Layers[i].Norm, worst)
	})
	if err != nil {
		t.Fatal(err)
	}
	v := mlx.Astype(logits, mlx.Float32).Float32s()
	logits.Free()
	for j, id := range ref.Top[:5] {
		t.Logf("mlx-lm top %d: token %d %.3f — kinfer %.3f", j+1, id, ref.TopValues[j], v[id])
	}
	best := 0
	for j := range v {
		if v[j] > v[best] {
			best = j
		}
	}
	if int32(best) != ref.Top[0] {
		t.Errorf("first token %d, mlx-lm %d", best, ref.Top[0])
	}

	// Greedy decode with the cache, one token at a time.
	got := []int32{int32(best)}
	for len(got) < len(ref.Greedy) {
		lg, err := m.Forward([][]int32{{got[len(got)-1]}}, cache, nil)
		if err != nil {
			t.Fatal(err)
		}
		next := mlx.Argmax(lg, -1, false)
		got = append(got, mlx.Astype(next, mlx.Int32).Int32s()[0])
		next.Free()
		lg.Free()
	}
	t.Logf("greedy: kinfer %v, mlx-lm %v", got, ref.Greedy)
	for j := range ref.Greedy {
		if got[j] != ref.Greedy[j] {
			t.Fatalf("greedy token %d is %d, mlx-lm %d", j, got[j], ref.Greedy[j])
		}
	}
}

// The Metal kernel and the ops loop agree on the DeltaNet recurrence, and the
// kernel makes prefill fast. Also reports decode speed for one sequence.
func TestQwen35KernelAndSpeed(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_MLX_MODEL")
	if lib == "" || dir == "" {
		t.Skip("KINFER_MLX_LIB and KINFER_MLX_MODEL not both set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadQwen35(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()

	prompt := make([]int32, 256)
	for i := range prompt {
		prompt[i] = int32(1000 + (i*7919)%50000)
	}
	run := func(ops bool) ([]float32, time.Duration) {
		m.OpsDeltaNet = ops
		c := m.NewCache()
		defer c.Free()
		start := time.Now()
		lg, err := m.Forward([][]int32{prompt}, c, nil)
		if err != nil {
			t.Fatal(err)
		}
		d := time.Since(start)
		v := mlx.Astype(lg, mlx.Float32).Float32s()
		lg.Free()
		return v, d
	}
	run(false) // compile the kernel
	fast, dk := run(false)
	slow, do := run(true)
	m.OpsDeltaNet = false
	var worst, scale float64
	for i := range fast {
		worst = math.Max(worst, math.Abs(float64(fast[i]-slow[i])))
		scale = math.Max(scale, math.Abs(float64(slow[i])))
	}
	t.Logf("256-token prefill: kernel %v (%.0f tok/s), ops loop %v; logits differ by at most %.3f of %.1f",
		dk.Round(time.Millisecond), 256/dk.Seconds(), do.Round(time.Millisecond), worst, scale)
	if worst > 0.05*scale {
		t.Errorf("kernel and ops loop disagree: %.3f", worst)
	}

	for _, pipelined := range []bool{false, true} {
		c := m.NewCache()
		lg, err := m.Forward([][]int32{prompt}, c, nil)
		if err != nil {
			t.Fatal(err)
		}
		y := mlx.Reshape(mlx.Astype(mlx.Argmax(lg, -1, false), mlx.Int32), 1, 1)
		lg.Free()
		const n = 200
		var toks []int32
		start := time.Now()
		for i := 0; i < n; i++ {
			s := mlx.NewScope()
			next := mlx.Reshape(mlx.Astype(mlx.Argmax(m.Step(y, c, nil), -1, false), mlx.Int32), 1, 1)
			s.Keep(next)
			s.Close()
			if pipelined {
				mlx.AsyncEval(append([]mlx.Array{next}, c.Arrays()...)...)
			} else {
				mlx.Eval(append([]mlx.Array{next}, c.Arrays()...)...)
			}
			toks = append(toks, y.Int32s()[0])
			y.Free()
			y = next
		}
		el := time.Since(start)
		y.Free()
		c.Free()
		t.Logf("decode, one sequence, pipelined %v: %d tokens in %v = %.1f tok/s (first %v)", pipelined, n,
			el.Round(time.Millisecond), n/el.Seconds(), toks[:6])
	}
}

// Sequences decoding together in a batch give each the tokens it gets alone,
// and the batch is faster in aggregate.
func TestQwen35Batch(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_MLX_MODEL")
	if lib == "" || dir == "" {
		t.Skip("KINFER_MLX_LIB and KINFER_MLX_MODEL not both set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadQwen35(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()

	// Prompts of different lengths, so rows join with different padding.
	prompt := func(seed, n int) []int32 {
		p := make([]int32, n)
		for i := range p {
			p[i] = int32(1000 + ((seed+1)*7919+i*104729)%50000)
		}
		return p
	}
	greedy := func(lg mlx.Array) mlx.Array {
		return mlx.Reshape(mlx.Astype(mlx.Argmax(lg, -1, false), mlx.Int32), lg.Shape()[0], 1)
	}
	const steps = 64
	// alone decodes p by itself and also returns, per step, how far the best
	// logit is ahead of the runner-up — a small gap is a near tie that any
	// change in summation order can flip.
	gapOf := func(lg mlx.Array) float32 {
		top := mlx.Astype(mlx.Topk(mlx.Astype(lg, mlx.Float32), 2, -1), mlx.Float32).Float32s()
		return float32(math.Abs(float64(top[1] - top[0])))
	}
	alone := func(p []int32) ([]int32, []float32) {
		c := m.NewCache()
		defer c.Free()
		lg, err := m.Forward([][]int32{p}, c, nil)
		if err != nil {
			t.Fatal(err)
		}
		gaps := []float32{gapOf(lg)}
		y := greedy(lg)
		lg.Free()
		var out []int32
		for i := 0; i < steps; i++ {
			out = append(out, y.Int32s()[0])
			s := mlx.NewScope()
			lg := m.Step(y, c, nil)
			next := greedy(lg)
			s.Keep(next, lg)
			s.Close()
			mlx.Eval(append([]mlx.Array{next, lg}, c.Arrays()...)...)
			gaps = append(gaps, gapOf(lg))
			lg.Free()
			y.Free()
			y = next
		}
		y.Free()
		return out, gaps
	}
	// matches checks that got, a row's tokens in a batch, is what p decodes to
	// alone, up to a near tie.
	matches := func(what string, got, p []int32) {
		want, gaps := alone(p)
		same := 0
		for same < len(got) && got[same] == want[same] {
			same++
		}
		switch {
		case same == len(got):
			t.Logf("%s: all %d tokens match decoding alone", what, len(got))
		case gaps[same] < 0.25:
			t.Logf("%s: first %d tokens match; at token %d the top two were %.3f apart (a near tie)",
				what, same, same, gaps[same])
		default:
			t.Errorf("%s: token %d differs (%d vs %d) though alone it led by %.3f",
				what, same, got[same], want[same], gaps[same])
		}
	}
	// join reads p alone and adds it to b, returning its first token and row.
	join := func(b *Qwen35Batch, p []int32, room int) (int32, int) {
		c := m.NewCache()
		lg, err := m.Forward([][]int32{p}, c, nil)
		if err != nil {
			t.Fatal(err)
		}
		tok := mlx.Astype(mlx.Argmax(lg, -1, false), mlx.Int32).Int32s()[0]
		lg.Free()
		row, err := b.Add(c, room)
		if err != nil {
			t.Fatal(err)
		}
		return tok, row
	}
	// run decodes n steps from the tokens first, one per row, and returns
	// each row's tokens and the ones the last step chose.
	run := func(b *Qwen35Batch, first []int32, n int) ([][]int32, []int32) {
		y := mlx.FromInt32(first, len(first), 1)
		rows := make([][]int32, len(first))
		for i := 0; i < n; i++ {
			s := mlx.NewScope()
			next := greedy(b.Step(y))
			s.Keep(next)
			s.Close()
			mlx.AsyncEval(append([]mlx.Array{next}, b.Arrays()...)...)
			for r, tok := range y.Int32s() {
				rows[r] = append(rows[r], tok)
			}
			y.Free()
			y = next
		}
		last := y.Int32s()
		y.Free()
		return rows, last
	}

	for _, B := range []int{4, 16, 64} {
		b := m.NewBatch()
		var first []int32
		for r := 0; r < B; r++ {
			c := m.NewCache()
			lg, err := m.Forward([][]int32{prompt(r, 40+(r*37)%200)}, c, nil)
			if err != nil {
				t.Fatal(err)
			}
			first = append(first, mlx.Astype(mlx.Argmax(lg, -1, false), mlx.Int32).Int32s()[0])
			lg.Free()
			if _, err := b.Add(c, B); err != nil {
				t.Fatal(err)
			}
		}
		y := mlx.FromInt32(first, B, 1)
		rows := make([][]int32, B)
		start := time.Now()
		for i := 0; i < steps; i++ {
			s := mlx.NewScope()
			next := greedy(b.Step(y))
			s.Keep(next)
			s.Close()
			mlx.AsyncEval(append([]mlx.Array{next}, b.Arrays()...)...)
			for r, tok := range y.Int32s() {
				rows[r] = append(rows[r], tok)
			}
			y.Free()
			y = next
		}
		el := time.Since(start)
		y.Free()
		b.Free()
		t.Logf("batch of %d: %d steps in %v = %.0f tok/s together", B, steps, el.Round(time.Millisecond),
			float64(B*steps)/el.Seconds())

		// The rows decode what they decode alone (checked on a few).
		for _, r := range []int{0, B / 2, B - 1} {
			matches(fmt.Sprintf("batch of %d, row %d", B, r), rows[r], prompt(r, 40+(r*37)%200))
		}
	}

	// A long sequence that has ended leaves its columns behind; the next one
	// to join cuts them off, and both rows go on as they would alone.
	b := m.NewBatch()
	defer b.Free()
	long, short, next := prompt(100, 1500), prompt(101, 60), prompt(102, 100)
	var first []int32
	for _, p := range [][]int32{long, short} {
		tok, _ := join(b, p, 2)
		first = append(first, tok)
	}
	const before = 8
	rows, last := run(b, first, before)
	b.Release(0)
	wide := b.idx
	tok, row := join(b, next, 2)
	if row != 0 {
		t.Fatalf("the newcomer went to row %d, not the free row 0", row)
	}
	if b.idx != len(next) {
		t.Errorf("after the long row ended and another joined, the batch is %d columns wide (was %d); want %d",
			b.idx, wide, len(next))
	}
	later, _ := run(b, []int32{tok, last[1]}, steps-before)
	matches("the short row", append(rows[1], later[1]...), short)
	matches("the row that joined", later[0], next)
}

// How a decode step's time grows with the batch — the model alone, greedy,
// no server and no sampling. KINFER_MLX_BATCHES lists the sizes ("1,4,16").
func TestQwen35BatchStepTime(t *testing.T) {
	lib, dir, sizes := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_MLX_MODEL"), os.Getenv("KINFER_MLX_BATCHES")
	if lib == "" || dir == "" || sizes == "" {
		t.Skip("KINFER_MLX_LIB, KINFER_MLX_MODEL and KINFER_MLX_BATCHES not all set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadQwen35(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	for _, f := range strings.Split(sizes, ",") {
		B, _ := strconv.Atoi(f)
		b := m.NewBatch()
		first := make([]int32, B)
		for r := 0; r < B; r++ {
			p := make([]int32, 100)
			for i := range p {
				p[i] = int32(1000 + (r*7919+i*104729)%50000)
			}
			c := m.NewCache()
			lg, err := m.Forward([][]int32{p}, c, nil)
			if err != nil {
				t.Fatal(err)
			}
			first[r] = mlx.Astype(mlx.Argmax(lg, -1, false), mlx.Int32).Int32s()[0]
			lg.Free()
			if _, err := b.Add(c, B); err != nil {
				t.Fatal(err)
			}
		}
		y := mlx.FromInt32(first, B, 1)
		const steps = 32
		var start time.Time
		for i := 0; i < steps+4; i++ {
			if i == 4 {
				start = time.Now()
			}
			s := mlx.NewScope()
			next := mlx.Reshape(mlx.Astype(mlx.Argmax(b.Step(y), -1, false), mlx.Int32), B, 1)
			s.Keep(next)
			s.Close()
			mlx.AsyncEval(append([]mlx.Array{next}, b.Arrays()...)...)
			y.Int32s()
			y.Free()
			y = next
		}
		y.Int32s()
		el := time.Since(start)
		y.Free()
		b.Free()
		t.Logf("batch %2d: %.1f ms a step, %.0f tok/s together", B, float64(el.Microseconds())/1000/steps,
			float64(B*steps)/el.Seconds())
	}
}

// A clone and the cache it came from read on separately: what one writes the
// other never sees, so both give exactly what a cache of their own would.
func TestQwen35CacheClone(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_MLX_MODEL")
	if lib == "" || dir == "" {
		t.Skip("KINFER_MLX_LIB and KINFER_MLX_MODEL not both set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadQwen35(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	part := func(seed, n int) []int32 {
		p := make([]int32, n)
		for i := range p {
			p[i] = int32(1000 + ((seed+1)*7919+i*104729)%50000)
		}
		return p
	}
	a, b, c, d := part(1, 300), part(2, 50), part(3, 50), part(4, 40)
	// read feeds chunks into cache and returns the last chunk's logits.
	read := func(cache *Qwen35Cache, chunks ...[]int32) []float32 {
		var out []float32
		for _, ch := range chunks {
			lg, err := m.Forward([][]int32{ch}, cache, nil)
			if err != nil {
				t.Fatal(err)
			}
			out = mlx.Astype(lg, mlx.Float32).Float32s()
			lg.Free()
		}
		return out
	}
	same := func(what string, got, want []float32) {
		worst := 0.0
		for i := range want {
			worst = math.Max(worst, math.Abs(float64(got[i]-want[i])))
		}
		if worst != 0 {
			t.Errorf("%s: logits differ by up to %g", what, worst)
		} else {
			t.Logf("%s: logits identical", what)
		}
	}

	orig := m.NewCache()
	defer orig.Free()
	read(orig, a)
	clone := orig.Clone()
	defer clone.Free()
	read(orig, b)
	gotC := read(clone, c) // writes the same positions orig just wrote
	gotD := read(orig, d)  // reads them back

	fresh := m.NewCache()
	wantD := read(fresh, a, b, d)
	fresh.Free()
	fresh = m.NewCache()
	wantC := read(fresh, a, c)
	fresh.Free()
	same("the cache, after its clone wrote over the same positions", gotD, wantD)
	same("the clone", gotC, wantC)
}
