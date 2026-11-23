package mlxmodel

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// TestInkling2Greedy reads KINFER_INK_TOKENS (token ids, comma separated)
// with the model in KINFER_INK_MODEL and writes on greedily for KINFER_INK_N
// tokens: the ids it writes, the first step's five likeliest, and the time a
// step takes. What it writes is checked against another runtime by hand.
func TestInkling2Greedy(t *testing.T) {
	lib, dir, toks := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_INK_MODEL"), os.Getenv("KINFER_INK_TOKENS")
	if lib == "" || dir == "" || toks == "" {
		t.Skip("KINFER_MLX_LIB, KINFER_INK_MODEL and KINFER_INK_TOKENS not all set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	m, err := LoadInkling2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	t.Logf("loaded in %v: %.1f GB, %d layers, %d visits", time.Since(start).Round(time.Millisecond), float64(m.Bytes())/1e9, len(m.layers), len(m.plan))
	var prompt []int32
	for _, f := range strings.Split(toks, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil {
			t.Fatal(err)
		}
		prompt = append(prompt, int32(n))
	}
	n, _ := strconv.Atoi(os.Getenv("KINFER_INK_N"))
	if n <= 0 {
		n = 24
	}
	// Reading the prompt a chunk at a time must give what one pass gives.
	chunk, _ := strconv.Atoi(os.Getenv("KINFER_INK_CHUNK"))
	if chunk <= 0 {
		chunk = len(prompt)
	}
	c := m.NewCache()
	defer c.Free()
	start = time.Now()
	var lg mlx.Array
	for at := 0; at < len(prompt); at += chunk {
		if lg != 0 {
			lg.Free()
		}
		if lg, err = m.Forward([][]int32{prompt[at:min(at+chunk, len(prompt))]}, c); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%d prompt tokens read in %v", len(prompt), time.Since(start).Round(time.Millisecond))
	top := mlx.Topk(lg, 5, -1)
	t.Logf("five highest scores after the prompt: %v", mlx.Astype(top, mlx.Float32).Float32s())
	y := mlx.Reshape(mlx.Astype(mlx.Argmax(lg, -1, false), mlx.Int32), 1, 1)
	lg.Free()
	var out []string
	var recording time.Duration
	made := 0
	start = time.Now()
	for i := 0; i < n; i++ {
		out = append(out, strconv.Itoa(int(y.Int32s()[0])))
		r := time.Now()
		s := mlx.NewScope()
		next := mlx.Reshape(mlx.Astype(mlx.Argmax(m.Step(y, c), -1, false), mlx.Int32), 1, 1)
		s.Escape(next)
		made += s.Len()
		s.Close()
		recording += time.Since(r)
		if os.Getenv("KINFER_INK_PLAIN_EVAL") != "" {
			mlx.Eval(append([]mlx.Array{next}, c.Arrays()...)...)
		} else {
			mlx.EvalStep(append([]mlx.Array{next}, c.Arrays()...)...)
		}
		y.Free()
		y = next
	}
	el := time.Since(start)
	y.Free()
	t.Logf("wrote: %s", strings.Join(out, ","))
	t.Logf("%d steps in %v: %.1f ms a step, %.1f tok/s; recording a step %.1f ms, %d arrays", n, el.Round(time.Millisecond),
		float64(el.Milliseconds())/float64(n), float64(n)/el.Seconds(), float64(recording.Microseconds())/1000/float64(n), made/n)
}

// TestInkling2Parts times the parts of a step on their own, one token.
func TestInkling2Parts(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_INK_MODEL")
	if lib == "" || dir == "" || os.Getenv("KINFER_INK_PARTS") == "" {
		t.Skip("KINFER_MLX_LIB, KINFER_INK_MODEL and KINFER_INK_PARTS not all set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadInkling2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	x := mlx.Astype(mlx.FromFloat32(make([]float32, m.hidden), 1, 1, m.hidden), mlx.Bfloat16)
	x = mlx.AddScalar(x, 0.01)
	mlx.Eval(x)
	timeIt := func(what string, f func() mlx.Array) {
		for i := 0; i < 3; i++ {
			s := mlx.NewScope()
			a := f()
			mlx.Eval(a)
			s.Close()
		}
		const n = 20
		start := time.Now()
		for i := 0; i < n; i++ {
			s := mlx.NewScope()
			a := f()
			mlx.Eval(a)
			s.Close()
		}
		t.Logf("%-46s %7.2f ms", what, float64(time.Since(start).Microseconds())/1000/n)
	}
	for _, li := range []int{0, 5, 20} {
		l := &m.layers[li]
		timeIt("feed-forward, layer "+strconv.Itoa(li)+" ("+l.gateUp.mode+")", func() mlx.Array { return m.feedForward(l, x) })
		timeIt("  its router and choice alone", func() mlx.Array {
			logits := mlx.Matmul(mlx.Astype(mlx.Reshape(x, 1, m.hidden), mlx.Float32), l.router)
			return mlx.Slice(mlx.Argpartition(mlx.Negative(mlx.Sigmoid(logits)), m.topK-1, -1), []int{0, 0}, []int{1, m.topK})
		})
		// The cache's arrays are the scope's, freed with it.
		timeIt("attention, layer "+strconv.Itoa(li), func() mlx.Array {
			return m.attention(&l.attn, x, x, x, m.NewCache(), 0)
		})
	}
	timeIt("logits", func() mlx.Array { return m.logits(mlx.Reshape(x, 1, m.hidden)) })
	timeIt("mixing four blocks", func() mlx.Array { return inkMix([]mlx.Array{x, x, x, x}, m.layers[5].attnQuery) })
	h := mlx.Zeros(mlx.Bfloat16, 1, 3, m.hidden)
	timeIt("a convolution", func() mlx.Array { hh := h; return m.layers[0].attnOut.run(x, 0, &hh) })
}

// TestInkling2Ops counts the operations each part of a step records.
func TestInkling2Ops(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_INK_MODEL")
	if lib == "" || dir == "" || os.Getenv("KINFER_INK_PARTS") == "" {
		t.Skip("KINFER_MLX_LIB, KINFER_INK_MODEL and KINFER_INK_PARTS not all set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadInkling2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	x := mlx.Astype(mlx.AddScalar(mlx.Zeros(mlx.Float32, 1, 1, m.hidden), 0.01), mlx.Bfloat16)
	mlx.Eval(x)
	count := func(what string, f func()) {
		s := mlx.NewScope()
		f()
		t.Logf("%-40s %4d", what, s.Len())
		s.Close()
	}
	for _, li := range []int{0, 5, 8} {
		l := &m.layers[li]
		count("layer "+strconv.Itoa(li)+": feed-forward", func() { m.feedForward(l, x) })
		count("  attention", func() { m.attention(&l.attn, x, x, x, m.NewCache(), 0) })
		count("  a fixed convolution", func() { var h mlx.Array; l.attnOut.run(x, 0, &h) })
		if l.attnQuery != 0 {
			count("  mixing 3 blocks for attention", func() { inkMix([]mlx.Array{x, x, x}, l.attnQuery) })
		}
		count("  mixing 3 blocks, one query", func() { inkPick(inkMix([]mlx.Array{x, x, x}, l.mlpQuery), 0) })
		count("  a norm", func() { inkRMS(x, l.mlpNorm, m.normEps) })
		if l.messages != 0 {
			count("  a gate", func() { l.gates[0].apply(x, x) })
		}
	}
	count("logits", func() { m.logits(mlx.Reshape(x, 1, m.hidden)) })
	c := m.NewCache()
	y := mlx.FromInt32([]int32{100}, 1, 1)
	s := mlx.NewScope()
	c.rows = m.stepRows(c, 1)
	h := m.hiddenStates(y, c)
	t.Logf("%-40s %4d", "a whole step before the logits", s.Len())
	mlx.Eval(h)
	s.Close()
}

// The kernels compute what the operations they stand for compute: the same
// prompt and the same tokens after it, read both ways, give scores that
// differ by rounding only.
func TestInkling2FusedMatchesPlain(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_INK_MODEL")
	if lib == "" || dir == "" {
		t.Skip("KINFER_MLX_LIB and KINFER_INK_MODEL not both set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadInkling2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	defer func() { inkPlain = false }()
	prompt := make([]int32, 40)
	for i := range prompt {
		prompt[i] = int32(1000 + (i*104729)%50000)
	}
	after := []int32{371, 150409, 171960, 13, 220, 23104}
	run := func(plain bool) [][]float32 {
		inkPlain = plain
		c := m.NewCache()
		defer c.Free()
		var out [][]float32
		lg, err := m.Forward([][]int32{prompt}, c)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, lg.Float32s())
		lg.Free()
		for _, tok := range after {
			lg, err := m.Forward([][]int32{{tok}}, c)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, lg.Float32s())
			lg.Free()
		}
		return out
	}
	plain, fused := run(true), run(false)
	for i := range plain {
		worst, top := float32(0), [2]int{}
		for j := range plain[i] {
			d := plain[i][j] - fused[i][j]
			if d < 0 {
				d = -d
			}
			worst = max(worst, d)
			if plain[i][j] > plain[i][top[0]] {
				top[0] = j
			}
			if fused[i][j] > fused[i][top[1]] {
				top[1] = j
			}
		}
		what := "the prompt"
		if i > 0 {
			what = "step " + strconv.Itoa(i)
		}
		t.Logf("%s: scores differ by at most %.4f; likeliest token %d and %d", what, worst, top[0], top[1])
		// A kernel differs from its operations by a last bit of bfloat16
		// here and there (TestInkling2KernelsMatchOps), which 48 layers that
		// each choose 8 experts of 128 make more of: scores of some tokens
		// move by one or so in thirty. What is likeliest must not, unless
		// another was as likely.
		if lead := plain[i][top[0]] - plain[i][top[1]]; worst > 3 || (top[0] != top[1] && lead > 0.5) {
			t.Errorf("%s: the kernels' scores are %.3f from the operations', their likeliest token another one %.3f behind", what, worst, lead)
		}
	}
}

// Each kernel against the operations it stands for, on the model's own
// weights and inputs of the size a layer sees.
func TestInkling2KernelsMatchOps(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_INK_MODEL")
	if lib == "" || dir == "" {
		t.Skip("KINFER_MLX_LIB and KINFER_INK_MODEL not both set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadInkling2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	defer func() { inkPlain = false }()
	rng := rand.New(rand.NewSource(7))
	random := func(scale float32, dt mlx.Dtype, shape ...int) mlx.Array {
		n := 1
		for _, d := range shape {
			n *= d
		}
		f := make([]float32, n)
		for i := range f {
			f[i] = float32(rng.NormFloat64()) * scale
		}
		a := mlx.Astype(mlx.FromFloat32(f, shape...), dt)
		mlx.Eval(a)
		return a
	}
	both := func(what string, tol float32, f func() []mlx.Array) {
		t.Helper()
		var got [2][][]float32
		for i, plain := range []bool{true, false} {
			inkPlain = plain
			s := mlx.NewScope()
			for _, a := range f() {
				// Contiguous: a slice of an array read as it lies is not the slice.
				got[i] = append(got[i], mlx.Contiguous(mlx.Astype(a, mlx.Float32)).Float32s())
			}
			s.Close()
		}
		for o := range got[0] {
			worst, size := float32(0), float32(0)
			for j := range got[0][o] {
				d := got[0][o][j] - got[1][o][j]
				worst, size = max(worst, max(d, -d)), max(size, max(got[0][o][j], -got[0][o][j]))
			}
			t.Logf("%-34s output %d: off by %.5f of %.3f", what, o, worst, size)
			if worst > tol*size {
				t.Errorf("%s, output %d: the kernel is %.5f from the operations (values to %.3f)", what, o, worst, size)
			}
		}
	}
	H := m.hidden
	for _, L := range []int{1, 5} {
		x := random(1, mlx.Bfloat16, 2, L, H)
		hist := random(1, mlx.Bfloat16, 2, 3, H)
		for _, first := range []bool{true, false} {
			both(fmt.Sprintf("fixed taps, %d tokens, first %v", L, first), 0.01, func() []mlx.Array {
				var h mlx.Array
				if !first {
					h = hist.Ref()
				}
				out := m.layers[0].attnOut.run(x, 0, &h)
				return []mlx.Array{out, h}
			})
		}
		a := &m.layers[5].attn
		kx := random(1, mlx.Bfloat16, 2, L, a.kvHeads*a.headDim)
		kh := random(1, mlx.Bfloat16, 2, 3, a.kvHeads*a.headDim)
		shifts := random(0.3, mlx.Bfloat16, 2, L, a.kvHeads*m.taps)
		both(fmt.Sprintf("projected taps, %d tokens", L), 0.01, func() []mlx.Array {
			h := kh.Ref()
			out := a.kConv.run(kx, shifts, &h)
			return []mlx.Array{out, h}
		})
		q := random(3, mlx.Bfloat16, 2, L, a.heads, a.headDim)
		tau := random(0.1, mlx.Float32, L)
		both(fmt.Sprintf("clip with a weight and a temperature, %d tokens", L), 0.01, func() []mlx.Array {
			if inkPlain {
				return []mlx.Array{inkClip(q, a.qScale, mlx.Reshape(tau, 1, L, 1, 1), m.Config.QKClip)}
			}
			return []mlx.Array{inkClipFused(q, a.qScale, tau, m.Config.QKClip)}
		})
		both(fmt.Sprintf("clip alone, %d tokens", L), 0.01, func() []mlx.Array {
			if inkPlain {
				return []mlx.Array{inkClip(q, 0, 0, 1.5)}
			}
			return []mlx.Array{inkClipFused(q, 0, 0, 1.5)}
		})
		for _, li := range []int{0, 5, 20} {
			l := &m.layers[li]
			fx := inkRMS(random(1, mlx.Bfloat16, 2, L, H), l.mlpNorm, m.normEps)
			mlx.Eval(fx)
			both(fmt.Sprintf("experts of layer %d (%s), %d tokens", li, l.gateUp.mode, L), 0.02, func() []mlx.Array {
				return []mlx.Array{m.feedForward(l, fx)}
			})
		}
		in, msg := random(1, mlx.Bfloat16, 2, L, H), random(1, mlx.Float32, 2, L, H)
		both(fmt.Sprintf("a gate, %d tokens", L), 0.01, func() []mlx.Array {
			return []mlx.Array{m.layers[8].gates[0].apply(in, msg)}
		})
	}

	// The residual mixes and the gates, whose kernels sum over a whole state.
	for _, L := range []int{1, 3} {
		var srcs []mlx.Array
		for i := 0; i < 5; i++ {
			dt := mlx.Bfloat16
			if i == 4 {
				dt = mlx.Float32 // the block being written is float32 inside a block
			}
			srcs = append(srcs, random(float32(i+1), dt, 2, L, H))
		}
		for _, c := range []struct {
			what          string
			n             int
			queries, norm mlx.Array
		}{
			{"one source, one query, normed", 1, m.layers[1].attnQuery, m.layers[1].attnNorms},
			{"three sources, three queries, normed", 3, m.layers[5].attnQuery, m.layers[5].attnNorms},
			{"five sources, three queries, normed", 5, m.layers[23].attnQuery, m.layers[23].attnNorms},
			{"four sources, one query, normed", 4, m.layers[20].mlpQuery, m.layers[20].mlpNorm},
			{"two sources, four queries, as they are", 2, m.layers[8].messages, 0},
			{"three sources, one query, as they are", 3, m.layers[8].mlpQuery, 0},
			{"one source, four queries, as they are", 1, m.layers[8].messages, 0},
			{"two sources, three queries, as they are", 2, m.layers[5].attnQuery, 0},
			{"four sources, four queries, as they are", 4, m.layers[8].messages, 0},
		} {
			both(fmt.Sprintf("mix of %s, %d tokens", c.what, L), 0.01, func() []mlx.Array {
				if !inkPlain {
					return inkMixFused(srcs[5-c.n:], c.queries, c.norm, m.normEps)
				}
				mixed := inkMix(srcs[5-c.n:], c.queries)
				var out []mlx.Array
				for q := 0; q < c.queries.Shape()[1]; q++ {
					x := inkPick(mixed, q)
					if c.norm != 0 {
						nw := c.norm
						if len(nw.Shape()) == 2 {
							nw = mlx.Reshape(mlx.Slice(nw, []int{q, 0}, []int{q + 1, H}), H)
						}
						x = inkRMS(x, nw, m.normEps)
					}
					out = append(out, x)
				}
				return out
			})
		}
		in, msg := random(2, mlx.Float32, 2, L, H), random(1, mlx.Float32, 2, L, H)
		g := m.layers[8].gates[3]
		both(fmt.Sprintf("a gate with the norm after it, %d tokens", L), 0.01, func() []mlx.Array {
			if !inkPlain {
				return []mlx.Array{g.applyNormed(in, msg, m.layers[8].mlpNorm, m.normEps)}
			}
			return []mlx.Array{inkRMS(g.apply(in, msg), m.layers[8].mlpNorm, m.normEps)}
		})
	}

	// One position's attention after a history: short, and longer than the
	// window, the table of biases and a chunk of keys.
	for _, li := range []int{0, 5} {
		for _, behind := range []int{37, 1300} {
			inkPlain = true
			c := m.NewCache()
			for at := 0; at < behind; at += 256 {
				n := min(256, behind-at)
				x := inkRMS(random(1, mlx.Bfloat16, 1, n, H), m.layers[li].attnNorm[0], m.normEps)
				s := mlx.NewScope()
				out := m.attention(&m.layers[li].attn, x, x, x, c, 0)
				s.Escape(c.Arrays()...)
				mlx.Eval(append([]mlx.Array{out}, c.Arrays()...)...)
				s.Close()
				c.Offset += n
			}
			x := inkRMS(random(1, mlx.Bfloat16, 1, 1, H), m.layers[li].attnNorm[0], m.normEps)
			mlx.Eval(x)
			both(fmt.Sprintf("attention of layer %d, %d tokens behind", li, behind), 0.02, func() []mlx.Array {
				cc := c.Clone()
				if !inkPlain {
					cc.rows = m.stepRows(cc, 1)
				}
				return []mlx.Array{m.attention(&m.layers[li].attn, x, x, x, cc, 0)}
			})
			c.Free()
		}
	}
}

// TestInkling2Heavy times the products with weights a step is made of, each
// kind for every layer twice in one evaluation: what the step would cost if
// it were nothing else.
func TestInkling2Heavy(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_INK_MODEL")
	if lib == "" || dir == "" || os.Getenv("KINFER_INK_PARTS") == "" {
		t.Skip("KINFER_MLX_LIB, KINFER_INK_MODEL and KINFER_INK_PARTS not all set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadInkling2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	H := m.hidden
	x := mlx.Astype(mlx.AddScalar(mlx.Zeros(mlx.Float32, 1, 1, H), 0.01), mlx.Bfloat16)
	act := mlx.Astype(mlx.AddScalar(mlx.Zeros(mlx.Float32, 1, m.topK, 1, m.width), 0.01), mlx.Bfloat16)
	idx := mlx.Astype(mlx.Reshape(mlx.Arange(3, float64(3+m.topK), 1, mlx.Int32), 1, m.topK), mlx.Uint32)
	mlx.Eval(x, act, idx)
	timeIt := func(what string, f func(l *inkLayer) mlx.Array) {
		run := func() {
			s := mlx.NewScope()
			var outs []mlx.Array
			for rep := 0; rep < 2; rep++ {
				for i := range m.layers {
					outs = append(outs, f(&m.layers[i]))
				}
			}
			mlx.Eval(outs...)
			s.Close()
		}
		run()
		run()
		const n = 10
		start := time.Now()
		for i := 0; i < n; i++ {
			run()
		}
		t.Logf("%-52s %6.2f ms for 48", what, float64(time.Since(start).Microseconds())/1000/n)
	}
	timeIt("the attention's projections of one input", func(l *inkLayer) mlx.Array { return l.attn.project(x, 0) })
	timeIt("the attention's output projection", func(l *inkLayer) mlx.Array {
		return l.attn.project(mlx.Zeros(mlx.Bfloat16, 1, 1, l.attn.heads*l.attn.headDim), 4)
	})
	timeIt("the router", func(l *inkLayer) mlx.Array {
		return mlx.Matmul(mlx.Astype(mlx.Reshape(x, 1, H), mlx.Float32), l.router)
	})
	timeIt("eight experts' first projection", func(l *inkLayer) mlx.Array { return l.gateUp.raw(mlx.Reshape(x, 1, 1, 1, H), idx) })
	timeIt("eight experts' second projection", func(l *inkLayer) mlx.Array { return l.down.raw(act, idx) })
	timeIt("a whole feed-forward", func(l *inkLayer) mlx.Array { return m.feedForward(l, x) })
	timeIt("a fixed convolution", func(l *inkLayer) mlx.Array { var h mlx.Array; return l.attnOut.run(x, 0, &h) })
	timeIt("a mix of four blocks and a norm", func(l *inkLayer) mlx.Array {
		return inkRMS(inkPick(inkMix([]mlx.Array{x, x, x, x}, l.mlpQuery), 0), l.mlpNorm, m.normEps)
	})
	start := time.Now()
	for i := 0; i < 10; i++ {
		s := mlx.NewScope()
		mlx.Eval(m.logits(mlx.Reshape(x, 1, H)))
		s.Close()
	}
	t.Logf("%-52s %6.2f ms", "the logits", float64(time.Since(start).Microseconds())/1000/10)
}

// TestInkling2PrefillParts times the parts of reading 512 prompt tokens in
// one pass, each for every layer twice in one evaluation.
func TestInkling2PrefillParts(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_INK_MODEL")
	if lib == "" || dir == "" || os.Getenv("KINFER_INK_PARTS") == "" {
		t.Skip("KINFER_MLX_LIB, KINFER_INK_MODEL and KINFER_INK_PARTS not all set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadInkling2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	const L = 512
	H := m.hidden
	rng := rand.New(rand.NewSource(3))
	f := make([]float32, L*H)
	for i := range f {
		f[i] = float32(rng.NormFloat64())
	}
	x := mlx.Astype(mlx.FromFloat32(f, 1, L, H), mlx.Bfloat16)
	mlx.Eval(x)
	timeIt := func(what string, f func(l *inkLayer) mlx.Array) {
		run := func() {
			s := mlx.NewScope()
			var outs []mlx.Array
			for rep := 0; rep < 2; rep++ {
				for i := range m.layers {
					outs = append(outs, f(&m.layers[i]))
				}
			}
			mlx.Eval(outs...)
			s.Close()
		}
		run()
		const n = 3
		start := time.Now()
		for i := 0; i < n; i++ {
			run()
		}
		ms := float64(time.Since(start).Microseconds()) / 1000 / n
		t.Logf("%-44s %7.1f ms for 48, %5.0f tokens a second if it were all", what, ms, L/ms*1000)
	}
	timeIt("the experts", func(l *inkLayer) mlx.Array { return m.feedForward(l, x) })
	timeIt("the attention's projections", func(l *inkLayer) mlx.Array { return l.attn.project(x, 0) })
	timeIt("an attention, 512 tokens behind it", func(l *inkLayer) mlx.Array {
		c := m.NewCache()
		c.keys[0] = mlx.Zeros(mlx.Bfloat16, 1, l.attn.kvHeads, 1024, l.attn.headDim)
		c.vals[0] = mlx.Zeros(mlx.Bfloat16, 1, l.attn.kvHeads, 1024, l.attn.headDim)
		c.Offset = 512
		return m.attention(&l.attn, x, x, x, c, 0)
	})
	timeIt("a mix of four blocks and its norm", func(l *inkLayer) mlx.Array {
		return inkMixFused([]mlx.Array{x, x, x, x}, l.mlpQuery, l.mlpNorm, m.normEps)[0]
	})
	timeIt("a convolution", func(l *inkLayer) mlx.Array { var h mlx.Array; return l.attnOut.run(x, 0, &h) })
}

// TestInkling2StepParts times every part of one token's step as the model
// now runs it, each 48 times in one evaluation (the gates 96, as a step has
// them), on a quiet GPU: what they add up to against the step itself.
func TestInkling2StepParts(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_INK_MODEL")
	if lib == "" || dir == "" || os.Getenv("KINFER_INK_PARTS") == "" {
		t.Skip("KINFER_MLX_LIB, KINFER_INK_MODEL and KINFER_INK_PARTS not all set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadInkling2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	H := m.hidden
	rng := rand.New(rand.NewSource(5))
	random := func(dt mlx.Dtype, shape ...int) mlx.Array {
		n := 1
		for _, d := range shape {
			n *= d
		}
		f := make([]float32, n)
		for i := range f {
			f[i] = float32(rng.NormFloat64())
		}
		a := mlx.Astype(mlx.FromFloat32(f, shape...), dt)
		mlx.Eval(a)
		return a
	}
	x, xf := random(mlx.Bfloat16, 1, 1, H), random(mlx.Float32, 1, 1, H)
	hist := random(mlx.Bfloat16, 1, 3, H)
	keys, vals := random(mlx.Bfloat16, 1, 8, 1024, 128), random(mlx.Bfloat16, 1, 8, 1024, 128)
	hk, hv := random(mlx.Bfloat16, 1, 3, 1024), random(mlx.Bfloat16, 1, 3, 1024)
	total := 0.0
	timeIt := func(what string, times int, f func(l *inkLayer) []mlx.Array) {
		run := func() {
			s := mlx.NewScope()
			var outs []mlx.Array
			for rep := 0; rep < times; rep++ {
				for i := range m.layers {
					outs = append(outs, f(&m.layers[i])...)
				}
			}
			mlx.EvalStep(outs...)
			s.Close()
		}
		run()
		run()
		const n = 10
		start := time.Now()
		for i := 0; i < n; i++ {
			run()
		}
		ms := float64(time.Since(start).Microseconds()) / 1000 / n
		total += ms
		t.Logf("%-58s %6.2f ms", what, ms)
	}
	var all, q, r mlx.Array
	timeIt("48 × the attention's projections", 2, func(l *inkLayer) []mlx.Array { return []mlx.Array{l.attn.project(x, 0)} })
	all = m.layers[0].attn.project(x, 0)
	allG := m.layers[5].attn.project(x, 0)
	mlx.Eval(all, allG)
	pick := func(l *inkLayer) mlx.Array {
		if l.local {
			return all
		}
		return allG
	}
	timeIt("48 × what follows them: clips, convolutions, norm", 2, func(l *inkLayer) []mlx.Array {
		a, b := hk, hv
		q, r, k, v := m.prepFused(&l.attn, pick(l), inkOnes(1), &a, &b)
		return []mlx.Array{q, r, k, v, a, b}
	})
	{
		a, b := hk, hv
		q, r, _, _ = m.prepFused(&m.layers[0].attn, all, inkOnes(1), &a, &b)
		mlx.Eval(q, r)
	}
	from0 := mlx.FromInt32([]int32{0}, 1)
	mlx.Eval(from0)
	timeIt("48 × the attention itself, 600 keys in view", 2, func(l *inkLayer) []mlx.Array {
		return []mlx.Array{inkAttend(q, keys, vals, r, l.attn.positions, pick(l), from0, 24*128+24*16, len24(l), 0, 600, 24)}
	})
	att := random(mlx.Bfloat16, 1, 1, 3072)
	timeIt("48 × the attention's output projection", 2, func(l *inkLayer) []mlx.Array { return []mlx.Array{l.attn.project(att, 4)} })
	timeIt("96 × a convolution", 4, func(l *inkLayer) []mlx.Array { h := hist; return []mlx.Array{l.attnOut.run(x, 0, &h), h} })
	timeIt("96 × a mix of blocks with its norm", 4, func(l *inkLayer) []mlx.Array {
		return inkMixFused([]mlx.Array{x, x, x, xf}, l.mlpQuery, l.mlpNorm, m.normEps)
	})
	timeIt("48 × the experts", 2, func(l *inkLayer) []mlx.Array { return []mlx.Array{m.feedForward(l, x)} })
	timeIt("96 × a gate (a step has four in each of 24)", 4, func(l *inkLayer) []mlx.Array {
		g := m.layers[8].gates[0]
		return []mlx.Array{g.applyNormed(xf, xf, l.mlpNorm, m.normEps)}
	})
	timeIt("24 × the messages from the pass before", 1, func(l *inkLayer) []mlx.Array {
		return m.messages([]mlx.Array{x, xf}, m.layers[8].messages)
	})
	timeIt("24 × the logits (a step has them once)", 1, func(l *inkLayer) []mlx.Array { return []mlx.Array{m.logits(mlx.Reshape(x, 1, H))} })
	t.Logf("in all %.1f ms, the logits counted 24 times too many", total)
}

func len24(l *inkLayer) int { return 24*128 + 24*16 + 24 }

// A batch's rows write what each writes alone: rows of different lengths, a
// row leaving and another taking its place, a row the batch grows for, and
// free rows dropped.
func TestInkling2Batch(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_INK_MODEL")
	if lib == "" || dir == "" {
		t.Skip("KINFER_MLX_LIB and KINFER_INK_MODEL not both set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadInkling2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
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
	read := func(p []int32) (*Inkling2Cache, int32, float32) {
		c := m.NewCache()
		lg, err := m.Forward([][]int32{p}, c)
		if err != nil {
			t.Fatal(err)
		}
		top := mlx.Topk(lg, 2, -1).Float32s()
		tok := mlx.Astype(mlx.Argmax(lg, -1, false), mlx.Int32).Int32s()[0]
		lg.Free()
		return c, tok, float32(math.Abs(float64(top[1] - top[0])))
	}
	// alone is what p decodes to by itself, with how far each token led.
	alone := func(p []int32, n int) (toks []int32, gaps []float32) {
		c, tok, gap := read(p)
		defer c.Free()
		y := mlx.FromInt32([]int32{tok}, 1, 1)
		for i := 0; i < n; i++ {
			toks, gaps = append(toks, y.Int32s()[0]), append(gaps, gap)
			s := mlx.NewScope()
			lg := m.Step(y, c)
			next := greedy(lg)
			s.Escape(next, lg)
			s.Close()
			mlx.Eval(append([]mlx.Array{next, lg}, c.Arrays()...)...)
			top := mlx.Topk(lg, 2, -1).Float32s()
			gap = float32(math.Abs(float64(top[1] - top[0])))
			lg.Free()
			y.Free()
			y = next
		}
		y.Free()
		return toks, gaps
	}
	matches := func(what string, got, p []int32) {
		t.Helper()
		want, gaps := alone(p, len(got))
		same := 0
		for same < len(got) && got[same] == want[same] {
			same++
		}
		switch {
		case same == len(got):
			t.Logf("%s: all %d tokens are what it writes alone", what, len(got))
		case gaps[same] < 1:
			t.Logf("%s: the first %d tokens are what it writes alone; the next was a near tie (%.3f apart)", what, same, gaps[same])
		default:
			t.Errorf("%s: token %d is %d, alone %d, which led by %.3f", what, same, got[same], want[same], gaps[same])
		}
	}
	join := func(b *Inkling2Batch, p []int32, room int) (int32, int) {
		c, tok, _ := read(p)
		row, err := b.Add(c, room)
		if err != nil {
			t.Fatal(err)
		}
		return tok, row
	}
	// run decodes n steps from the tokens first, one a row, each step started
	// before the one before it is read; it returns what each row read and
	// wrote, and the tokens the last step chose.
	run := func(b *Inkling2Batch, first []int32, n int) ([][]int32, []int32) {
		y := mlx.FromInt32(first, len(first), 1)
		rows := make([][]int32, len(first))
		for i := 0; i < n; i++ {
			s := mlx.NewScope()
			next := greedy(b.Step(y))
			s.Escape(next)
			s.Close()
			mlx.AsyncEvalStep(append([]mlx.Array{next}, b.Arrays()...)...)
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
	const n = 40
	b := m.NewBatch()
	defer b.Free()
	ps := [][]int32{prompt(1, 70), prompt(2, 31), prompt(3, 120)}
	var first []int32
	for _, p := range ps {
		tok, _ := join(b, p, len(ps))
		first = append(first, tok)
	}
	rows, last := run(b, first, 16)
	// Row 1 ends; a longer one than the batch is wide takes its place.
	b.Release(1)
	p4 := prompt(4, 330)
	tok4, row := join(b, p4, 3)
	if row != 1 {
		t.Fatalf("the newcomer went to row %d, not the free row 1", row)
	}
	more, last := run(b, []int32{last[0], tok4, last[2]}, 16)
	rows[0], rows[2] = append(rows[0], more[0]...), append(rows[2], more[2]...)
	fourth := more[1]
	// A fifth joins, for which the batch grows.
	p5 := prompt(5, 45)
	tok5, row := join(b, p5, 4)
	if row != 3 {
		t.Fatalf("the fifth went to row %d, not a new row 3", row)
	}
	more, last = run(b, append(last, tok5), 8)
	rows[0], rows[2], fourth = append(rows[0], more[0]...), append(rows[2], more[2]...), append(fourth, more[1]...)
	fifth := more[3]
	// Rows 0 and 2 end and are dropped; the two others go on.
	b.Release(0)
	b.Release(2)
	moved, err := b.Compact()
	if err != nil {
		t.Fatal(err)
	}
	if b.Len() != 2 || moved[1] != 0 || moved[3] != 1 {
		t.Fatalf("after two rows were dropped the batch has %d rows, moved %v", b.Len(), moved)
	}
	more, _ = run(b, []int32{last[1], last[3]}, 16)
	fourth, fifth = append(fourth, more[0]...), append(fifth, more[1]...)
	matches("a row there from the start", rows[0][:n], ps[0])
	matches("another, longer", rows[2][:n], ps[2])
	matches("the row that took a free row", fourth[:n], p4)
	matches("the row the batch grew for", fifth[:24], p5)
}

// How a step's time grows with the batch: KINFER_INK_BATCHES lists the sizes.
func TestInkling2BatchStepTime(t *testing.T) {
	lib, dir, sizes := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_INK_MODEL"), os.Getenv("KINFER_INK_BATCHES")
	if lib == "" || dir == "" || sizes == "" {
		t.Skip("KINFER_MLX_LIB, KINFER_INK_MODEL and KINFER_INK_BATCHES not all set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadInkling2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	for _, f := range strings.Split(sizes, ",") {
		B, _ := strconv.Atoi(f)
		b := m.NewBatch()
		var first []int32
		for r := 0; r < B; r++ {
			p := make([]int32, 60+11*r)
			for i := range p {
				p[i] = int32(1000 + ((r+1)*7919+i*104729)%50000)
			}
			c := m.NewCache()
			lg, err := m.Forward([][]int32{p}, c)
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
		const steps, warm = 60, 8
		var start time.Time
		var recording, starting, waiting time.Duration
		for i := 0; i < steps+warm; i++ {
			if i == warm {
				y.Int32s()
				start = time.Now()
			}
			t0 := time.Now()
			s := mlx.NewScope()
			next := mlx.Reshape(mlx.Astype(mlx.Argmax(b.Step(y), -1, false), mlx.Int32), B, 1)
			s.Escape(next)
			s.Close()
			t1 := time.Now()
			mlx.AsyncEvalStep(append([]mlx.Array{next}, b.Arrays()...)...)
			t2 := time.Now()
			y.Int32s()
			t3 := time.Now()
			if i >= warm {
				recording, starting, waiting = recording+t1.Sub(t0), starting+t2.Sub(t1), waiting+t3.Sub(t2)
			}
			y.Free()
			y = next
		}
		y.Int32s()
		el := time.Since(start)
		y.Free()
		b.Free()
		ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / steps }
		t.Logf("%2d rows: %.1f ms a step, %.0f tok/s in all; the host records it in %.1f ms and starts it in %.1f, then waits %.1f for the step before",
			B, float64(el.Microseconds())/1000/steps, float64(B*steps)/el.Seconds(), ms(recording), ms(starting), ms(waiting))
	}
}
