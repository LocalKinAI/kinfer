package mlxmodel

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// draftKit is what the batch drafting tests share: a model with an MTP head
// and prompts to read.
type draftKit struct {
	t *testing.T
	m *Qwen35
}

func newDraftKit(t *testing.T) *draftKit {
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
	if !m.CanDraft() {
		m.Free()
		t.Skip("the model has no MTP head")
	}
	return &draftKit{t, m}
}

func (k *draftKit) prompt(seed, n int) []int32 {
	p := make([]int32, n)
	for i := range p {
		p[i] = int32(1000 + ((seed+1)*7919+i*104729)%50000)
	}
	return p
}

// read reads p into a new cache, the head too, and returns it with the first
// token of the reply.
func (k *draftKit) read(p []int32) (*Qwen35Cache, int32) {
	c := k.m.NewCache()
	c.Drafting = true
	lg, err := k.m.Forward([][]int32{p}, c, nil)
	if err != nil {
		k.t.Fatal(err)
	}
	tok := mlx.Astype(mlx.Argmax(lg, -1, false), mlx.Int32).Int32s()[0]
	lg.Free()
	return c, tok
}

// alone is what p decodes to by itself, a token a step, with how far each
// token led the runner-up: a small lead is a near tie that any change in the
// order of a sum can turn.
func (k *draftKit) alone(p []int32, n int) (toks []int32, gaps []float32) {
	c, tok := k.read(p)
	defer c.Free()
	c.Drafting = false
	y := mlx.FromInt32([]int32{tok}, 1, 1)
	for i := 0; i < n; i++ {
		toks = append(toks, y.Int32s()[0])
		s := mlx.NewScope()
		lg := k.m.Step(y, c, nil)
		next := mlx.Reshape(mlx.Astype(mlx.Argmax(lg, -1, false), mlx.Int32), 1, 1)
		s.Keep(next, lg)
		s.Close()
		mlx.Eval(append([]mlx.Array{next, lg}, c.Arrays()...)...)
		top := mlx.Astype(mlx.Topk(mlx.Astype(lg, mlx.Float32), 2, -1), mlx.Float32).Float32s()
		gaps = append(gaps, float32(math.Abs(float64(top[1]-top[0]))))
		lg.Free()
		y.Free()
		y = next
	}
	y.Free()
	return toks, gaps
}

// matches checks that got, a row's tokens from a batch, is what p decodes to
// alone, up to a near tie.
func (k *draftKit) matches(what string, got, p []int32) {
	k.t.Helper()
	want, gaps := k.alone(p, len(got))
	same := 0
	for same < len(got) && got[same] == want[same] {
		same++
	}
	switch {
	case same == len(got):
		k.t.Logf("%s: all %d tokens are what it writes alone", what, len(got))
	case same > 0 && gaps[same-1] < 0.25:
		k.t.Logf("%s: the first %d tokens are what it writes alone; the next was a near tie (%.3f apart)", what, same, gaps[same-1])
	default:
		lead := float32(0)
		if same > 0 {
			lead = gaps[same-1]
		}
		k.t.Errorf("%s: token %d is %d, alone %d, which led by %.3f", what, same, got[same], want[same], lead)
	}
}

// flight is a pass recorded and started: what its rows take, and of that how
// much each keeps.
type flight struct {
	picks, next, kept mlx.Array
	L                 int
}

// start records a pass of d drafts from tokens, greedy, and starts it.
func (k *draftKit) start(b *Qwen35Batch, tokens mlx.Array, d int) flight {
	s := mlx.NewScope()
	logits := b.Pass(tokens, d)
	picks := mlx.Astype(mlx.Argmax(logits, -1, false), mlx.Int32)
	next, kept := b.Settle(picks)
	s.Escape(picks, next, kept)
	s.Close()
	mlx.AsyncEvalStep(append([]mlx.Array{picks, next, kept}, b.Arrays()...)...)
	return flight{picks, next, kept, d + 1}
}

// take reads a pass and appends to each row what it wrote; it returns how
// many drafts were kept.
func (f flight) take(rows [][]int32) (kept int) {
	picks, n := f.picks.Int32s(), f.kept.Int32s()
	for r := range rows {
		rows[r] = append(rows[r], picks[r*f.L:r*f.L+int(n[r])+1]...)
		kept += int(n[r])
	}
	f.picks.Free()
	f.kept.Free()
	return kept
}

// run decodes passes of the depths given, in turn, from the tokens first, each
// pass recorded and started before the one before it is read, until every row
// has written n tokens more. It returns what each row wrote — its first token
// not among them — and the token each reads next.
func (k *draftKit) run(b *Qwen35Batch, first []int32, depths []int, n int) (rows [][]int32, last []int32) {
	rows = make([][]int32, len(first))
	y := mlx.FromInt32(first, len(first), 1)
	cur := k.start(b, y, depths[0])
	y.Free()
	drafted, kept := depths[0]*len(first), 0
	short := func() bool {
		for _, r := range rows {
			if len(r) < n {
				return true
			}
		}
		return false
	}
	for i := 1; ; i++ {
		d := depths[i%len(depths)]
		next := k.start(b, cur.next, d)
		drafted += d * len(first)
		kept += cur.take(rows)
		cur.next.Free()
		cur = next
		if !short() {
			break
		}
	}
	// The pass in flight is read too: the batch is where it leaves it.
	kept += cur.take(rows)
	last = cur.next.Int32s()
	cur.next.Free()
	k.t.Logf("  %d rows, depths %v: %d of %d drafts kept", len(first), depths, kept, drafted)
	return rows, last
}

// A batch's rows, drafting, write what each writes alone: with passes of
// several depths, rows of different lengths, a row leaving and another taking
// its place, a row joining a batch that has to grow, and free rows dropped.
func TestQwen35BatchDrafts(t *testing.T) {
	k := newDraftKit(t)
	defer k.m.Free()
	const n = 48
	join := func(b *Qwen35Batch, p []int32, room int) (int32, int) {
		c, tok := k.read(p)
		row, err := b.Add(c, room)
		if err != nil {
			t.Fatal(err)
		}
		return tok, row
	}
	with := func(first int32, wrote []int32) []int32 { return append([]int32{first}, wrote...) }

	for _, depths := range [][]int{{1}, {3}, {2, 0, 1, 3}} {
		b := k.m.NewBatch()
		b.Drafting = true
		ps := [][]int32{k.prompt(1, 60), k.prompt(2, 140), k.prompt(3, 97)}
		var first []int32
		for _, p := range ps {
			tok, _ := join(b, p, len(ps))
			first = append(first, tok)
		}
		rows, _ := k.run(b, first, depths, n)
		for r, p := range ps {
			k.matches(fmt.Sprintf("depths %v, row %d", depths, r), with(first[r], rows[r])[:n], p)
		}
		b.Free()
	}

	// Rows leave and join a batch that has columns left out.
	b := k.m.NewBatch()
	defer b.Free()
	b.Drafting = true
	ps := [][]int32{k.prompt(4, 80), k.prompt(5, 50)}
	var first []int32
	for _, p := range ps {
		tok, _ := join(b, p, 2)
		first = append(first, tok)
	}
	rows, last := k.run(b, first, []int{2, 1}, 16)
	// Row 0 ends; a longer one takes its place.
	b.Release(0)
	p2 := k.prompt(6, 300)
	tok2, row := join(b, p2, 2)
	if row != 0 {
		t.Fatalf("the newcomer went to row %d, not the free row 0", row)
	}
	more, last := k.run(b, []int32{tok2, last[1]}, []int{1, 3}, 20)
	rows[1] = append(rows[1], more[1]...)
	joined := with(tok2, more[0])
	// A third joins, for which the batch grows.
	p3 := k.prompt(7, 120)
	tok3, row := join(b, p3, 3)
	if row != 2 {
		t.Fatalf("the third went to row %d, not a new row 2", row)
	}
	more, last = k.run(b, []int32{last[0], last[1], tok3}, []int{1}, 20)
	rows[1], joined = append(rows[1], more[1]...), append(joined, more[0]...)
	third := with(tok3, more[2])
	// Rows 0 and 1 end and are dropped; the third goes on alone, drafting deeper.
	b.Release(0)
	b.Release(1)
	moved, err := b.Compact()
	if err != nil {
		t.Fatal(err)
	}
	if moved[2] != 0 || b.Len() != 1 {
		t.Fatalf("after two rows were dropped the batch has %d rows and row 2 went to %d", b.Len(), moved[2])
	}
	more, _ = k.run(b, []int32{last[2]}, []int{5}, 24)
	third = append(third, more[0]...)
	k.matches("the row there from the start", with(first[1], rows[1])[:n], ps[1])
	k.matches("the row that took a free row", joined[:36], p2)
	k.matches("the row the batch grew for", third[:40], p3)
}

// The head drafts for a row of a batch what it drafts for the sequence alone,
// pass after pass.
func TestQwen35BatchDraftsAsAlone(t *testing.T) {
	k := newDraftKit(t)
	defer k.m.Free()
	const d, passes = 3, 12
	p := k.prompt(8, 90)
	c, tok := k.read(p)
	defer c.Free()
	other, _ := k.read(p)
	b := k.m.NewBatch()
	defer b.Free()
	b.Drafting = true
	if _, err := b.Add(other, 1); err != nil {
		t.Fatal(err)
	}
	same := 0
	y := mlx.FromInt32([]int32{tok}, 1, 1)
	for i := 0; i < passes; i++ {
		// Alone.
		drafts, err := k.m.Draft(c, tok, d)
		if err != nil {
			t.Fatal(err)
		}
		mark := c.Mark()
		logits, hidden, err := k.m.Verify(tok, drafts, c)
		if err != nil {
			t.Fatal(err)
		}
		best := mlx.Astype(mlx.Argmax(logits, -1, false), mlx.Int32)
		mlx.Eval(append([]mlx.Array{hidden, best, drafts}, c.Arrays()...)...)
		ds, bs := drafts.Int32s(), best.Int32s()
		kept := 0
		for kept < d && bs[kept] == ds[kept] {
			kept++
		}
		if err := c.Keep(mark, kept, hidden); err != nil {
			t.Fatal(err)
		}
		for _, a := range []mlx.Array{drafts, logits, hidden, best} {
			a.Free()
		}

		// In the batch.
		s := mlx.NewScope()
		lg := b.Pass(y, d)
		bd := b.last.drafts.Ref()
		picks := mlx.Astype(mlx.Argmax(lg, -1, false), mlx.Int32)
		next, bk := b.Settle(picks)
		s.Escape(picks, next, bk, bd)
		s.Close()
		mlx.Eval(append([]mlx.Array{picks, next, bk, bd}, b.Arrays()...)...)
		gd, gp, gk := bd.Int32s(), picks.Int32s(), int(bk.Int32s()[0])
		if fmt.Sprint(gd) == fmt.Sprint(ds) && gk == kept && fmt.Sprint(gp[:kept+1]) == fmt.Sprint(bs[:kept+1]) {
			same++
		} else {
			t.Logf("pass %d: alone drafts %v, takes %v, keeps %d; in the batch drafts %v, takes %v, keeps %d",
				i, ds, bs[:kept+1], kept, gd, gp[:gk+1], gk)
			if i == 0 {
				t.Errorf("the first pass already differs")
			}
		}
		if gk != kept || gp[gk] != bs[kept] {
			// The two have parted, at a near tie or by a fault; no more to compare.
			for _, a := range []mlx.Array{picks, next, bk, bd} {
				a.Free()
			}
			break
		}
		tok = bs[kept]
		y.Free()
		y = next
		for _, a := range []mlx.Array{picks, bk, bd} {
			a.Free()
		}
	}
	y.Free()
	t.Logf("%d of %d passes drafted, took and kept the same alone and in a batch", same, passes)
	if same < passes*3/4 {
		t.Errorf("only %d of %d passes were the same", same, passes)
	}
}

// What a pass costs with the passes following one another as a server's do,
// each recorded and started before the one before it is read:
// KINFER_MLX_PASSES lists rows x drafts ("4x1,2x3"); "4x0" is a pass of one
// token for a batch that drafts, "4xs" the plain step of one that does not.
func TestQwen35BatchPassTime(t *testing.T) {
	list := os.Getenv("KINFER_MLX_PASSES")
	if list == "" {
		t.Skip("KINFER_MLX_PASSES not set")
	}
	k := newDraftKit(t)
	defer k.m.Free()
	const passes, warm = 40, 6
	for _, f := range strings.Split(list, ",") {
		var B, d int
		plain := strings.HasSuffix(f, "xs")
		if plain {
			fmt.Sscanf(f, "%dxs", &B)
		} else if _, err := fmt.Sscanf(f, "%dx%d", &B, &d); err != nil {
			t.Fatal(err)
		}
		b := k.m.NewBatch()
		b.Drafting = !plain
		var first []int32
		for r := 0; r < B; r++ {
			c, tok := k.read(k.prompt(20+r, 100+13*r))
			c.Drafting = !plain
			if _, err := b.Add(c, B); err != nil {
				t.Fatal(err)
			}
			first = append(first, tok)
		}
		y := mlx.FromInt32(first, B, 1)
		var el time.Duration
		kept := 0
		if plain {
			var start time.Time
			for i := 0; i < passes+warm; i++ {
				if i == warm {
					y.Int32s()
					start = time.Now()
				}
				s := mlx.NewScope()
				next := mlx.Reshape(mlx.Astype(mlx.Argmax(b.Step(y), -1, false), mlx.Int32), B, 1)
				s.Escape(next)
				s.Close()
				mlx.AsyncEvalStep(append([]mlx.Array{next}, b.Arrays()...)...)
				y.Int32s()
				y.Free()
				y = next
			}
			y.Int32s()
			el = time.Since(start)
			y.Free()
		} else {
			rows := make([][]int32, B)
			cur := k.start(b, y, d)
			y.Free()
			var start time.Time
			for i := 0; i < passes+warm; i++ {
				next := k.start(b, cur.next, d)
				n := cur.take(rows)
				cur.next.Free()
				cur = next
				if i == warm-1 {
					start, kept = time.Now(), 0
				} else if i >= warm {
					kept += n
				}
			}
			el = time.Since(start)
			cur.take(rows)
			cur.next.Free()
		}
		b.Free()
		ms := float64(el.Microseconds()) / 1000 / passes
		at := func(p float64) float64 {
			n, all := 1.0, 1.0
			for i := 0; i < d; i++ {
				all *= p
				n += all
			}
			return float64(B) * n / ms * 1000
		}
		t.Logf("%s: %.1f ms a pass, %.2f drafts kept a row; tok/s with none kept %.0f, half %.0f, 0.7 %.0f, 0.9 %.0f",
			f, ms, float64(kept)/float64(passes*B), at(0), at(0.5), at(0.7), at(0.9))
	}
}
