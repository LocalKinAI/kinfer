package mlxmodel

import (
	"math"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// Drafting in a batch.
//
// A reply writing alone drafts with the MTP head and has the model check the
// drafts in one pass (Verify). Pass is that for every row of a batch at once:
// each row reads the token it is sure of and d drafts after it, so the pass
// is B × (d+1) rows through every weight, read once. Settle then says what
// each row took — it keeps the drafts that are what it would have written
// itself, up to the first that is not — and puts every row back to just
// after the last token it kept.
//
// Rows keep different numbers of drafts, and the batch writes every row's
// keys and values at the same columns. So the columns of the drafts a row
// did not keep stay where they are, marked not the row's (valid), and every
// later pass leaves them out of that row's attention; a row's position, which
// RoPE reads, is counted on its own (pos). What Settle does is all on the
// GPU, from the tokens the rows take as they come out of the sampling — so
// the pass after can be recorded and started while this one is still being
// worked out, as a plain step is, and the host only reads what was written.
//
// The DeltaNet layers leave their state after every token of the pass, and
// their convolution inputs, from which Settle picks each row's.

// passed is what a Pass leaves for Settle.
type passed struct {
	L              int
	steps, convIns []mlx.Array // per DeltaNet layer; 0 for a pass of one token
	hidden         mlx.Array   // [B, L, hidden], the head's next state among them; 0 without the head
	drafts         mlx.Array   // [B, L-1]; 0 for a pass of one token
}

func (p *passed) arrays() []mlx.Array {
	var r []mlx.Array
	for _, set := range [][]mlx.Array{p.steps, p.convIns, {p.hidden, p.drafts}} {
		for _, a := range set {
			if a != 0 {
				r = append(r, a)
			}
		}
	}
	return r
}

func (p *passed) free() {
	for _, a := range p.arrays() {
		a.Free()
	}
	*p = passed{}
}

// Plain reports whether the batch decodes with Step: it does not draft and
// has no columns left out of a row.
func (b *Qwen35Batch) Plain() bool { return !b.Drafting && b.valid == 0 }

// StopDrafting ends drafting for the batch: the head is no longer fed.
func (b *Qwen35Batch) StopDrafting() {
	n := len(b.m.layers)
	for _, a := range []*mlx.Array{&b.keys[n], &b.vals[n], &b.headH} {
		if *a != 0 {
			(*a).Free()
			*a = 0
		}
	}
	b.Drafting = false
}

// ownColumns is a row's columns in a batch idx wide when its last n are its
// own.
func ownColumns(idx, n int) []int32 {
	row := make([]int32, idx)
	for c := idx - n; c < idx; c++ {
		row[c] = 1
	}
	return row
}

// pull reads valid and pos to the host, where rows joining and leaving change
// them along with the rest (fill, makeRoom, appendRows, trim, Remove); push
// writes them back. Both do nothing for a batch that has not run a Pass.
func (b *Qwen35Batch) pull() {
	if b.valid == 0 {
		return
	}
	B, w := b.Len(), b.valid.Shape()[1]
	flat := b.valid.Int32s()
	b.hv = make([][]int32, B)
	for r := range b.hv {
		b.hv[r] = append([]int32(nil), flat[r*w:r*w+b.idx]...)
	}
	b.hp = append([]int32(nil), b.pos.Int32s()...)
}

func (b *Qwen35Batch) push() {
	if b.hv == nil {
		return
	}
	B, w := len(b.hv), roundUp(b.idx+1)
	flat := make([]int32, B*w)
	for r, row := range b.hv {
		// A free row whose columns were cut keeps one, so that its attention
		// is over something.
		any := false
		for _, v := range row {
			any = any || v != 0
		}
		copy(flat[r*w:], row)
		if !any && b.idx > 0 {
			flat[r*w+b.idx-1] = 1
		}
	}
	b.valid, b.pos = mlx.FromInt32(flat, B, w), mlx.FromInt32(b.hp, B)
	b.hv, b.hp = nil, nil
}

// ready gives the batch its valid and pos when it has none yet.
func (b *Qwen35Batch) ready() {
	if b.valid != 0 {
		return
	}
	B, w := b.Len(), roundUp(b.idx+1)
	flat, pos := make([]int32, B*w), make([]int32, B)
	for r, p := range b.pad {
		copy(flat[r*w:], ownColumns(b.idx, b.idx-p))
		pos[r] = int32(b.idx - p)
	}
	b.valid, b.pos = mlx.FromInt32(flat, B, w), mlx.FromInt32(pos, B)
}

// grow makes the key and value buffers, and valid, at least need columns
// wide.
func (b *Qwen35Batch) grow(need int) {
	c := b.m.Config
	B, Hkv, D := b.Len(), c.NumKeyValueHeads, c.HeadDim
	for i := range b.keys {
		if b.keys[i] == 0 || b.keys[i].Shape()[2] >= need {
			continue
		}
		for _, buf := range []*mlx.Array{&b.keys[i], &b.vals[i]} {
			grown := mlx.Zeros((*buf).Dtype(), B, Hkv, roundUp(need), D)
			*buf = mlx.SliceUpdate(grown, mlx.Slice(*buf, []int{0, 0, 0, 0}, []int{B, Hkv, b.idx, D}),
				[]int{0, 0, 0, 0}, []int{B, Hkv, b.idx, D})
		}
	}
	if b.valid != 0 && b.valid.Shape()[1] < need {
		grown := mlx.Zeros(mlx.Int32, B, roundUp(need))
		b.valid = mlx.SliceUpdate(grown, mlx.Slice(b.valid, []int{0, 0}, []int{B, b.idx}), []int{0, 0}, []int{B, b.idx})
	}
}

// causal is the mask of a pass's own columns, [1, 1, L, L]: each position
// reads those up to itself.
func causal(L int) mlx.Array {
	tri := make([]int32, L*L)
	for i := 0; i < L; i++ {
		for j := 0; j <= i; j++ {
			tri[i*L+j] = 1
		}
	}
	return mlx.Greater(mlx.FromInt32(tri, 1, 1, L, L), mlx.ConstInt(0))
}

// Pass records a pass in which every row reads its token, tokens [B, 1]
// int32, and then d drafts of the head's after it; d is 0 for a batch that
// does not draft. It returns the logits after each of them, [B*(d+1), vocab],
// row r's at r*(d+1) on, without running anything, as Step does. Settle must
// follow, with what the rows take, before the next Pass.
func (b *Qwen35Batch) Pass(tokens mlx.Array, d int) mlx.Array {
	m, c := b.m, b.m.Config
	B, L, n, eps := b.Len(), d+1, len(m.layers), b.m.Config.RMSNormEps
	scope := mlx.NewScope()
	defer scope.Close()
	old := b.Arrays()
	b.ready()
	b.grow(b.idx + L)

	// seen is what each row had before the pass; the pass's own columns are
	// read causally, drafts taken as written.
	seen := mlx.Reshape(mlx.Greater(mlx.Slice(b.valid, []int{0, 0}, []int{B, b.idx}), mlx.ConstInt(0)), B, 1, 1, b.idx)
	p := &passed{L: L}
	ids := tokens
	if d > 0 {
		p.drafts = b.draft(tokens, d, seen)
		ids = mlx.Concatenate(1, tokens, p.drafts)
	}
	mask := mlx.Concatenate(-1, mlx.BroadcastTo(seen, B, 1, L, b.idx), mlx.BroadcastTo(causal(L), B, 1, L, L))

	h := m.embed.rows(ids)
	one := &Qwen35Cache{conv: b.conv, state: b.state, keepSteps: L > 1,
		steps: make([]mlx.Array, n), convIns: make([]mlx.Array, n)}
	for i := range m.layers {
		l := &m.layers[i]
		x := mlx.RMSNorm(h, l.inNorm, eps)
		if l.linear != nil {
			h = addProjected(h, m.deltaNet(l.linear, x, one, i), l.linear.outProj)
		} else {
			h = addProjected(h, b.attend(l.full, x, i, b.idx, b.pos, mask), l.full.o)
		}
		x = mlx.RMSNorm(h, l.inPost, eps)
		h = l.addMLP(h, x)
	}
	all := mlx.RMSNorm(h, m.norm, eps)
	if b.Drafting {
		// The head reads each token paired with the model's state before it:
		// the row's own for the first, the pass's for the rest.
		before := b.headH
		if L > 1 {
			before = mlx.Concatenate(1, before, mlx.Slice(all, []int{0, 0, 0}, []int{B, L - 1, c.HiddenSize}))
		}
		b.headRead(ids, before, b.idx, b.pos, mask)
		p.hidden = all
	}
	logits := mlx.Reshape(linear(all, m.head()), B*L, c.VocabSize)
	p.steps, p.convIns = one.steps, one.convIns

	b.last = p
	scope.Keep(logits)
	fresh := b.Arrays()
	scope.Escape(fresh...)
	scope.Escape(p.arrays()...)
	freeOld(old, fresh)
	return logits
}

// draft has the head propose d tokens after each row's token, [B, d]. The
// head's keys and values for them are written where the pass that checks them
// writes its own, over them.
func (b *Qwen35Batch) draft(tokens mlx.Array, d int, seen mlx.Array) mlx.Array {
	m, B := b.m, b.Len()
	t, h := tokens, b.headH
	toks := make([]mlx.Array, d)
	for i := range d {
		mask := mlx.Concatenate(-1, seen, mlx.Greater(mlx.Ones(mlx.Int32, B, 1, 1, i+1), mlx.ConstInt(0)))
		at := b.pos
		if i > 0 {
			at = mlx.Add(b.pos, mlx.ConstInt(i))
		}
		h = b.headRead(t, h, b.idx+i, at, mask)
		t = mlx.Astype(mlx.Argmax(linear(h, m.head()), -1, false), mlx.Int32)
		toks[i] = t
	}
	return mlx.Concatenate(1, toks...)
}

// headRead is one pass of the MTP head over tokens [B, L] and the states
// paired with them, [B, L, hidden], written at columns col on; it returns the
// head's normed output, as Qwen35.headRead does for one sequence.
func (b *Qwen35Batch) headRead(tokens, before mlx.Array, col int, at, mask mlx.Array) mlx.Array {
	m := b.m
	t, eps, n := m.mtp, m.Config.RMSNormEps, len(m.layers)
	e := m.embed.rows(tokens)
	x := linear(mlx.Concatenate(-1, mlx.RMSNorm(e, t.enorm, eps), mlx.RMSNorm(before, t.hnorm, eps)), t.fc)
	x = addProjected(x, b.attend(t.block.full, mlx.RMSNorm(x, t.block.inNorm, eps), n, col, at, mask), t.block.full.o)
	y := mlx.RMSNorm(x, t.block.inPost, eps)
	x = t.block.addMLP(x, y)
	return mlx.RMSNorm(x, t.norm, eps)
}

// attend is a full-attention layer of x [B, L, hidden] up to its last
// projection: the rows' keys and values go into columns [col, col+L) of the
// batch's entry i, each row's at its own positions from at [B] on, and every
// new position reads the columns mask gives it, [B, 1, L, col+L].
func (b *Qwen35Batch) attend(w *fullAttn, x mlx.Array, i, col int, at, mask mlx.Array) mlx.Array {
	c := b.m.Config
	s := x.Shape()
	B, L := s[0], s[1]
	H, Hkv, D := c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim

	qg := mlx.Split(mlx.Reshape(project(x, w.q), B, L, H, 2*D), -1, D)
	q, gate := qg[0], mlx.Reshape(qg[1], B, L, H*D)
	k := mlx.Reshape(project(x, w.k), B, L, Hkv, D)
	v := mlx.Reshape(project(x, w.v), B, L, Hkv, D)
	q = mlx.Transpose(mlx.RMSNorm(q, w.qNorm, c.RMSNormEps), 0, 2, 1, 3)
	k = mlx.Transpose(mlx.RMSNorm(k, w.kNorm, c.RMSNormEps), 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	dims := int(float32(D) * c.RopeParameters.PartialRotaryFactor)
	q = mlx.RoPEAt(q, dims, false, c.RopeParameters.RopeTheta, 1, at)
	k = mlx.RoPEAt(k, dims, false, c.RopeParameters.RopeTheta, 1, at)

	end := col + L
	b.keys[i] = mlx.SliceUpdate(b.keys[i], k, []int{0, 0, col, 0}, []int{B, Hkv, end, D})
	b.vals[i] = mlx.SliceUpdate(b.vals[i], v, []int{0, 0, col, 0}, []int{B, Hkv, end, D})
	keys := mlx.Slice(b.keys[i], []int{0, 0, 0, 0}, []int{B, Hkv, end, D})
	vals := mlx.Slice(b.vals[i], []int{0, 0, 0, 0}, []int{B, Hkv, end, D})
	o := mlx.SDPA(q, keys, vals, float32(1/math.Sqrt(float64(D))), "", mask)
	o = mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), B, L, H*D)
	return gated(o, gate)
}

// Settle tells the batch what its rows take of the last Pass: picks is the
// token each row would write at each of the pass's positions, [B*L] or [B, L]
// int32 — the sampling of the pass's logits, evaluated or not. A row keeps
// the drafts that are its picks, up to the first that is not, and writes its
// own pick there; the batch goes back to just after it. It returns that
// token for every row, [B, 1], which the next pass reads, and how many drafts
// each row kept, [B], neither evaluated: a row has written picks[r][0..kept].
func (b *Qwen35Batch) Settle(picks mlx.Array) (next, kept mlx.Array) {
	c := b.m.Config
	p := b.last
	B, L := b.Len(), p.L
	scope := mlx.NewScope()
	defer scope.Close()
	old := b.Arrays()
	hidden := p.hidden
	if L == 1 {
		next, kept = mlx.Reshape(picks, B, 1), mlx.Zeros(mlx.Int32, B)
		if hidden != 0 {
			b.headH, p.hidden = hidden, 0
		}
		b.pos = mlx.Add(b.pos, mlx.ConstInt(1))
		b.valid = mlx.SliceUpdate(b.valid, mlx.Ones(mlx.Int32, B, 1), []int{0, b.idx}, []int{B, b.idx + 1})
	} else {
		picks = mlx.Reshape(picks, B, L)
		// kept: the drafts before the first that is not the row's pick.
		miss := mlx.Subtract(mlx.ConstInt(1), mlx.Astype(mlx.Equal(mlx.Slice(picks, []int{0, 0}, []int{B, L - 1}), p.drafts), mlx.Int32))
		none := mlx.Equal(mlx.Cumsum(miss, 1, false, true), mlx.ConstInt(0))
		kept = mlx.Sum(mlx.Astype(none, mlx.Int32), 1, false)

		// Row r's place in an array of B*L: the last token it kept.
		K := c.LinearConvKernelDim
		first, conv := make([]int32, B), make([]int32, B*(K-1))
		for r := 0; r < B; r++ {
			first[r] = int32(r * L)
			for j := 0; j < K-1; j++ {
				conv[r*(K-1)+j] = int32(r*(K-1+L) + 1 + j)
			}
		}
		sel := mlx.Add(kept, mlx.FromInt32(first, B))
		csel := mlx.Reshape(mlx.Add(mlx.FromInt32(conv, B, K-1), mlx.Reshape(kept, B, 1)), B*(K-1))
		next = mlx.Reshape(mlx.Take(mlx.Reshape(picks, B*L), sel, 0), B, 1)
		for l, st := range p.steps {
			if st == 0 {
				continue
			}
			s := st.Shape()
			b.state[l] = mlx.Take(mlx.Reshape(st, B*L, s[2], s[3], s[4]), sel, 0)
			in := p.convIns[l]
			C := in.Shape()[2]
			b.conv[l] = mlx.Reshape(mlx.Take(mlx.Reshape(in, B*(K-1+L), C), csel, 0), B, K-1, C)
		}
		if hidden != 0 {
			H := c.HiddenSize
			b.headH = mlx.Reshape(mlx.Take(mlx.Reshape(hidden, B*L, H), sel, 0), B, 1, H)
		}
		b.pos = mlx.Add(b.pos, mlx.Add(kept, mlx.ConstInt(1)))
		cols := mlx.Reshape(mlx.Arange(0, float64(L), 1, mlx.Int32), 1, L)
		own := mlx.Astype(mlx.Less(cols, mlx.Add(mlx.Reshape(kept, B, 1), mlx.ConstInt(1))), mlx.Int32)
		b.valid = mlx.SliceUpdate(b.valid, own, []int{0, b.idx}, []int{B, b.idx + L})
	}
	b.idx += L
	scope.Keep(next, kept)
	fresh := b.Arrays()
	scope.Escape(fresh...)
	freeOld(old, fresh)
	p.free()
	b.last = nil
	return next, kept
}
