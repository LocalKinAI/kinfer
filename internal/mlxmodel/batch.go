package mlxmodel

import (
	"math"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// Qwen35Batch is B sequences decoding together, one token each per step, so
// every weight is read once per step for all of them.
//
// Each sequence reads its prompt alone into a Qwen35Cache, then joins with
// Add. The DeltaNet state is per row already and stacks as is. Keys and
// values share one buffer per layer, [B, Hkv, capacity, D], written at one
// column for every row each step; a row that joined with a shorter history
// than the others sits right-aligned, its first pad[row] columns empty and
// masked out of attention. A row's next position is therefore idx - pad[row],
// which is what RoPE gets per row.
//
// A row whose sequence ended is released, not removed: removing means
// copying every other row's state, which at 64 rows is over a gigabyte per
// departure. A released row keeps being computed, uselessly, until a newcomer
// is written into it or Compact drops the free rows in one copy. The columns
// a long sequence leaves behind are cut when the next sequence joins.
//
// A batch whose rows draft (Drafting) is described in batch_draft.go: its
// keys and values have one more entry, the MTP head's, after the layers'.
type Qwen35Batch struct {
	m           *Qwen35
	conv, state []mlx.Array
	keys, vals  []mlx.Array // per layer, then the head's
	idx         int
	pad         []int
	free        []bool

	// Drafting is set, before the first row joins, for a batch whose rows
	// draft with the MTP head: every pass then feeds the head too, and each
	// row's head state is headH, [B, 1, hidden]. A row that joins with a head
	// that has not read all its model has ends it for the batch.
	Drafting bool
	headH    mlx.Array
	// valid and pos are there once the batch has run a Pass: valid[r][c] is
	// 1 where column c holds a token row r kept, [B, capacity] int32, and
	// pos[r] is how many tokens row r has read, [B] int32. Until then every
	// column from pad[r] on is the row's and pos is idx - pad.
	valid, pos mlx.Array
	// last is the pass Settle has still to be told the outcome of.
	last *passed
	// hv and hp are valid and pos on the host while rows join and leave.
	hv [][]int32
	hp []int32
}

// NewBatch makes an empty batch.
func (m *Qwen35) NewBatch() *Qwen35Batch {
	n := len(m.layers)
	return &Qwen35Batch{m: m, conv: make([]mlx.Array, n), state: make([]mlx.Array, n),
		keys: make([]mlx.Array, n+1), vals: make([]mlx.Array, n+1)}
}

// Len is the number of rows, free ones included.
func (b *Qwen35Batch) Len() int { return len(b.pad) }

// Free reports how many rows are free.
func (b *Qwen35Batch) FreeRows() int {
	n := 0
	for _, f := range b.free {
		if f {
			n++
		}
	}
	return n
}

// Release frees row r for the next sequence that joins.
func (b *Qwen35Batch) Release(r int) { b.free[r] = true }

// Compact removes the free rows, keeping the others in order, and returns
// where each kept row went (old index → new).
func (b *Qwen35Batch) Compact() (map[int]int, error) {
	var gone []int
	moved := map[int]int{}
	for r, f := range b.free {
		if f {
			gone = append(gone, r)
		} else {
			moved[r] = len(moved)
		}
	}
	return moved, b.Remove(gone...)
}

// Arrays are the batch's current arrays, to evaluate with a Step's logits.
func (b *Qwen35Batch) Arrays() []mlx.Array {
	var r []mlx.Array
	for _, set := range [][]mlx.Array{b.conv, b.state, b.keys, b.vals, {b.headH, b.valid, b.pos}} {
		for _, a := range set {
			if a != 0 {
				r = append(r, a)
			}
		}
	}
	return r
}

// Free releases the batch's arrays.
func (b *Qwen35Batch) Free() {
	for _, a := range b.Arrays() {
		a.Free()
	}
	if b.last != nil {
		b.last.free()
	}
	n := len(b.conv)
	*b = Qwen35Batch{m: b.m, conv: make([]mlx.Array, n), state: make([]mlx.Array, n),
		keys: make([]mlx.Array, n+1), vals: make([]mlx.Array, n+1)}
}

// rowKV is c's keys and values for the batch's entry i: a layer's, or the
// head's.
func (b *Qwen35Batch) rowKV(c *Qwen35Cache, i int) (keys, vals mlx.Array) {
	if i == len(b.m.layers) {
		return c.headKeys, c.headVals
	}
	return c.keys[i], c.vals[i]
}

// fill writes c, which has read n tokens, into free row r. The batch must
// already have room: idx >= n and a column past idx.
func (b *Qwen35Batch) fill(r int, c *Qwen35Cache) {
	n := c.Offset
	mc := b.m.Config
	Hkv, D := mc.NumKeyValueHeads, mc.HeadDim
	for i := range b.keys {
		if b.keys[i] == 0 {
			continue
		}
		ck, cv := b.rowKV(c, i)
		for _, kv := range []struct {
			buf *mlx.Array
			row mlx.Array
		}{{&b.keys[i], ck}, {&b.vals[i], cv}} {
			*kv.buf = mlx.SliceUpdate(*kv.buf, mlx.Slice(kv.row, []int{0, 0, 0, 0}, []int{1, Hkv, n, D}),
				[]int{r, 0, b.idx - n, 0}, []int{r + 1, Hkv, b.idx, D})
		}
	}
	if b.headH != 0 {
		b.headH = mlx.SliceUpdate(b.headH, c.headH, []int{r, 0, 0}, []int{r + 1, 1, mc.HiddenSize})
	}
	if b.hv != nil {
		b.hv[r] = ownColumns(b.idx, n)
		b.hp[r] = int32(n)
	}
	for i := range b.conv {
		if b.conv[i] == 0 {
			continue
		}
		for _, st := range []struct{ buf, row *mlx.Array }{{&b.conv[i], &c.conv[i]}, {&b.state[i], &c.state[i]}} {
			sh := (*st.buf).Shape()
			start := make([]int, len(sh))
			stop := append([]int(nil), sh...)
			start[0], stop[0] = r, r+1
			*st.buf = mlx.SliceUpdate(*st.buf, *st.row, start, stop)
		}
	}
	b.pad[r] = b.idx - n
	b.free[r] = false
}

func roundUp(n int) int { return (n + kvStep - 1) / kvStep * kvStep }

// Add puts a sequence that has read its prompt into c into the batch — into a
// free row when there is one — and returns its row; c is consumed. When the
// batch has to grow it grows to room rows at once, the rest free for the
// sequences the caller knows are coming: growing copies every row, so a
// burst of newcomers should pay for it once, while a free row costs a step's
// work for each step it stays empty.
func (b *Qwen35Batch) Add(c *Qwen35Cache, room int) (row int, err error) {
	defer mlx.Recover(&err)
	scope := mlx.NewScope()
	defer scope.Close()
	n := c.Offset
	if b.Len() == 0 {
		b.Drafting = b.Drafting && c.HeadReady()
	} else if b.Drafting && !c.HeadReady() {
		b.StopDrafting()
	}
	old := b.Arrays()

	if b.Len() == 0 {
		copy(b.conv, c.conv)
		copy(b.state, c.state)
		copy(b.keys, c.keys)
		copy(b.vals, c.vals)
		b.idx, b.pad, b.free = n, []int{0}, []bool{false}
		c.conv, c.state, c.keys, c.vals = make([]mlx.Array, len(c.conv)), make([]mlx.Array, len(c.conv)),
			make([]mlx.Array, len(c.conv)), make([]mlx.Array, len(c.conv))
		if b.Drafting {
			last := len(b.m.layers)
			b.keys[last], b.vals[last], b.headH = c.headKeys, c.headVals, c.headH
			c.headKeys, c.headVals, c.headH = 0, 0, 0
		}
		// What the batch did not take of c — the head's arrays, when the
		// batch does not draft — goes with it.
		c.Free()
		return 0, nil
	}

	b.pull()
	b.trim(b.idx - n)
	b.makeRoom(n)
	row = -1
	for r, f := range b.free {
		if f {
			row = r
			break
		}
	}
	if row >= 0 {
		b.fill(row, c)
	} else {
		row = b.appendRows(c, max(0, room-b.Len()-1))
	}

	b.push()

	// Our handles to the replaced arrays go before evaluating, so that MLX,
	// seeing the graph as their only user, writes the updates into them in
	// place instead of copying the whole batch for each newcomer.
	fresh := b.Arrays()
	scope.Escape(fresh...)
	freeOld(old, fresh)
	mlx.Eval(fresh...)
	c.Free()
	return row, nil
}

// freeOld frees the arrays of old that fresh no longer holds.
func freeOld(old, fresh []mlx.Array) {
	still := make(map[mlx.Array]bool, len(fresh))
	for _, a := range fresh {
		still[a] = true
	}
	for _, a := range old {
		if !still[a] {
			a.Free()
		}
	}
}

// makeRoom shifts every row right, if need be, so a sequence of n tokens
// fits before idx, and grows the buffers to hold column idx.
func (b *Qwen35Batch) makeRoom(n int) {
	mc := b.m.Config
	Hkv, D := mc.NumKeyValueHeads, mc.HeadDim
	shift := max(0, n-b.idx)
	B := b.Len()
	for i := range b.keys {
		if b.keys[i] == 0 {
			continue
		}
		for _, buf := range []*mlx.Array{&b.keys[i], &b.vals[i]} {
			capacity := max((*buf).Shape()[2], roundUp(b.idx+shift+1))
			if shift == 0 && capacity == (*buf).Shape()[2] {
				continue
			}
			grown := mlx.Zeros((*buf).Dtype(), B, Hkv, capacity, D)
			*buf = mlx.SliceUpdate(grown, mlx.Slice(*buf, []int{0, 0, 0, 0}, []int{B, Hkv, b.idx, D}),
				[]int{0, 0, shift, 0}, []int{B, Hkv, shift + b.idx, D})
		}
	}
	for r := range b.pad {
		b.pad[r] += shift
	}
	if shift > 0 {
		for r := range b.hv {
			b.hv[r] = append(make([]int32, shift), b.hv[r]...)
		}
	}
	b.idx += shift
}

// appendRows adds c as a new last row, then extra free rows, and returns c's
// row. A free row attends to one column of its own, so it computes something
// harmless rather than an empty softmax.
func (b *Qwen35Batch) appendRows(c *Qwen35Cache, extra int) int {
	n := c.Offset
	mc := b.m.Config
	Hkv, D := mc.NumKeyValueHeads, mc.HeadDim
	for i := range b.keys {
		if b.keys[i] == 0 {
			continue
		}
		ck, cv := b.rowKV(c, i)
		for _, kv := range []struct {
			buf *mlx.Array
			row mlx.Array
		}{{&b.keys[i], ck}, {&b.vals[i], cv}} {
			capacity := (*kv.buf).Shape()[2]
			row := mlx.Zeros((*kv.buf).Dtype(), 1+extra, Hkv, capacity, D)
			row = mlx.SliceUpdate(row, mlx.Slice(kv.row, []int{0, 0, 0, 0}, []int{1, Hkv, n, D}),
				[]int{0, 0, b.idx - n, 0}, []int{1, Hkv, b.idx, D})
			*kv.buf = mlx.Concatenate(0, *kv.buf, row)
		}
	}
	if b.headH != 0 {
		parts := []mlx.Array{b.headH, c.headH}
		if extra > 0 {
			parts = append(parts, mlx.Zeros(b.headH.Dtype(), extra, 1, mc.HiddenSize))
		}
		b.headH = mlx.Concatenate(0, parts...)
	}
	if b.hv != nil {
		b.hv, b.hp = append(b.hv, ownColumns(b.idx, n)), append(b.hp, int32(n))
		for k := 0; k < extra; k++ {
			b.hv, b.hp = append(b.hv, ownColumns(b.idx, 1)), append(b.hp, 1)
		}
	}
	for i := range b.conv {
		if b.conv[i] == 0 {
			continue
		}
		for _, st := range []struct{ buf, row *mlx.Array }{{&b.conv[i], &c.conv[i]}, {&b.state[i], &c.state[i]}} {
			parts := []mlx.Array{*st.buf, *st.row}
			if extra > 0 {
				sh := (*st.row).Shape()
				sh[0] = extra
				parts = append(parts, mlx.Zeros((*st.row).Dtype(), sh...))
			}
			*st.buf = mlx.Concatenate(0, parts...)
		}
	}
	row := len(b.pad)
	b.pad = append(b.pad, b.idx-n)
	b.free = append(b.free, false)
	for k := 0; k < extra; k++ {
		b.pad = append(b.pad, max(0, b.idx-1))
		b.free = append(b.free, true)
	}
	return row
}

// trim cuts off the columns that every live row has as padding — left behind
// by a long sequence that has since finished — but no more than the first
// most of them: each step attends over all of them, for every row, until
// they go. Fewer than kvStep are not worth copying the buffers for.
func (b *Qwen35Batch) trim(most int) {
	cut := most
	for r, p := range b.pad {
		if !b.free[r] {
			cut = min(cut, p)
		}
	}
	if cut < kvStep {
		return
	}
	mc := b.m.Config
	B, Hkv, D, n := len(b.pad), mc.NumKeyValueHeads, mc.HeadDim, b.idx-cut
	for _, set := range [][]mlx.Array{b.keys, b.vals} {
		for i, a := range set {
			if a != 0 {
				grown := mlx.Zeros(a.Dtype(), B, Hkv, roundUp(n+1), D)
				set[i] = mlx.SliceUpdate(grown, mlx.Slice(a, []int{0, 0, cut, 0}, []int{B, Hkv, b.idx, D}),
					[]int{0, 0, 0, 0}, []int{B, Hkv, n, D})
			}
		}
	}
	for r := range b.pad {
		b.pad[r] = max(0, b.pad[r]-cut)
	}
	for r := range b.hv {
		b.hv[r] = b.hv[r][cut:]
	}
	b.idx = n
}

// Remove drops the given rows, keeping the others in order, and trims the
// columns no remaining row uses.
func (b *Qwen35Batch) Remove(rows ...int) (err error) {
	defer mlx.Recover(&err)
	drop := map[int]bool{}
	for _, r := range rows {
		drop[r] = true
	}
	var keep []int32
	var pad []int
	var free []bool
	for r := range b.pad {
		if !drop[r] {
			keep = append(keep, int32(r))
			pad = append(pad, b.pad[r])
			free = append(free, b.free[r])
		}
	}
	if len(keep) == 0 {
		b.Free()
		return nil
	}
	scope := mlx.NewScope()
	defer scope.Close()
	old := b.Arrays()
	b.pull()
	idx := mlx.FromInt32(keep, len(keep))
	for _, set := range [][]mlx.Array{b.conv, b.state, b.keys, b.vals} {
		for i, a := range set {
			if a != 0 {
				set[i] = mlx.Take(a, idx, 0)
			}
		}
	}
	if b.headH != 0 {
		b.headH = mlx.Take(b.headH, idx, 0)
	}
	if b.hv != nil {
		hv, hp := make([][]int32, len(keep)), make([]int32, len(keep))
		for i, r := range keep {
			hv[i], hp[i] = b.hv[r], b.hp[r]
		}
		b.hv, b.hp = hv, hp
	}
	b.pad, b.free = pad, free
	b.trim(b.idx)
	b.push()

	fresh := b.Arrays()
	scope.Escape(fresh...)
	for _, a := range old {
		a.Free()
	}
	mlx.Eval(fresh...)
	return nil
}

// Step records one decode step: tokens [B, 1] int32, one per row, and returns
// the logits [B, vocab] without running anything, as Qwen35.Step does.
func (b *Qwen35Batch) Step(tokens mlx.Array) mlx.Array {
	m, c := b.m, b.m.Config
	B := b.Len()
	scope := mlx.NewScope()
	defer scope.Close()
	old := b.Arrays()

	// Grow the KV buffers when the next column does not fit.
	Hkv, D := c.NumKeyValueHeads, c.HeadDim
	for i := range b.keys {
		if b.keys[i] == 0 || b.keys[i].Shape()[2] > b.idx {
			continue
		}
		for _, buf := range []*mlx.Array{&b.keys[i], &b.vals[i]} {
			grown := mlx.Zeros((*buf).Dtype(), B, Hkv, roundUp(b.idx+1), D)
			*buf = mlx.SliceUpdate(grown, mlx.Slice(*buf, []int{0, 0, 0, 0}, []int{B, Hkv, b.idx, D}),
				[]int{0, 0, 0, 0}, []int{B, Hkv, b.idx, D})
		}
	}

	pos := make([]int32, B)
	for r, p := range b.pad {
		pos[r] = int32(b.idx - p)
	}
	offsets := mlx.FromInt32(pos, B)
	// Column j of row r is real when j >= pad[r].
	pads := make([]int32, B)
	for r, p := range b.pad {
		pads[r] = int32(p)
	}
	starts := mlx.FromInt32(pads, B)

	h := m.embed.rows(tokens)
	for i := range m.layers {
		l := &m.layers[i]
		x := mlx.RMSNorm(h, l.inNorm, c.RMSNormEps)
		if l.linear != nil {
			one := &Qwen35Cache{conv: b.conv, state: b.state}
			h = addProjected(h, m.deltaNet(l.linear, x, one, i), l.linear.outProj)
		} else {
			h = addProjected(h, b.attention(l.full, x, i, offsets, starts), l.full.o)
		}
		x = mlx.RMSNorm(h, l.inPost, c.RMSNormEps)
		h = l.addMLP(h, x)
	}
	h = mlx.RMSNorm(h, m.norm, c.RMSNormEps)
	logits := mlx.Reshape(linear(h, m.head()), B, c.VocabSize)

	b.idx++
	scope.Keep(logits)
	scope.Escape(b.Arrays()...)
	for _, a := range old {
		a.Free()
	}
	return logits
}

// attention is a full-attention layer's step up to its last projection.
// Each row reads only its own columns (decodeAttention), not the batch's
// whole width.
func (b *Qwen35Batch) attention(w *fullAttn, x mlx.Array, i int, offsets, starts mlx.Array) mlx.Array {
	c := b.m.Config
	B := x.Shape()[0]
	H, Hkv, D := c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim

	qg := mlx.Split(mlx.Reshape(project(x, w.q), B, 1, H, 2*D), -1, D)
	q, gate := qg[0], mlx.Reshape(qg[1], B, 1, H*D)
	k := mlx.Reshape(project(x, w.k), B, 1, Hkv, D)
	v := mlx.Reshape(project(x, w.v), B, 1, Hkv, D)
	q = mlx.Transpose(mlx.RMSNorm(q, w.qNorm, c.RMSNormEps), 0, 2, 1, 3)
	k = mlx.Transpose(mlx.RMSNorm(k, w.kNorm, c.RMSNormEps), 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	dims := int(float32(D) * c.RopeParameters.PartialRotaryFactor)
	q = mlx.RoPEAt(q, dims, false, c.RopeParameters.RopeTheta, 1, offsets)
	k = mlx.RoPEAt(k, dims, false, c.RopeParameters.RopeTheta, 1, offsets)

	end := b.idx + 1
	b.keys[i] = mlx.SliceUpdate(b.keys[i], k, []int{0, 0, b.idx, 0}, []int{B, Hkv, end, D})
	b.vals[i] = mlx.SliceUpdate(b.vals[i], v, []int{0, 0, b.idx, 0}, []int{B, Hkv, end, D})
	scale := float32(1 / math.Sqrt(float64(D)))
	var o mlx.Array
	if D%32 == 0 && raggedDecode {
		o = mlx.Reshape(decodeAttention(q, b.keys[i], b.vals[i], starts, end, scale), B, 1, H*D)
	} else {
		cols := mlx.Arange(0, float64(end), 1, mlx.Int32)
		mask := mlx.Reshape(mlx.Greater(mlx.Reshape(cols, 1, end), mlx.AddScalar(mlx.Reshape(starts, B, 1), -1)), B, 1, 1, end)
		keys := mlx.Slice(b.keys[i], []int{0, 0, 0, 0}, []int{B, Hkv, end, D})
		vals := mlx.Slice(b.vals[i], []int{0, 0, 0, 0}, []int{B, Hkv, end, D})
		o = mlx.SDPA(q, keys, vals, scale, "", mask)
		o = mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), B, 1, H*D)
	}
	return gated(o, gate)
}
