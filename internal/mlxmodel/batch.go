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
// is written into it or Compact drops the free rows in one copy.
type Qwen35Batch struct {
	m           *Qwen35
	conv, state []mlx.Array
	keys, vals  []mlx.Array
	idx         int
	pad         []int
	free        []bool
}

// NewBatch makes an empty batch.
func (m *Qwen35) NewBatch() *Qwen35Batch {
	n := len(m.layers)
	return &Qwen35Batch{m: m, conv: make([]mlx.Array, n), state: make([]mlx.Array, n),
		keys: make([]mlx.Array, n), vals: make([]mlx.Array, n)}
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
	for _, set := range [][]mlx.Array{b.conv, b.state, b.keys, b.vals} {
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
	n := len(b.conv)
	*b = Qwen35Batch{m: b.m, conv: make([]mlx.Array, n), state: make([]mlx.Array, n),
		keys: make([]mlx.Array, n), vals: make([]mlx.Array, n)}
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
		for _, kv := range []struct{ buf, row *mlx.Array }{{&b.keys[i], &c.keys[i]}, {&b.vals[i], &c.vals[i]}} {
			*kv.buf = mlx.SliceUpdate(*kv.buf, mlx.Slice(*kv.row, []int{0, 0, 0, 0}, []int{1, Hkv, n, D}),
				[]int{r, 0, b.idx - n, 0}, []int{r + 1, Hkv, b.idx, D})
		}
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
// free row when there is one — and returns its row; c is consumed.
func (b *Qwen35Batch) Add(c *Qwen35Cache) (row int, err error) {
	defer mlx.Recover(&err)
	scope := mlx.NewScope()
	defer scope.Close()
	old := b.Arrays()
	n := c.Offset

	if b.Len() == 0 {
		copy(b.conv, c.conv)
		copy(b.state, c.state)
		copy(b.keys, c.keys)
		copy(b.vals, c.vals)
		b.idx, b.pad, b.free = n, []int{0}, []bool{false}
		c.conv, c.state, c.keys, c.vals = make([]mlx.Array, len(c.conv)), make([]mlx.Array, len(c.conv)),
			make([]mlx.Array, len(c.conv)), make([]mlx.Array, len(c.conv))
		return 0, nil
	}

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
		row = b.appendRow(c)
	}

	fresh := b.Arrays()
	mlx.Eval(fresh...)
	scope.Escape(fresh...)
	freeOld(old, fresh)
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
	b.idx += shift
}

// appendRow adds c as a new last row and returns its index.
func (b *Qwen35Batch) appendRow(c *Qwen35Cache) int {
	n := c.Offset
	mc := b.m.Config
	Hkv, D := mc.NumKeyValueHeads, mc.HeadDim
	for i := range b.keys {
		if b.keys[i] == 0 {
			continue
		}
		for _, kv := range []struct{ buf, row *mlx.Array }{{&b.keys[i], &c.keys[i]}, {&b.vals[i], &c.vals[i]}} {
			capacity := (*kv.buf).Shape()[2]
			row := mlx.Zeros((*kv.buf).Dtype(), 1, Hkv, capacity, D)
			row = mlx.SliceUpdate(row, mlx.Slice(*kv.row, []int{0, 0, 0, 0}, []int{1, Hkv, n, D}),
				[]int{0, 0, b.idx - n, 0}, []int{1, Hkv, b.idx, D})
			*kv.buf = mlx.Concatenate(0, *kv.buf, row)
		}
	}
	for i := range b.conv {
		if b.conv[i] != 0 {
			b.conv[i] = mlx.Concatenate(0, b.conv[i], c.conv[i])
			b.state[i] = mlx.Concatenate(0, b.state[i], c.state[i])
		}
	}
	b.pad = append(b.pad, b.idx-n)
	b.free = append(b.free, false)
	return len(b.pad) - 1
}

// Remove drops the given rows, keeping the others in order.
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
	idx := mlx.FromInt32(keep, len(keep))
	for _, set := range [][]mlx.Array{b.conv, b.state, b.keys, b.vals} {
		for i, a := range set {
			if a != 0 {
				set[i] = mlx.Take(a, idx, 0)
			}
		}
	}
	b.pad, b.free = pad, free
	fresh := b.Arrays()
	mlx.Eval(fresh...)
	scope.Escape(fresh...)
	for _, a := range old {
		a.Free()
	}
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
	cols := mlx.Arange(0, float64(b.idx+1), 1, mlx.Int32)
	mask := mlx.Reshape(mlx.Greater(mlx.Reshape(cols, 1, b.idx+1),
		mlx.AddScalar(mlx.Reshape(mlx.FromInt32(pads, B), B, 1), -1)), B, 1, 1, b.idx+1)

	h := mlx.Take(m.embed, tokens, 0)
	for i := range m.layers {
		l := &m.layers[i]
		x := mlx.RMSNorm(h, l.inNorm, c.RMSNormEps)
		var r mlx.Array
		if l.linear != nil {
			one := &Qwen35Cache{conv: b.conv, state: b.state}
			r = m.deltaNet(l.linear, x, one, i)
		} else {
			r = b.attention(l.full, x, i, offsets, mask)
		}
		h = mlx.Add(h, r)
		x = mlx.RMSNorm(h, l.inPost, c.RMSNormEps)
		h = mlx.Add(h, linear(mlx.Multiply(mlx.SiLU(linear(x, l.gate)), linear(x, l.up)), l.down))
	}
	h = mlx.RMSNorm(h, m.norm, c.RMSNormEps)
	logits := mlx.Reshape(linear(h, m.embed), B, c.VocabSize)

	b.idx++
	scope.Keep(logits)
	scope.Escape(b.Arrays()...)
	for _, a := range old {
		a.Free()
	}
	return logits
}

func (b *Qwen35Batch) attention(w *fullAttn, x mlx.Array, i int, offsets, mask mlx.Array) mlx.Array {
	c := b.m.Config
	B := x.Shape()[0]
	H, Hkv, D := c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim

	qg := mlx.Split(mlx.Reshape(linear(x, w.q), B, 1, H, 2*D), -1, D)
	q, gate := qg[0], mlx.Reshape(qg[1], B, 1, H*D)
	k := mlx.Reshape(linear(x, w.k), B, 1, Hkv, D)
	v := mlx.Reshape(linear(x, w.v), B, 1, Hkv, D)
	q = mlx.Transpose(mlx.RMSNorm(q, w.qNorm, c.RMSNormEps), 0, 2, 1, 3)
	k = mlx.Transpose(mlx.RMSNorm(k, w.kNorm, c.RMSNormEps), 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	dims := int(float32(D) * c.RopeParameters.PartialRotaryFactor)
	q = mlx.RoPEAt(q, dims, false, c.RopeParameters.RopeTheta, 1, offsets)
	k = mlx.RoPEAt(k, dims, false, c.RopeParameters.RopeTheta, 1, offsets)

	end := b.idx + 1
	b.keys[i] = mlx.SliceUpdate(b.keys[i], k, []int{0, 0, b.idx, 0}, []int{B, Hkv, end, D})
	b.vals[i] = mlx.SliceUpdate(b.vals[i], v, []int{0, 0, b.idx, 0}, []int{B, Hkv, end, D})
	keys := mlx.Slice(b.keys[i], []int{0, 0, 0, 0}, []int{B, Hkv, end, D})
	vals := mlx.Slice(b.vals[i], []int{0, 0, 0, 0}, []int{B, Hkv, end, D})

	o := mlx.SDPA(q, keys, vals, float32(1/math.Sqrt(float64(D))), "", mask)
	o = mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), B, 1, H*D)
	return linear(mlx.Multiply(o, mlx.Sigmoid(gate)), w.o)
}
