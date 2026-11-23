package mlxmodel

import (
	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// Inkling2Batch is B sequences decoding together, one token each a step. A
// step of this model is some thirty operations for each of its 48 layer
// applications whatever it computes, and half its time is their overhead:
// that half is paid once for all the rows of a batch, as the weights every
// row shares are read once.
//
// It is laid out as Qwen35Batch is. Each sequence reads its prompt alone
// into an Inkling2Cache and joins with Add. Keys and values share one buffer
// a layer application, [B, Hkv, capacity, D], written at one column for
// every row each step; a row that joined with a shorter history sits
// right-aligned, its first pad[row] columns empty and left out of its
// attention. How far back a key is from a row's position is then the same
// number of columns for every row, which is all the attention's biases ask.
// A row whose sequence ended is released and computed on, uselessly, until a
// newcomer takes it or Compact drops the free rows.
type Inkling2Batch struct {
	m    *Inkling2
	c    *Inkling2Cache // every row's state; c.Offset is the column the next step writes, c.pad each row's first
	free []bool
}

// NewBatch makes an empty batch.
func (m *Inkling2) NewBatch() *Inkling2Batch {
	return &Inkling2Batch{m: m, c: m.NewCache()}
}

// Len is the number of rows, free ones included.
func (b *Inkling2Batch) Len() int { return len(b.free) }

// FreeRows reports how many rows are free.
func (b *Inkling2Batch) FreeRows() int {
	n := 0
	for _, f := range b.free {
		if f {
			n++
		}
	}
	return n
}

// Release frees row r for the next sequence that joins.
func (b *Inkling2Batch) Release(r int) { b.free[r] = true }

// Arrays are the batch's current arrays, to evaluate with a Step's logits.
func (b *Inkling2Batch) Arrays() []mlx.Array { return b.c.Arrays() }

// Free releases the batch's arrays.
func (b *Inkling2Batch) Free() {
	b.c.Free()
	b.c.pad, b.free = nil, nil
}

// Step records one decode step: tokens [B, 1] int32, one a row, and returns
// the logits [B, vocab] without running anything, as Inkling2.Step does.
func (b *Inkling2Batch) Step(tokens mlx.Array) mlx.Array { return b.m.Step(tokens, b.c) }

// buffers calls f for every key and value buffer of the batch.
func (b *Inkling2Batch) buffers(f func(buf *mlx.Array, visit int, keys bool)) {
	for v := range b.c.keys {
		if b.c.keys[v] != 0 {
			f(&b.c.keys[v], v, true)
			f(&b.c.vals[v], v, false)
		}
	}
}

// Add puts a sequence that has read its prompt into c into the batch — into
// a free row when there is one — and returns its row; c is consumed. When the
// batch has to grow it grows to room rows at once, the rest free for the
// sequences the caller knows are coming.
func (b *Inkling2Batch) Add(c *Inkling2Cache, room int) (row int, err error) {
	defer mlx.Recover(&err)
	n := c.Offset
	if b.Len() == 0 {
		b.c.Free()
		*b.c = *c
		b.c.pad, b.free = []int{0}, []bool{false}
		*c = Inkling2Cache{}
		return 0, nil
	}
	scope := mlx.NewScope()
	defer scope.Close()
	old := b.Arrays()

	b.trim(b.c.Offset - n)
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

// fill writes c, which has read n tokens, into free row r. The batch must
// already have room: Offset >= n.
func (b *Inkling2Batch) fill(r int, c *Inkling2Cache) {
	n, idx := c.Offset, b.c.Offset
	b.buffers(func(buf *mlx.Array, v int, keys bool) {
		from := c.vals[v]
		if keys {
			from = c.keys[v]
		}
		s := (*buf).Shape()
		*buf = mlx.SliceUpdate(*buf, mlx.Slice(from, []int{0, 0, 0, 0}, []int{1, s[1], n, s[3]}),
			[]int{r, 0, idx - n, 0}, []int{r + 1, s[1], idx, s[3]})
	})
	for v := range b.c.conv {
		for j, h := range b.c.conv[v] {
			if h == 0 {
				continue
			}
			s := h.Shape()
			b.c.conv[v][j] = mlx.SliceUpdate(h, c.conv[v][j], []int{r, 0, 0}, []int{r + 1, s[1], s[2]})
		}
	}
	b.c.pad[r] = idx - n
	b.free[r] = false
}

// makeRoom shifts every row right, if need be, so a sequence of n tokens
// fits before the column the next step writes.
func (b *Inkling2Batch) makeRoom(n int) {
	idx := b.c.Offset
	shift := n - idx
	if shift <= 0 {
		return
	}
	B := b.Len()
	b.buffers(func(buf *mlx.Array, _ int, _ bool) {
		s := (*buf).Shape()
		grown := mlx.Zeros((*buf).Dtype(), B, s[1], roundUp(idx+shift+1), s[3])
		*buf = mlx.SliceUpdate(grown, mlx.Slice(*buf, []int{0, 0, 0, 0}, []int{B, s[1], idx, s[3]}),
			[]int{0, 0, shift, 0}, []int{B, s[1], shift + idx, s[3]})
	})
	for r := range b.c.pad {
		b.c.pad[r] += shift
	}
	b.c.Offset += shift
}

// appendRows adds c as a new last row, then extra free rows, and returns c's
// row. A free row has one column of its own, so its attention is over
// something.
func (b *Inkling2Batch) appendRows(c *Inkling2Cache, extra int) int {
	n, idx := c.Offset, b.c.Offset
	b.buffers(func(buf *mlx.Array, v int, keys bool) {
		from := c.vals[v]
		if keys {
			from = c.keys[v]
		}
		s := (*buf).Shape()
		row := mlx.Zeros((*buf).Dtype(), 1+extra, s[1], s[2], s[3])
		row = mlx.SliceUpdate(row, mlx.Slice(from, []int{0, 0, 0, 0}, []int{1, s[1], n, s[3]}),
			[]int{0, 0, idx - n, 0}, []int{1, s[1], idx, s[3]})
		*buf = mlx.Concatenate(0, *buf, row)
	})
	for v := range b.c.conv {
		for j, h := range b.c.conv[v] {
			if h == 0 {
				continue
			}
			parts := []mlx.Array{h, c.conv[v][j]}
			if extra > 0 {
				s := h.Shape()
				parts = append(parts, mlx.Zeros(h.Dtype(), extra, s[1], s[2]))
			}
			b.c.conv[v][j] = mlx.Concatenate(0, parts...)
		}
	}
	row := len(b.free)
	b.c.pad, b.free = append(b.c.pad, idx-n), append(b.free, false)
	for k := 0; k < extra; k++ {
		b.c.pad, b.free = append(b.c.pad, max(0, idx-1)), append(b.free, true)
	}
	return row
}

// trim cuts off the columns that every live row has as padding — left behind
// by a long sequence that has since finished — but no more than the first
// most of them. Fewer than kvStep are not worth copying the buffers for.
func (b *Inkling2Batch) trim(most int) {
	cut := most
	for r, p := range b.c.pad {
		if !b.free[r] {
			cut = min(cut, p)
		}
	}
	if cut < kvStep {
		return
	}
	B, n := b.Len(), b.c.Offset-cut
	b.buffers(func(buf *mlx.Array, _ int, _ bool) {
		s := (*buf).Shape()
		grown := mlx.Zeros((*buf).Dtype(), B, s[1], roundUp(n+1), s[3])
		*buf = mlx.SliceUpdate(grown, mlx.Slice(*buf, []int{0, 0, cut, 0}, []int{B, s[1], b.c.Offset, s[3]}),
			[]int{0, 0, 0, 0}, []int{B, s[1], n, s[3]})
	})
	for r := range b.c.pad {
		b.c.pad[r] = max(0, b.c.pad[r]-cut)
	}
	b.c.Offset = n
}

// Compact removes the free rows, keeping the others in order, and returns
// where each kept row went (old index → new).
func (b *Inkling2Batch) Compact() (map[int]int, error) {
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

// Remove drops the given rows, keeping the others in order, and trims the
// columns no remaining row uses.
func (b *Inkling2Batch) Remove(rows ...int) (err error) {
	defer mlx.Recover(&err)
	drop := map[int]bool{}
	for _, r := range rows {
		drop[r] = true
	}
	var keep []int32
	var pad []int
	var free []bool
	for r := range b.free {
		if !drop[r] {
			keep = append(keep, int32(r))
			pad = append(pad, b.c.pad[r])
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
	b.buffers(func(buf *mlx.Array, _ int, _ bool) { *buf = mlx.Take(*buf, idx, 0) })
	for v := range b.c.conv {
		for j, h := range b.c.conv[v] {
			if h != 0 {
				b.c.conv[v][j] = mlx.Take(h, idx, 0)
			}
		}
	}
	b.c.pad, b.free = pad, free
	b.trim(b.c.Offset)

	fresh := b.Arrays()
	scope.Escape(fresh...)
	for _, a := range old {
		a.Free()
	}
	mlx.Eval(fresh...)
	return nil
}
