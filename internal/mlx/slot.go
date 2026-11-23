package mlx

import "unsafe"

// A Slot is an int32 array whose numbers are written after the graphs that
// read it are recorded: the tokens a pass starts from, when the pass is
// recorded before they are known. MLX reads an array when the GPU gets to
// the operation that takes it, so what is in the slot when its graph is
// evaluated is what the graph reads — Set, then evaluate, and do not Set
// again until what was evaluated has been read back. The array's memory is
// the GPU's and the host's alike on Apple silicon, which is what makes
// writing it from here mean anything.
type Slot struct {
	Array Array
	p     *int32
	n     int
}

// NewSlot makes a slot of the given shape, zeros until Set. It is the
// caller's to Free, whatever scope is open.
func NewSlot(shape ...int) *Slot {
	n := 1
	for _, d := range shape {
		n *= d
	}
	a := fromData(unsafe.Pointer(unsafe.SliceData(make([]int32, n))), n, shape, Int32)
	if current != nil {
		current.Escape(a)
	}
	a.Eval()
	p := arrayDataI32(a)
	if p == nil {
		check(1)
	}
	return &Slot{Array: a, p: p, n: n}
}

// Set writes v, one number for each of the slot's.
func (s *Slot) Set(v ...int32) {
	if len(v) != s.n {
		panic(&Error{"a slot is set with as many numbers as it holds"})
	}
	copy(unsafe.Slice(s.p, s.n), v)
}

// Free releases the slot's array.
func (s *Slot) Free() { s.Array.Free() }
