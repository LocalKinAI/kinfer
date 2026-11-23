package mlxmodel

import (
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// The model's numbers as one line: the logits of a prompt, of steps after
// it, of a pass that checks drafts and of steps in a batch, each as its type
// and the md5 of its bits. Two builds that print the same line for a model
// compute the same thing for it, to the bit and in the same types — which a
// test inside one build cannot say of a change made to both of the things it
// compares. KINFER_MLX_FINGERPRINT=1 prints it.
func TestQwen35Fingerprint(t *testing.T) {
	lib, dir := os.Getenv("KINFER_MLX_LIB"), os.Getenv("KINFER_MLX_MODEL")
	if lib == "" || dir == "" || os.Getenv("KINFER_MLX_FINGERPRINT") == "" {
		t.Skip("KINFER_MLX_LIB, KINFER_MLX_MODEL and KINFER_MLX_FINGERPRINT not all set")
	}
	if err := mlx.Open(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadQwen35(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	line := ""
	mark := func(name string, a mlx.Array) {
		f := mlx.Astype(a, mlx.Float32).Float32s()
		b := make([]byte, 4*len(f))
		for i, x := range f {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
		}
		line += fmt.Sprintf(" %s=%v:%x", name, a.Dtype(), md5.Sum(b))
	}
	prompt := make([]int32, 300)
	for i := range prompt {
		prompt[i] = int32(1000 + (i*104729)%50000)
	}
	c := m.NewCache()
	defer c.Free()
	c.Drafting = m.CanDraft()
	lg, err := m.Forward([][]int32{prompt}, c, nil)
	if err != nil {
		t.Fatal(err)
	}
	mark("prompt", lg)
	lg.Free()
	for i, tok := range []int32{2001, 2002, 2003} {
		lg, err := m.Forward([][]int32{{tok}}, c, nil)
		if err != nil {
			t.Fatal(err)
		}
		mark(fmt.Sprint("step", i), lg)
		lg.Free()
	}
	var drafts mlx.Array
	if m.CanDraft() {
		if drafts, err = m.Draft(c, 2004, 3); err != nil {
			t.Fatal(err)
		}
		mark("drafts", drafts)
	} else {
		drafts = mlx.FromInt32([]int32{2005, 2006, 2007}, 1, 3)
	}
	mk := c.Mark()
	logits, hidden, err := m.Verify(2004, drafts, c)
	if err != nil {
		t.Fatal(err)
	}
	mlx.Eval(append([]mlx.Array{logits, hidden}, c.Arrays()...)...)
	mark("pass", logits)
	mark("hidden", hidden)
	if err := c.Keep(mk, 1, hidden); err != nil {
		t.Fatal(err)
	}
	logits.Free()
	hidden.Free()
	drafts.Free()
	if lg, err = m.Forward([][]int32{{2008}}, c, nil); err != nil {
		t.Fatal(err)
	}
	mark("kept", lg)
	lg.Free()

	b := m.NewBatch()
	defer b.Free()
	for r := 0; r < 3; r++ {
		rc := m.NewCache()
		lg, err := m.Forward([][]int32{prompt[:40+30*r]}, rc, nil)
		if err != nil {
			t.Fatal(err)
		}
		lg.Free()
		if _, err := b.Add(rc, 3); err != nil {
			t.Fatal(err)
		}
	}
	y := mlx.FromInt32([]int32{2009, 2010, 2011}, 3, 1)
	for i := 0; i < 3; i++ {
		lg := b.Step(y)
		mlx.Eval(append([]mlx.Array{lg}, b.Arrays()...)...)
		mark(fmt.Sprint("batch", i), lg)
		lg.Free()
	}
	y.Free()
	t.Logf("fingerprint:%s", line)
}
