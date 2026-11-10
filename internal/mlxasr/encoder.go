package mlxasr

import (
	"fmt"
	"math"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// The audio encoder (mlx-audio's AudioEncoder): the features cut into chunks
// of 2·n_window frames (one second), each through three stride-2 convolutions
// to 13 positions, a projection and sinusoidal positions; the chunks' valid
// positions joined into one sequence; transformer layers in which a position
// attends only within its window of n_window_infer frames (eight seconds);
// then a norm and two projections to the decoder's width.

type encLayer struct {
	ln1w, ln1b, ln2w, ln2b mlx.Array
	q, k, v, o, fc1, fc2   linear
}

type encoder struct {
	cfg                          audioConfig
	c1w, c1b, c2w, c2b, c3w, c3b mlx.Array
	convOut, proj1, proj2        linear
	lnW, lnB                     mlx.Array
	layers                       []encLayer
	pos                          mlx.Array // [max positions, d] float32
}

const layerNormEps = 1e-5 // mlx.nn.LayerNorm's

// checkpoint, when set (KINFER_ASR_DEBUG), evaluates a at named points and
// logs MLX's memory there.
var checkpoint func(what string, a mlx.Array)

func loadEncoder(w *weights, cfg audioConfig) (*encoder, error) {
	e := &encoder{cfg: cfg}
	var err error
	p := "audio_tower."
	for _, x := range []struct {
		dst  *mlx.Array
		name string
	}{
		{&e.c1w, "conv2d1.weight"}, {&e.c1b, "conv2d1.bias"}, {&e.c2w, "conv2d2.weight"}, {&e.c2b, "conv2d2.bias"},
		{&e.c3w, "conv2d3.weight"}, {&e.c3b, "conv2d3.bias"}, {&e.lnW, "ln_post.weight"}, {&e.lnB, "ln_post.bias"},
	} {
		if *x.dst, err = w.get(p + x.name); err != nil {
			return nil, err
		}
	}
	if s := e.c1w.Shape(); len(s) != 4 || s[3] != 1 {
		return nil, fmt.Errorf("audio_tower.conv2d1.weight is %v, not [out, kh, kw, 1] as an MLX conversion stores it", s)
	}
	for _, x := range []struct {
		dst  *linear
		name string
	}{{&e.convOut, "conv_out"}, {&e.proj1, "proj1"}, {&e.proj2, "proj2"}} {
		if *x.dst, err = w.lin(p + x.name); err != nil {
			return nil, err
		}
	}
	e.layers = make([]encLayer, cfg.Layers)
	for i := range e.layers {
		l := &e.layers[i]
		lp := fmt.Sprintf("%slayers.%d.", p, i)
		for _, x := range []struct {
			dst  *mlx.Array
			name string
		}{
			{&l.ln1w, "self_attn_layer_norm.weight"}, {&l.ln1b, "self_attn_layer_norm.bias"},
			{&l.ln2w, "final_layer_norm.weight"}, {&l.ln2b, "final_layer_norm.bias"},
		} {
			if *x.dst, err = w.get(lp + x.name); err != nil {
				return nil, err
			}
		}
		for _, x := range []struct {
			dst  *linear
			name string
		}{
			{&l.q, "self_attn.q_proj"}, {&l.k, "self_attn.k_proj"}, {&l.v, "self_attn.v_proj"},
			{&l.o, "self_attn.out_proj"}, {&l.fc1, "fc1"}, {&l.fc2, "fc2"},
		} {
			if *x.dst, err = w.lin(lp + x.name); err != nil {
				return nil, err
			}
		}
	}
	// Sinusoidal positions, as SinusoidalPositionEmbedding: the sines of the
	// first half of the channels, then the cosines.
	d, half := cfg.DModel, cfg.DModel/2
	inc := math.Log(10000) / float64(half-1)
	pos := make([]float32, cfg.MaxPositions*d)
	for t := 0; t < cfg.MaxPositions; t++ {
		for i := 0; i < half; i++ {
			a := float64(t) * math.Exp(-inc*float64(i))
			pos[t*d+i] = float32(math.Sin(a))
			pos[t*d+half+i] = float32(math.Cos(a))
		}
	}
	e.pos = mlx.FromFloat32(pos, cfg.MaxPositions, d)
	return e, nil
}

func floorDiv(a, b int) int {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}

// convLen is how many positions the three stride-2 convolutions leave of n
// frames.
func convLen(n int) int {
	for i := 0; i < 3; i++ {
		n = floorDiv(n-1, 2) + 1
	}
	return n
}

// tokens is how many audio tokens frames of features become.
func (e *encoder) tokens(frames int) int {
	chunk := 2 * e.cfg.NWindow
	return (frames/chunk)*convLen(chunk) + convLen(frames%chunk)
}

// convGroup is how many one-second chunks go through the convolutions at a
// time. mlx-audio sends them all at once, but its GELU is compiled into one
// kernel; here each of the GELU's five ops has an intermediate as large as
// the convolution's output, 6 MB a chunk, so a group bounds what a long
// recording takes: 163 s at once peaked at 9.7 GB in the first convolution.
const convGroup = 16

// forward encodes features [mels, T] into [tokens(T), output dim]. Every
// stage runs in a scope of its own, so that each one's intermediates are let
// go as soon as the next has what it needs: kept until the end, as one scope
// keeps them, the 24 layers' intermediates added up to 19 GB for 163 s of
// sound.
func (e *encoder) forward(feats mlx.Array, T int) mlx.Array {
	cfg := e.cfg
	chunk := 2 * cfg.NWindow
	nC := (T + chunk - 1) / chunk
	x := feats
	if pad := nC*chunk - T; pad > 0 {
		x = mlx.Concatenate(1, x, mlx.Zeros(x.Dtype(), cfg.Mels, pad))
	}
	x = mlx.Transpose(mlx.Reshape(x, cfg.Mels, nC, chunk), 1, 0, 2) // [nC, mels, chunk]
	x = mlx.ExpandDims(x, 3)                                        // NHWC, one channel

	var groups []mlx.Array
	tc := convLen(chunk)
	for g := 0; g < nC; g += convGroup {
		n := min(convGroup, nC-g)
		s := mlx.NewScope()
		y := mlx.Slice(x, []int{g, 0, 0, 0}, []int{g + n, cfg.Mels, chunk, 1})
		y = mlx.GELU(mlx.Add(mlx.Conv2d(y, e.c1w, 2, 1), e.c1b))
		y = mlx.GELU(mlx.Add(mlx.Conv2d(y, e.c2w, 2, 1), e.c2b))
		y = mlx.GELU(mlx.Add(mlx.Conv2d(y, e.c3w, 2, 1), e.c3b))
		sh := y.Shape() // [n, freq, time, channels]
		y = mlx.Reshape(mlx.Transpose(y, 0, 2, 3, 1), n, sh[2], sh[3]*sh[1])
		y = e.convOut.apply(y)
		y = mlx.Add(y, mlx.Slice(e.pos, []int{0, 0}, []int{sh[2], cfg.DModel}))
		s.Escape(y)
		s.Close()
		mlx.Eval(y)
		groups = append(groups, y)
	}
	x = mlx.Concatenate(0, groups...) // [nC, tc, d]
	for _, g := range groups {
		g.Free()
	}
	if checkpoint != nil {
		checkpoint("convs", x)
	}
	d := cfg.DModel

	// The chunks' valid positions, one sequence: all of every chunk but the last.
	last := convLen(T - (nC-1)*chunk)
	var h mlx.Array
	if last == tc {
		h = mlx.Reshape(x, nC*tc, d)
	} else {
		tail := mlx.Reshape(mlx.Slice(x, []int{nC - 1, 0, 0}, []int{nC, last, d}), last, d)
		if nC == 1 {
			h = tail
		} else {
			head := mlx.Reshape(mlx.Slice(x, []int{0, 0, 0}, []int{nC - 1, tc, d}), (nC-1)*tc, d)
			h = mlx.Concatenate(0, head, tail)
		}
	}
	N := (nC-1)*tc + last
	most := tc
	if nC == 1 {
		most = last
	}
	window := most * (cfg.NWindowInfer / chunk)

	owned := false // h is ours to free, not the caller's scope's
	for i := range e.layers {
		l := &e.layers[i]
		s := mlx.NewScope()
		y := mlx.LayerNorm(h, l.ln1w, l.ln1b, layerNormEps)
		next := mlx.Add(h, e.attention(l, y, N, window))
		y = mlx.LayerNorm(next, l.ln2w, l.ln2b, layerNormEps)
		next = mlx.Add(next, l.fc2.apply(mlx.GELU(l.fc1.apply(y))))
		s.Escape(next)
		s.Close()
		if owned {
			h.Free()
		}
		h, owned = next, true
		if checkpoint != nil && (i == 0 || i == len(e.layers)-1) {
			checkpoint(fmt.Sprintf("layer%d", i), h)
		}
	}
	out := e.proj2.apply(mlx.GELU(e.proj1.apply(mlx.LayerNorm(h, e.lnW, e.lnB, layerNormEps))))
	if owned {
		h.Free()
	}
	return out
}

// attention is the encoder's self-attention over y [N, d], each position
// within its window: the windows batched, the last one, shorter, apart —
// what the block-diagonal mask of mlx-audio computes.
func (e *encoder) attention(l *encLayer, y mlx.Array, N, window int) mlx.Array {
	H := e.cfg.Heads
	d := y.Shape()[1]
	D := d / H
	q := mlx.MulScalar(l.q.apply(y), float32(1/math.Sqrt(float64(D))))
	k, v := l.k.apply(y), l.v.apply(y)
	part := func(start, nw, w int) mlx.Array {
		cut := func(a mlx.Array) mlx.Array {
			a = mlx.Slice(a, []int{start, 0}, []int{start + nw*w, d})
			return mlx.Transpose(mlx.Reshape(a, nw, w, H, D), 0, 2, 1, 3)
		}
		o := mlx.SDPA(cut(q), cut(k), cut(v), 1, "", 0)
		return mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), nw*w, d)
	}
	full, rem := N/window, N%window
	var o mlx.Array
	switch {
	case rem == 0:
		o = part(0, full, window)
	case full == 0:
		o = part(0, 1, rem)
	default:
		o = mlx.Concatenate(0, part(0, full, window), part(full*window, 1, rem))
	}
	return l.o.apply(o)
}
