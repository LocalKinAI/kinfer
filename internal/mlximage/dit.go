package mlximage

import (
	"fmt"
	"math"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// The diffusion transformer: Z-Image's single-stream DiT as mflux's
// ZImageTransformer — the picture's 2×2 patches and the caption refined apart
// (two blocks each), then thirty blocks over both, each modulated by the
// timestep (adaLN), with a three-axis rotary position.
const (
	ditDim     = 3840
	ditHeads   = 30
	ditHeadDim = 128
	ditEps     = 1e-5
	capDim     = 2560
	patch      = 2
	latentC    = 16
	padTo      = 32 // the image's and the caption's tokens are padded to a multiple of it
	tScale     = 1000
)

var (
	ropeAxes  = [3]int{32, 48, 48}
	ropeLens  = [3]int{1024, 512, 512}
	ropeTheta = 256.0
)

type block struct {
	q, k, v, o         linear
	normQ, normK       mlx.Array
	w1, w2, w3         linear
	an1, an2, fn1, fn2 mlx.Array
	ada                linear
	modulated          bool
}

type dit struct {
	xEmb, finalLin, finalAda linear
	t1, t2                   linear
	capNorm                  mlx.Array
	capLin                   linear
	xPad, capPad             mlx.Array
	noise, context, layers   []block
	rope                     [3]mlx.Array // [len, d/2, 2] float32: cos, sin
}

func loadBlock(w *weights, p string, modulated bool) block {
	b := block{
		q: w.lin(p + "attention.to_q"), k: w.lin(p + "attention.to_k"), v: w.lin(p + "attention.to_v"),
		o: w.lin(p + "attention.to_out.0"), normQ: w.get(p + "attention.norm_q.weight"), normK: w.get(p + "attention.norm_k.weight"),
		w1: w.lin(p + "feed_forward.w1"), w2: w.lin(p + "feed_forward.w2"), w3: w.lin(p + "feed_forward.w3"),
		an1: w.get(p + "attention_norm1.weight"), an2: w.get(p + "attention_norm2.weight"),
		fn1: w.get(p + "ffn_norm1.weight"), fn2: w.get(p + "ffn_norm2.weight"), modulated: modulated,
	}
	if modulated {
		b.ada = w.lin(p + "adaLN_modulation.0")
	}
	return b
}

func loadDiT(w *weights) *dit {
	key := fmt.Sprintf("%d-%d", patch, 1)
	d := &dit{
		xEmb: w.lin("all_x_embedder." + key), finalLin: w.lin("all_final_layer." + key + ".linear"),
		finalAda: w.lin("all_final_layer." + key + ".adaLN_modulation.0"),
		t1:       w.lin("t_embedder.linear1"), t2: w.lin("t_embedder.linear2"),
		capNorm: w.get("cap_embedder.0.weight"), capLin: w.lin("cap_embedder.1"),
		xPad: w.get("x_pad_token"), capPad: w.get("cap_pad_token"),
	}
	for i := 0; i < 2; i++ {
		d.noise = append(d.noise, loadBlock(w, fmt.Sprintf("noise_refiner.%d.", i), true))
		d.context = append(d.context, loadBlock(w, fmt.Sprintf("context_refiner.%d.", i), false))
	}
	for i := 0; i < 30; i++ {
		d.layers = append(d.layers, loadBlock(w, fmt.Sprintf("layers.%d.", i), true))
	}
	// RopeEmbedder's tables, as _precompute_freqs_cis makes them.
	for i, dim := range ropeAxes {
		f := mlx.Divide(mlx.Scalar(1), mlx.Power(mlx.Scalar(float32(ropeTheta)),
			mlx.Divide(mlx.Arange(0, float64(dim), 2, mlx.Float32), mlx.Scalar(float32(dim)))))
		t := mlx.Arange(0, float64(ropeLens[i]), 1, mlx.Float32)
		fr := mlx.Multiply(mlx.Reshape(t, ropeLens[i], 1), mlx.Reshape(f, 1, dim/2))
		d.rope[i] = mlx.Stack(-1, mlx.Cos(fr), mlx.Sin(fr))
		mlx.Eval(d.rope[i])
	}
	return d
}

func (d *dit) free() {
	for _, r := range d.rope {
		if r != 0 {
			r.Free()
		}
	}
}

// timestepEmbedding is TimestepEmbedder's sinusoid of t [1], float32 [1, 256].
func timestepEmbedding(t mlx.Array) mlx.Array {
	const half = 128
	freqs := mlx.Exp(mlx.Divide(mlx.Multiply(mlx.Scalar(float32(-math.Log(10000))), mlx.Arange(0, half, 1, mlx.Float32)), mlx.Scalar(half)))
	args := mlx.Multiply(mlx.Reshape(mlx.Astype(t, mlx.Float32), 1, 1), mlx.Reshape(freqs, 1, half))
	return mlx.Concatenate(-1, mlx.Cos(args), mlx.Sin(args))
}

// timeEmbedding is TimestepEmbedder of t·1000 in the stream's dtype, as
// stream_t_emb casts it: [1, 256].
func (d *dit) timeEmbedding(t mlx.Array) mlx.Array {
	ts := mlx.MulScalar(mlx.Astype(t, mlx.Float32), tScale)
	return mlx.Astype(d.t2.apply(mlx.SiLU(d.t1.apply(timestepEmbedding(ts)))), mlx.Bfloat16)
}

// ropeFor is the rotary angles of each position id [n, 3]: [n, 64, 2].
func (d *dit) ropeFor(ids []int32, n int) mlx.Array {
	var parts []mlx.Array
	for i := range ropeAxes {
		col := make([]int32, n)
		for j := 0; j < n; j++ {
			col[j] = ids[j*3+i]
		}
		parts = append(parts, mlx.Take(d.rope[i], mlx.FromInt32(col, n), 0))
	}
	return mlx.Concatenate(1, parts...)
}

// rotary is ZImageAttention._apply_rotary_emb: x [1, L, H, D] turned by the
// angles [L, D/2, 2], each pair of its last axis a complex number; the result
// in x's dtype.
func rotary(x, freqs mlx.Array) mlx.Array {
	s := x.Shape()
	dt := x.Dtype()
	x5 := mlx.Reshape(x, s[0], s[1], s[2], s[3]/2, 2)
	f := mlx.Reshape(freqs, 1, s[1], 1, s[3]/2, 2)
	xp := mlx.Split(x5, 4, 1)
	fp := mlx.Split(f, 4, 1)
	xr, xi := mlx.Squeeze(xp[0], 4), mlx.Squeeze(xp[1], 4)
	c, sn := mlx.Squeeze(fp[0], 4), mlx.Squeeze(fp[1], 4)
	re := mlx.Subtract(mlx.Multiply(xr, c), mlx.Multiply(xi, sn))
	im := mlx.Add(mlx.Multiply(xr, sn), mlx.Multiply(xi, c))
	return mlx.Astype(mlx.Reshape(mlx.Stack(-1, re, im), s...), dt)
}

func (b *block) attention(x, mask, freqs mlx.Array) mlx.Array {
	s := x.Shape()
	L := s[1]
	q := mlx.RMSNorm(mlx.Reshape(b.q.apply(x), 1, L, ditHeads, ditHeadDim), b.normQ, 1e-5)
	k := mlx.RMSNorm(mlx.Reshape(b.k.apply(x), 1, L, ditHeads, ditHeadDim), b.normK, 1e-5)
	v := mlx.Reshape(b.v.apply(x), 1, L, ditHeads, ditHeadDim)
	q, k = rotary(q, freqs), rotary(k, freqs)
	o := mlx.SDPA(mlx.Transpose(q, 0, 2, 1, 3), mlx.Transpose(k, 0, 2, 1, 3), mlx.Transpose(v, 0, 2, 1, 3),
		float32(math.Pow(ditHeadDim, -0.5)), "", mask)
	return b.o.apply(mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), 1, L, ditDim))
}

func (b *block) feedForward(x mlx.Array) mlx.Array {
	return b.w2.apply(mlx.Multiply(mlx.SiLU(b.w1.apply(x)), b.w3.apply(x)))
}

// forward is ZImageTransformerBlock (modulated) or ZImageContextBlock.
func (b *block) forward(x, mask, freqs, tEmb mlx.Array) mlx.Array {
	if !b.modulated {
		x = mlx.Add(x, mlx.RMSNorm(b.attention(mlx.RMSNorm(x, b.an1, ditEps), mask, freqs), b.an2, ditEps))
		return mlx.Add(x, mlx.RMSNorm(b.feedForward(mlx.RMSNorm(x, b.fn1, ditEps)), b.fn2, ditEps))
	}
	mod := mlx.ExpandDims(b.ada.apply(tEmb), 1)
	p := mlx.Split(mod, 2, ditDim, 2*ditDim, 3*ditDim)
	scaleMSA, gateMSA := mlx.AddScalar(p[0], 1), mlx.Tanh(p[1])
	scaleMLP, gateMLP := mlx.AddScalar(p[2], 1), mlx.Tanh(p[3])
	a := b.attention(mlx.Multiply(mlx.RMSNorm(x, b.an1, ditEps), scaleMSA), mask, freqs)
	x = mlx.Add(x, mlx.Multiply(gateMSA, mlx.RMSNorm(a, b.an2, ditEps)))
	f := b.feedForward(mlx.Multiply(mlx.RMSNorm(x, b.fn1, ditEps), scaleMLP))
	return mlx.Add(x, mlx.Multiply(gateMLP, mlx.RMSNorm(f, b.fn2, ditEps)))
}

// padRows pads x [n, d] to a multiple of padTo by repeating its last row, and
// says how many it added.
func padRows(x mlx.Array) (mlx.Array, int) {
	s := x.Shape()
	pad := (padTo - s[0]%padTo) % padTo
	if pad == 0 {
		return x, 0
	}
	last := mlx.Slice(x, []int{s[0] - 1, 0}, []int{s[0], s[1]})
	return mlx.Concatenate(0, x, mlx.Repeat(last, pad, 0)), pad
}

// zeroMask is the additive mask of an all-true attention mask, as
// ZImageAttention makes one: [1, 1, 1, L] of zeros in dt.
func zeroMask(L int, dt mlx.Dtype) mlx.Array { return mlx.Zeros(dt, 1, 1, 1, L) }

// forward is ZImageTransformer.__call__: the velocity, negated, for latents x
// [16, 1, h, w] at timestep t [1] (1 - sigma) under the caption cap [n, 2560].
func (d *dit) forward(x, t, cap mlx.Array) mlx.Array {
	return d.forwardEmb(x, d.timeEmbedding(t), cap)
}

func (d *dit) forwardEmb(x, tEmb, cap mlx.Array) mlx.Array {
	s := x.Shape()
	C, H, W := s[0], s[2], s[3]
	Ht, Wt := H/patch, W/patch

	// The caption: padded, its positions (i+1, 0, 0).
	capOri := cap.Shape()[0]
	capP, capPad := padRows(cap)
	capLen := capOri + capPad
	capIDs := make([]int32, capLen*3)
	for i := 0; i < capLen; i++ {
		capIDs[i*3] = int32(i + 1)
	}
	// The picture: 2×2 patches of the 16 channels, [Ht·Wt, 64], positions
	// (capLen+1, row, column), its padding at (0, 0, 0).
	img := mlx.Reshape(x, C, 1, 1, Ht, patch, Wt, patch)
	img = mlx.Transpose(img, 1, 3, 5, 2, 4, 6, 0)
	img = mlx.Reshape(img, Ht*Wt, patch*patch*C)
	imgOri := Ht * Wt
	img, imgPad := padRows(img)
	imgLen := imgOri + imgPad
	imgIDs := make([]int32, imgLen*3)
	for i := 0; i < imgOri; i++ {
		imgIDs[i*3], imgIDs[i*3+1], imgIDs[i*3+2] = int32(capLen+1), int32(i/Wt), int32(i%Wt)
	}

	xe := d.xEmb.apply(img)
	if imgPad > 0 {
		xe = mlx.Where(padMask(imgOri, imgPad), d.xPad, xe)
	}
	xFreqs := d.ropeFor(imgIDs, imgLen)
	xe = mlx.ExpandDims(xe, 0)
	m := zeroMask(imgLen, xe.Dtype())
	for i := range d.noise {
		xe = d.noise[i].forward(xe, m, xFreqs, tEmb)
	}

	ce := d.capLin.apply(mlx.RMSNorm(capP, d.capNorm, ditEps))
	if capPad > 0 {
		ce = mlx.Where(padMask(capOri, capPad), d.capPad, ce)
	}
	capFreqs := d.ropeFor(capIDs, capLen)
	ce = mlx.ExpandDims(ce, 0)
	m = zeroMask(capLen, ce.Dtype())
	for i := range d.context {
		ce = d.context[i].forward(ce, m, capFreqs, 0)
	}

	u := mlx.Concatenate(1, xe, ce)
	freqs := mlx.Concatenate(0, xFreqs, capFreqs)
	m = zeroMask(imgLen+capLen, u.Dtype())
	for i := range d.layers {
		u = d.layers[i].forward(u, m, freqs, tEmb)
	}

	// FinalLayer, then the patches back into latents.
	scale := mlx.ExpandDims(mlx.AddScalar(d.finalAda.apply(mlx.SiLU(tEmb)), 1), 1)
	u = d.finalLin.apply(mlx.Multiply(mlx.LayerNorm(u, 0, 0, 1e-6), scale))
	u = mlx.Slice(u, []int{0, 0, 0}, []int{1, imgOri, patch * patch * C})
	u = mlx.Reshape(u, 1, Ht, Wt, 1, patch, patch, C)
	u = mlx.Transpose(u, 6, 0, 3, 1, 4, 2, 5)
	return mlx.Negative(mlx.Reshape(u, C, 1, H, W))
}

// padMask is [ori+pad, 1]: false for the first ori rows, true after.
func padMask(ori, pad int) mlx.Array {
	v := make([]float32, ori+pad)
	for i := ori; i < ori+pad; i++ {
		v[i] = 1
	}
	return mlx.Astype(mlx.FromFloat32(v, ori+pad, 1), mlx.Bool)
}
