package mlxembed

import (
	"fmt"
	"math"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// Pictures, as EmbeddingGemma 2 sees them: resized, keeping their shape, to
// the most 48-pixel cells 280 soft tokens allow (image.go), cut into 16-pixel
// patches, read by Gemma 4's vision tower — a bidirectional transformer with
// a position on each axis, its rotary embedding split between the two — and
// every 3×3 patches averaged into one soft token, projected into the text's
// space. Ported from Ollama (mlxrunner/model/gemma4), as audio.go is.

// visionConfig is config.json's vision_config.
type visionConfig struct {
	ModelType  string  `json:"model_type"`
	Hidden     int     `json:"hidden_size"`
	Layers     int     `json:"num_hidden_layers"`
	Heads      int     `json:"num_attention_heads"`
	HeadDim    int     `json:"head_dim"`
	Patch      int     `json:"patch_size"`
	Pool       int     `json:"pooling_kernel_size"`
	Eps        float32 `json:"rms_norm_eps"`
	Standarize bool    `json:"standardize"`
	Rope       struct {
		Theta float32 `json:"rope_theta"`
	} `json:"rope_parameters"`
}

type visionTower struct {
	cfg       visionConfig
	inputProj tlinear
	posTable  mlx.Array // [2, positions, hidden]: x and y
	positions int
	layers    []visionLayer
	stdBias   mlx.Array
	stdScale  mlx.Array
	proj      tlinear // into the text's space
	keep      []mlx.Array
}

type visionLayer struct {
	in, postAttn, preFF, postFF mlx.Array
	q, k, v, o                  tlinear
	qNorm, kNorm                mlx.Array
	gate, up, down              tlinear
}

func loadVision(t map[string]mlx.Array, cfg visionConfig) (*visionTower, error) {
	if cfg.HeadDim == 0 && cfg.Heads > 0 {
		cfg.HeadDim = cfg.Hidden / cfg.Heads
	}
	if cfg.Patch == 0 {
		cfg.Patch = 16
	}
	if cfg.Pool == 0 {
		cfg.Pool = 3
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-6
	}
	if cfg.Rope.Theta == 0 {
		cfg.Rope.Theta = 100
	}
	w := &towerWeights{t: t}
	vt := &visionTower{cfg: cfg,
		inputProj: w.linear("vision_tower.patch_embedder.input_proj"),
		posTable:  w.arr("vision_tower.patch_embedder.position_embedding_table"),
		proj:      w.linear("embed_vision.embedding_projection")}
	for i := 0; i < cfg.Layers; i++ {
		p := fmt.Sprintf("vision_tower.encoder.layers.%d.", i)
		vt.layers = append(vt.layers, visionLayer{
			in: w.arr(p + "input_layernorm.weight"), postAttn: w.arr(p + "post_attention_layernorm.weight"),
			preFF: w.arr(p + "pre_feedforward_layernorm.weight"), postFF: w.arr(p + "post_feedforward_layernorm.weight"),
			q: w.linear(p + "self_attn.q_proj"), k: w.linear(p + "self_attn.k_proj"),
			v: w.linear(p + "self_attn.v_proj"), o: w.linear(p + "self_attn.o_proj"),
			qNorm: w.arr(p + "self_attn.q_norm.weight"), kNorm: w.arr(p + "self_attn.k_norm.weight"),
			gate: w.linear(p + "mlp.gate_proj"), up: w.linear(p + "mlp.up_proj"), down: w.linear(p + "mlp.down_proj"),
		})
	}
	if cfg.Standarize {
		vt.stdBias, vt.stdScale = w.arr("vision_tower.std_bias"), w.arr("vision_tower.std_scale")
	}
	if w.missing != "" {
		return nil, fmt.Errorf("no tensor %s", w.missing)
	}
	if sh := vt.posTable.Shape(); len(sh) == 3 && sh[0] == 2 {
		vt.positions = sh[1]
	} else {
		return nil, fmt.Errorf("the position table is %v, want [2, positions, %d]", sh, cfg.Hidden)
	}
	return vt, nil
}

func (vt *visionTower) free() {
	for _, a := range vt.keep {
		a.Free()
	}
	vt.keep = nil
}

// visionInput is one picture prepared for the tower: its patches' pixels in
// [0, 1], [patches, patch·patch·3], each patch's column and row, and the grid.
type visionInput struct {
	pixels         []float32
	positions      []int32 // x0, y0, x1, y1, …
	patchesW       int
	patchesH       int
	softTokens     int
	pixelsPerPatch int
}

// encode runs the tower over one picture: [soft tokens, text hidden].
func (vt *visionTower) encode(in visionInput) mlx.Array {
	c := vt.cfg
	n := in.patchesW * in.patchesH
	// The processor scales to [0, 1]; the reference folds [-1, 1] into the
	// patch embedder, in f32 before the cast.
	x := mlx.FromFloat32(in.pixels, n, in.pixelsPerPatch)
	x = mlx.Astype(mlx.MulScalar(mlx.AddScalar(x, -0.5), 2), mlx.Bfloat16)
	h := vt.inputProj.apply(x)

	xs, ys := make([]int32, n), make([]int32, n)
	for i := 0; i < n; i++ {
		xs[i], ys[i] = in.positions[2*i], in.positions[2*i+1]
	}
	for axis, pos := range [][]int32{xs, ys} {
		table := mlx.Squeeze(mlx.Slice(vt.posTable, []int{axis, 0, 0}, []int{axis + 1, vt.positions, c.Hidden}), 0)
		h = mlx.Add(h, mlx.Astype(mlx.Take(table, mlx.FromInt32(pos, n), 0), h.Dtype()))
	}
	h = mlx.Reshape(h, 1, n, c.Hidden)

	cosX, sinX := visionRope(xs, c)
	cosY, sinY := visionRope(ys, c)
	for i := range vt.layers {
		l := &vt.layers[i]
		a := l.attend(mlx.RMSNorm(h, l.in, c.Eps), cosX, sinX, cosY, sinY, n, c)
		h = mlx.Add(h, mlx.RMSNorm(a, l.postAttn, c.Eps))
		f := mlx.RMSNorm(h, l.preFF, c.Eps)
		f = l.down.apply(mlx.Multiply(geluTanh(l.gate.apply(f)), l.up.apply(f)))
		h = mlx.Add(h, mlx.RMSNorm(f, l.postFF, c.Eps))
	}

	// Each pool×pool cell of patches averaged, scaled by √hidden, and
	// standardized, in f32 with the reference's round trip through bfloat16.
	p := c.Pool
	f := mlx.Reshape(mlx.Astype(h, mlx.Float32), in.patchesH/p, p, in.patchesW/p, p, c.Hidden)
	f = mlx.Mean(mlx.Mean(f, 3, false), 1, false)
	f = mlx.Reshape(f, n/(p*p), c.Hidden)
	f = mlx.Astype(mlx.Astype(f, mlx.Bfloat16), mlx.Float32)
	f = mlx.MulScalar(f, float32(math.Sqrt(float64(c.Hidden))))
	if vt.stdBias != 0 {
		f = mlx.Multiply(mlx.Subtract(f, mlx.Astype(vt.stdBias, mlx.Float32)), mlx.Astype(vt.stdScale, mlx.Float32))
	}
	f = mlx.RMSNorm(mlx.Astype(f, mlx.Bfloat16), 0, c.Eps)
	return vt.proj.apply(f)
}

// visionRope is one axis's rotation tables, [1, n, 1, headDim/2], shared by
// every layer.
func visionRope(pos []int32, c visionConfig) (cos, sin mlx.Array) {
	n := len(pos)
	spatial := c.HeadDim / 2
	nFreq := spatial / 2
	p := make([]float32, n)
	for i, v := range pos {
		p[i] = float32(v)
	}
	inv := make([]float32, nFreq)
	for i := range inv {
		inv[i] = float32(1 / math.Pow(float64(c.Rope.Theta), float64(2*i)/float64(spatial)))
	}
	f := mlx.Matmul(mlx.FromFloat32(p, n, 1), mlx.FromFloat32(inv, 1, nFreq))
	emb := mlx.Concatenate(1, f, f)
	cos = mlx.Astype(mlx.Reshape(mlx.Cos(emb), 1, n, 1, spatial), mlx.Bfloat16)
	sin = mlx.Astype(mlx.Reshape(mlx.Sin(emb), 1, n, 1, spatial), mlx.Bfloat16)
	return cos, sin
}

// rotate2D turns the first half of the head dim by the x tables and the
// second by the y tables, each half rotated within itself.
func rotate2D(t, cosX, sinX, cosY, sinY mlx.Array, hd int) mlx.Array {
	half := hd / 2
	sh := t.Shape()
	part := func(lo, hi int) mlx.Array {
		return mlx.Slice(t, []int{0, 0, 0, lo}, []int{sh[0], sh[1], sh[2], hi})
	}
	rot := func(p mlx.Array) mlx.Array {
		q := half / 2
		ps := p.Shape()
		p1 := mlx.Slice(p, []int{0, 0, 0, 0}, []int{ps[0], ps[1], ps[2], q})
		p2 := mlx.Slice(p, []int{0, 0, 0, q}, []int{ps[0], ps[1], ps[2], half})
		return mlx.Concatenate(3, mlx.Negative(p2), p1)
	}
	tx, ty := part(0, half), part(half, hd)
	tx = mlx.Add(mlx.Multiply(tx, cosX), mlx.Multiply(rot(tx), sinX))
	ty = mlx.Add(mlx.Multiply(ty, cosY), mlx.Multiply(rot(ty), sinY))
	return mlx.Concatenate(3, tx, ty)
}

func (l *visionLayer) attend(x, cosX, sinX, cosY, sinY mlx.Array, n int, c visionConfig) mlx.Array {
	heads := func(p tlinear) mlx.Array { return mlx.Reshape(p.apply(x), 1, n, c.Heads, c.HeadDim) }
	q := mlx.RMSNorm(heads(l.q), l.qNorm, c.Eps)
	k := mlx.RMSNorm(heads(l.k), l.kNorm, c.Eps)
	v := mlx.RMSNorm(heads(l.v), 0, c.Eps)
	q = rotate2D(q, cosX, sinX, cosY, sinY, c.HeadDim)
	k = rotate2D(k, cosX, sinX, cosY, sinY, c.HeadDim)
	q, k, v = mlx.Transpose(q, 0, 2, 1, 3), mlx.Transpose(k, 0, 2, 1, 3), mlx.Transpose(v, 0, 2, 1, 3)
	// Both ways, and scale 1: the q and k norms keep the scores' size.
	o := mlx.SDPA(q, k, v, 1, "", 0)
	return l.o.apply(mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), 1, n, c.Heads*c.HeadDim))
}
