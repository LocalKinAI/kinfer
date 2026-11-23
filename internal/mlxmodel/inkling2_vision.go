package mlxmodel

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// Inkling 2's eyes: a vision transformer whose output stands where a
// picture's tokens are in the prompt, one for every tile of 40 × 40 pixels.
//
// A picture is not made one size. It is cut, as it is, into patches of 20 × 20
// pixels; the tower reads them with a class token before them — positions
// from a 32 × 32 table stretched over the picture's own grid, and rotations by
// column and row inside each attention — and its patches, 2 × 2 at a time, go
// through two layers into the text model's width. The text model then reads
// those like any tokens, in order.
//
// This follows Ollama's reading of the checkpoint (mlxrunner/model/inkling2,
// vision.go), which is what the weights here are made from; it computes in
// the weights' own bfloat16, as that does.

// Inkling2Vision is config.json's vision_config.
type Inkling2Vision struct {
	Patch     int     `json:"native_patch_size"`
	Merge     int     `json:"adapter_merge_size"`
	Projector int     `json:"projection_hidden_dim"`
	Width     int     `json:"hidden_size"`
	Layers    int     `json:"num_hidden_layers"`
	Heads     int     `json:"num_attention_heads"`
	MLPRatio  float64 `json:"mlp_ratio"`
	Grid      int     `json:"posemb_grid_size"`
	Eps       float32 `json:"layer_norm_eps"`
	ProjBias  bool    `json:"projector_bias"`
}

type inkEyeLayer struct {
	norm, ffNorm         [2]mlx.Array // weight, bias
	q, k, v, o, up, down [2]mlx.Array // weight [out, in], bias
	attnScale, ffScale   mlx.Array
}

type inkEyes struct {
	cfg       Inkling2Vision
	patch     mlx.Array // [width, patch·patch·3]
	class     mlx.Array // [1, width]: the class token with its position
	positions mlx.Array // [grid·grid, width]
	pre       [2]mlx.Array
	layers    []inkEyeLayer
	project   mlx.Array // [projector, merge·merge·width]
	out       mlx.Array // [hidden, projector]
	all       []mlx.Array
}

// The pixels a picture is handed over in: CLIP's, which the tower turns back
// into −1…1 once they are in its own number format.
var (
	inkPixelMean = [3]float32{0.48145466, 0.4578275, 0.40821073}
	inkPixelStd  = [3]float32{0.26862954, 0.26130258, 0.27577711}
)

// InklingPixel is the number a channel's value v, 0 to 255 — or −1 for the
// padding beyond a picture's edge — is handed to EncodePicture as.
func InklingPixel(v int, channel int) float32 {
	return (float32(v)/255 - inkPixelMean[channel]) / inkPixelStd[channel]
}

// loadInkEyes loads the vision tower of the model in dir, or nil when the
// model has none.
func loadInkEyes(dir string, w *inkWeights, hidden int) (*inkEyes, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var top struct {
		Vision *Inkling2Vision `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("%s/config.json: %w", dir, err)
	}
	const prefix = "model.visual."
	if top.Vision == nil || !w.has(prefix+"proj_out.weight") {
		return nil, nil
	}
	c := *top.Vision
	if c.Patch <= 0 || c.Merge <= 0 || c.Projector <= 0 || c.Width <= 0 || c.Layers <= 0 || c.Heads <= 0 ||
		c.Width%c.Heads != 0 || (c.Width/c.Heads)%4 != 0 || c.MLPRatio <= 0 || c.Grid <= 0 || c.Eps <= 0 || c.ProjBias {
		return nil, fmt.Errorf("%s: an Inkling 2 vision tower of this kind is not read", dir)
	}
	e := &inkEyes{cfg: c}
	take := func(name string, shape ...int) mlx.Array {
		a := w.take(prefix + name)
		e.all = append(e.all, a)
		if got := a.Shape(); fmt.Sprint(got) != fmt.Sprint(shape) {
			panic(&mlx.Error{Msg: fmt.Sprintf("weight %s%s is %v, not %v", prefix, name, got, shape)})
		}
		return a
	}
	linear := func(stem string, out, in int) [2]mlx.Array {
		return [2]mlx.Array{take(stem+".weight", out, in), take(stem+".bias", out)}
	}
	norm := func(stem string) [2]mlx.Array {
		return [2]mlx.Array{take(stem+".weight", c.Width), take(stem+".bias", c.Width)}
	}
	emb := "backbone.embeddings."
	e.patch = take(emb+"patch_embedding.weight", c.Width, c.Patch*c.Patch*3)
	class, offset := take(emb+"class_embedding", c.Width), take(emb+"cls_position_embedding", c.Width)
	e.class = mlx.Reshape(mlx.Add(class, offset), 1, c.Width)
	e.all = append(e.all, e.class)
	e.positions = take(emb+"position_embedding", c.Grid*c.Grid, c.Width)
	e.pre = norm("backbone.ln_pre")
	e.project = take("proj_in.weight", c.Projector, c.Merge*c.Merge*c.Width)
	e.out = take("proj_out.weight", hidden, c.Projector)
	wide := int(float64(c.Width) * c.MLPRatio)
	e.layers = make([]inkEyeLayer, c.Layers)
	for i := range e.layers {
		p := fmt.Sprintf("backbone.layers.%d.", i)
		e.layers[i] = inkEyeLayer{
			norm: norm(p + "ln_1"), ffNorm: norm(p + "ln_2"),
			q: linear(p+"self_attn.q_proj", c.Width, c.Width), k: linear(p+"self_attn.k_proj", c.Width, c.Width),
			v: linear(p+"self_attn.v_proj", c.Width, c.Width), o: linear(p+"self_attn.out_proj", c.Width, c.Width),
			up: linear(p+"mlp.c_fc", wide, c.Width), down: linear(p+"mlp.c_proj", c.Width, wide),
			attnScale: take(p+"ls_1.gamma", c.Width), ffScale: take(p+"ls_2.gamma", c.Width),
		}
	}
	return e, nil
}

// ReadsPictures says whether the model has its vision tower.
func (m *Inkling2) ReadsPictures() bool { return m.eyes != nil }

// PictureTile is the side, in pixels, of the square of a picture that one
// token stands for.
func (m *Inkling2) PictureTile() int {
	if m.eyes == nil {
		return 0
	}
	return m.eyes.cfg.Patch * m.eyes.cfg.Merge
}

// EncodePicture reads a picture of rows × cols tiles — pixels is its
// rows·tile × cols·tile × 3 values, a row of pixels after another, each made
// by InklingPixel — and returns what the text model reads in the place of its
// tokens: [rows·cols, hidden], evaluated.
func (m *Inkling2) EncodePicture(pixels []float32, rows, cols int) (features mlx.Array, err error) {
	defer mlx.Recover(&err)
	e := m.eyes
	if e == nil {
		return 0, fmt.Errorf("this Inkling 2 has no vision tower")
	}
	c := e.cfg
	tile := c.Patch * c.Merge
	if rows <= 0 || cols <= 0 || len(pixels) != rows*tile*cols*tile*3 {
		return 0, fmt.Errorf("a picture of %d × %d tiles is not %d values", rows, cols, len(pixels))
	}
	scope := mlx.NewScope()
	defer scope.Close()
	pr, pc := rows*c.Merge, cols*c.Merge
	dt := e.patch.Dtype()

	data := mlx.Astype(mlx.FromFloat32(pixels, rows*tile, cols*tile, 3), dt)
	mean, std := mlx.Astype(mlx.FromFloat32(inkPixelMean[:], 3), dt), mlx.Astype(mlx.FromFloat32(inkPixelStd[:], 3), dt)
	data = mlx.AddScalar(mlx.MulScalar(mlx.Add(mlx.Multiply(data, std), mean), 2), -1)
	patches := mlx.Transpose(mlx.Reshape(data, pr, c.Patch, pc, c.Patch, 3), 0, 2, 1, 3, 4)
	h := mlx.Matmul(mlx.Reshape(patches, pr*pc, c.Patch*c.Patch*3), mlx.T(e.patch))
	h = mlx.Add(h, e.stretched(pr, pc))
	h = mlx.Concatenate(0, e.class, h)
	h = mlx.LayerNorm(h, e.pre[0], e.pre[1], c.Eps)
	cos, sin := inkEyePhases(pr, pc, c.Width/c.Heads)
	for i := range e.layers {
		h = e.layers[i].forward(h, cos, sin, c.Heads, c.Eps)
	}
	n := pr*pc + 1
	out := mlx.Slice(h, []int{1, 0}, []int{n, c.Width})
	out = mlx.Transpose(mlx.Reshape(out, rows, c.Merge, cols, c.Merge, c.Width), 0, 2, 1, 3, 4)
	out = mlx.Reshape(out, rows*cols, c.Merge*c.Merge*c.Width)
	out = mlx.Matmul(mlx.GELU(mlx.Matmul(out, mlx.T(e.project))), mlx.T(e.out))
	out = mlx.Astype(out, mlx.Bfloat16)
	mlx.Eval(out)
	scope.Keep(out)
	return out, nil
}

// stretched is the table of positions laid over a grid of rows × cols
// patches, each patch reading the four entries around it: [rows·cols, width].
func (e *inkEyes) stretched(rows, cols int) mlx.Array {
	size := e.cfg.Grid
	idx := make([]int32, 0, 4*rows*cols)
	weights := make([]float32, 0, 4*rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			y := min(float64(size-1), max(0, (float64(r)+0.5)*float64(size)/float64(rows)-0.5))
			x := min(float64(size-1), max(0, (float64(c)+0.5)*float64(size)/float64(cols)-0.5))
			y0, x0 := int(y), int(x)
			y1, x1 := min(y0+1, size-1), min(x0+1, size-1)
			dy, dx := float32(y-float64(y0)), float32(x-float64(x0))
			idx = append(idx, int32(y0*size+x0), int32(y0*size+x1), int32(y1*size+x0), int32(y1*size+x1))
			weights = append(weights, (1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx)
		}
	}
	four := mlx.Astype(mlx.Take(e.positions, mlx.FromInt32(idx, rows*cols, 4), 0), mlx.Float32)
	sum := mlx.Sum(mlx.Multiply(four, mlx.FromFloat32(weights, rows*cols, 4, 1)), 1, false)
	return mlx.Astype(sum, e.positions.Dtype())
}

// inkEyePhases is the rotations a patch's queries and keys take, by its
// column and its row, counted from 1 — the class token, at 0, takes none:
// cosines and sines, [n, 1, headDim/2].
func inkEyePhases(rows, cols, headDim int) (cos, sin mlx.Array) {
	n := rows*cols + 1
	quarter := headDim / 4
	angles := make([]float32, n*2*quarter)
	for i := 0; i < n-1; i++ {
		x, y := float64(i%cols+1), float64(i/cols+1)
		row := angles[(i+1)*2*quarter:]
		for f := 0; f < quarter; f++ {
			freq := math.Pow(10000, -4*float64(f)/float64(headDim))
			row[f], row[quarter+f] = float32(x)*float32(freq), float32(y)*float32(freq)
		}
	}
	a := mlx.FromFloat32(angles, n, 1, 2*quarter)
	return mlx.Cos(a), mlx.Sin(a)
}

// inkEyeTurn rotates x [n, heads, dim], its numbers in pairs.
func inkEyeTurn(x, cos, sin mlx.Array) mlx.Array {
	s := x.Shape()
	n, heads, dim := s[0], s[1], s[2]
	pairs := mlx.Reshape(mlx.Astype(x, mlx.Float32), n, heads, dim/2, 2)
	even := mlx.Reshape(mlx.Slice(pairs, []int{0, 0, 0, 0}, []int{n, heads, dim / 2, 1}), n, heads, dim/2)
	odd := mlx.Reshape(mlx.Slice(pairs, []int{0, 0, 0, 1}, []int{n, heads, dim / 2, 2}), n, heads, dim/2)
	re := mlx.Subtract(mlx.Multiply(even, cos), mlx.Multiply(odd, sin))
	im := mlx.Add(mlx.Multiply(even, sin), mlx.Multiply(odd, cos))
	return mlx.Astype(mlx.Reshape(mlx.Stack(-1, re, im), n, heads, dim), x.Dtype())
}

func inkEyeLinear(x mlx.Array, l [2]mlx.Array) mlx.Array {
	return mlx.Add(mlx.Matmul(x, mlx.T(l[0])), l[1])
}

func (l *inkEyeLayer) forward(h, cos, sin mlx.Array, heads int, eps float32) mlx.Array {
	s := h.Shape()
	n, width := s[0], s[1]
	dim := width / heads
	x := mlx.LayerNorm(h, l.norm[0], l.norm[1], eps)
	split := func(a mlx.Array) mlx.Array { return mlx.Reshape(a, n, heads, dim) }
	rows := func(a mlx.Array) mlx.Array { return mlx.ExpandDims(mlx.Transpose(a, 1, 0, 2), 0) }
	q := rows(inkEyeTurn(split(inkEyeLinear(x, l.q)), cos, sin))
	k := rows(inkEyeTurn(split(inkEyeLinear(x, l.k)), cos, sin))
	v := rows(split(inkEyeLinear(x, l.v)))
	a := mlx.SDPA(q, k, v, float32(1/math.Sqrt(float64(dim))), "", 0)
	a = mlx.Reshape(mlx.Transpose(mlx.Reshape(a, heads, n, dim), 1, 0, 2), n, width)
	h = mlx.Add(h, mlx.Multiply(inkEyeLinear(a, l.o), l.attnScale))
	f := inkEyeLinear(mlx.GELU(inkEyeLinear(mlx.LayerNorm(h, l.ffNorm[0], l.ffNorm[1], eps), l.up)), l.down)
	return mlx.Add(h, mlx.Multiply(f, l.ffScale))
}

// Inkling2Span is what a picture's tokens are read as: features [n, hidden]
// for the n tokens of a pass from At on.
type Inkling2Span struct {
	At       int
	Features mlx.Array
}

// ForwardSpans is Forward for one sequence whose tokens include pictures':
// each span's features stand where the tokens' own embeddings would.
func (m *Inkling2) ForwardSpans(tokens []int32, spans []Inkling2Span, cache *Inkling2Cache) (logits mlx.Array, err error) {
	defer mlx.Recover(&err)
	for _, s := range spans {
		if n := s.Features.Shape()[0]; s.At < 0 || s.At+n > len(tokens) {
			return 0, fmt.Errorf("a picture's %d tokens at %d are not within a pass of %d", n, s.At, len(tokens))
		}
	}
	ids := mlx.FromInt32(tokens, 1, len(tokens))
	m.spans = spans
	logits = m.Step(ids, cache)
	m.spans = nil
	ids.Free()
	mlx.Eval(append([]mlx.Array{logits}, cache.Arrays()...)...)
	return logits, nil
}

// placed is a pass's embeddings [1, L, hidden] with the pictures' features in
// their places.
func (m *Inkling2) placed(emb mlx.Array) mlx.Array {
	for _, s := range m.spans {
		n := s.Features.Shape()[0]
		f := mlx.Reshape(mlx.Astype(s.Features, emb.Dtype()), 1, n, m.hidden)
		emb = mlx.SliceUpdate(emb, f, []int{0, s.At, 0}, []int{1, s.At + n, m.hidden})
	}
	return emb
}
