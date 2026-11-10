package mlxtts

import (
	"fmt"
	"math"
	"sync"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// The talker and its code predictor are Qwen3 stacks (mlx-audio's talker.py):
// RMSNorm, attention with RMS-normed queries and keys and rotary positions, a
// SwiGLU MLP. The talker's rotary embedding is multimodal, but the three
// positions it is given are always the same one, which makes it the plain one.
// The rotation is done as mlx-audio does it — cos and sin computed in float32,
// cast to the activations' type, q·cos + rotate_half(q)·sin — rather than
// with the fused kernel, to compute what it computes.

// ropeTable is how many positions the rotary tables hold: a prompt and the
// 4,096 frames mlx-audio allows, with room.
const ropeTable = 8192

type layer struct {
	inNorm, postNorm, qNorm, kNorm mlx.Array // qNorm, kNorm 0 when there are none
	qkv, o, gateUp, down           linear    // q, k, v and gate, up side by side: one product each
}

type stack struct {
	cfg      stackConfig
	layers   []layer
	norm     mlx.Array
	cos, sin mlx.Array // [ropeTable, HeadDim]
	dt       mlx.Dtype // the activations'
	ropes    map[int]*mlx.Launch
	swiglus  map[int]*mlx.Launch
	made     []mlx.Array
}

func loadStack(w *weights, prefix string, cfg stackConfig, qkNorm bool) (stack, error) {
	s := stack{cfg: cfg, layers: make([]layer, cfg.Layers), ropes: map[int]*mlx.Launch{}, swiglus: map[int]*mlx.Launch{}}
	var err error
	if s.norm, err = w.get(prefix + "norm.weight"); err != nil {
		return s, err
	}
	for i := range s.layers {
		l := &s.layers[i]
		p := fmt.Sprintf("%slayers.%d.", prefix, i)
		norms := []struct {
			dst  *mlx.Array
			name string
		}{{&l.inNorm, "input_layernorm.weight"}, {&l.postNorm, "post_attention_layernorm.weight"}}
		if qkNorm {
			norms = append(norms, struct {
				dst  *mlx.Array
				name string
			}{&l.qNorm, "self_attn.q_norm.weight"}, struct {
				dst  *mlx.Array
				name string
			}{&l.kNorm, "self_attn.k_norm.weight"})
		}
		for _, x := range norms {
			if *x.dst, err = w.get(p + x.name); err != nil {
				return s, err
			}
		}
		var q, k, v, gate, up linear
		for _, x := range []struct {
			dst  *linear
			name string
		}{
			{&q, "self_attn.q_proj"}, {&k, "self_attn.k_proj"}, {&v, "self_attn.v_proj"}, {&l.o, "self_attn.o_proj"},
			{&gate, "mlp.gate_proj"}, {&up, "mlp.up_proj"}, {&l.down, "mlp.down_proj"},
		} {
			if *x.dst, err = w.lin(p + x.name); err != nil {
				return s, err
			}
		}
		l.qkv = w.joined(p+"self_attn.", []string{"q_proj", "k_proj", "v_proj"}, q, k, v)
		l.gateUp = w.joined(p+"mlp.", []string{"gate_proj", "up_proj"}, gate, up)
		s.made = append(s.made, l.qkv.arrays()...)
		s.made = append(s.made, l.gateUp.arrays()...)
	}
	return s, nil
}

// joined is linear layers side by side, their outputs one after another:
// one product where there were several. Each output is its own row's sum
// over the input, as it was, so nothing it computes changes. The parts'
// weights are let go — the model keeps the joined ones.
func (w *weights) joined(prefix string, names []string, parts ...linear) linear {
	l := parts[0]
	cat := func(get func(linear) mlx.Array) mlx.Array {
		var a []mlx.Array
		for _, p := range parts {
			a = append(a, get(p))
		}
		r := mlx.Concatenate(0, a...)
		mlx.Eval(r)
		return r
	}
	l.W = cat(func(p linear) mlx.Array { return p.W })
	if l.Scales != 0 {
		l.Scales = cat(func(p linear) mlx.Array { return p.Scales })
		l.Biases = cat(func(p linear) mlx.Array { return p.Biases })
	}
	if l.Bias != 0 {
		l.Bias = cat(func(p linear) mlx.Array { return p.Bias })
	}
	for _, n := range names {
		for _, suffix := range []string{".weight", ".scales", ".biases", ".bias"} {
			delete(w.taken, prefix+n+suffix)
		}
	}
	return l
}

// arrays are l's own arrays.
func (l linear) arrays() []mlx.Array {
	var r []mlx.Array
	for _, a := range []mlx.Array{l.W, l.Scales, l.Biases, l.Bias} {
		if a != 0 {
			r = append(r, a)
		}
	}
	return r
}

// swigluSource is silu(gate)·up for gu [L, 2F], the gate then the up side
// by side: mlx's Sigmoid as it is written, then the two products, each
// rounded to T as the separate ops round — what mlx-audio's compiled swiglu
// computes.
const swigluSource = `
	uint i = thread_position_in_grid.x;
	uint l = i / F;
	uint f = i % F;
	T g = gu[l * 2 * F + f];
	T u = gu[l * 2 * F + F + f];
	auto y = 1 / (1 + metal::precise::exp(metal::abs(g)));
	T s = (g < 0) ? y : 1 - y;
	T a = g * s;
	out[i] = a * u;
`

var (
	swigluOnce   sync.Once
	swigluKernel *mlx.Kernel
)

// swiglu is silu(gate)·up of gu [1, L, 2F]: [1, L, F].
func (s *stack) swiglu(gu mlx.Array, L int) mlx.Array {
	swigluOnce.Do(func() {
		swigluKernel = mlx.NewKernel("kinfer_tts_swiglu", []string{"gu"}, []string{"out"}, swigluSource)
	})
	F := s.cfg.FFN
	l, ok := s.swiglus[L]
	if !ok {
		n := L * F
		l = swigluKernel.Prepare([]mlx.KernelOutput{{Shape: []int{1, L, F}, Dtype: s.dt}},
			[]mlx.KernelTemplate{{Name: "T", Dtype: s.dt}, {Name: "F", Int: F, IsInt: true}},
			[3]int{n, 1, 1}, [3]int{min(256, n), 1, 1})
		if L <= 2 {
			s.swiglus[L] = l
		} else {
			defer l.Free()
		}
	}
	return l.Run(gu)[0]
}

// ropeTables are cos and sin for positions [0, n) as mlx-audio's rotary
// embeddings compute them, cast to dt: inv_freq = 1 / base^(2i/d) in float32,
// times the position, the half repeated.
func ropeTables(headDim int, base float32, n int, dt mlx.Dtype) (cos, sin mlx.Array) {
	sc := mlx.NewScope()
	defer sc.Close()
	exps := mlx.Divide(mlx.Arange(0, float64(headDim), 2, mlx.Float32), mlx.ScalarAs(float32(headDim), mlx.Float32))
	inv := mlx.Divide(mlx.ScalarAs(1, mlx.Float32), mlx.Power(mlx.ScalarAs(base, mlx.Float32), exps))
	pos := mlx.Arange(0, float64(n), 1, mlx.Float32)
	freqs := mlx.Multiply(mlx.Reshape(pos, n, 1), mlx.Reshape(inv, 1, headDim/2))
	emb := mlx.Concatenate(1, freqs, freqs)
	cos, sin = mlx.Astype(mlx.Cos(emb), dt), mlx.Astype(mlx.Sin(emb), dt)
	mlx.Eval(cos, sin)
	sc.Escape(cos, sin)
	return cos, sin
}

// ropeSource is x·cos + rotate_half(x)·sin for the queries and the keys in
// one kernel, each product and the sum rounded to T as the separate ops round
// them — what mlx-audio's compiled apply_rotary_pos_emb computes — for q
// [L, HQ, D] and k [L, HK, D] (before the heads are moved in front: the
// rotation is elementwise, so it commutes with that) at positions offset + l.
const ropeSource = `
	uint i = thread_position_in_grid.x;
	uint nq = L * HQ * D;
	bool isq = i < nq;
	uint j = isq ? i : i - nq;
	uint d = j % D;
	uint l = j / (D * (isq ? HQ : HK));
	uint at = (offset + l) * D + d;
	T x = isq ? q[j] : k[j];
	T o = isq ? (d < D / 2 ? q[j + D / 2] : q[j - D / 2]) : (d < D / 2 ? k[j + D / 2] : k[j - D / 2]);
	T r = d < D / 2 ? T(-float(o)) : o;
	T a = T(float(x) * float(cos[at]));
	T b = T(float(r) * float(sin[at]));
	T y = T(float(a) + float(b));
	if (isq) {
		yq[j] = y;
	} else {
		yk[j] = y;
	}
`

var (
	ropeOnce   sync.Once
	ropeKernel *mlx.Kernel
)

// rotate turns q [1, L, H, D] and k [1, L, Hkv, D] by their positions
// [offset, offset+L). The launch for one and two positions — a frame's
// steps — is set up once.
func (s *stack) rotate(q, k mlx.Array, L, offset int) (mlx.Array, mlx.Array) {
	ropeOnce.Do(func() {
		ropeKernel = mlx.NewKernel("kinfer_tts_rope", []string{"q", "k", "cos", "sin", "offset"}, []string{"yq", "yk"}, ropeSource)
	})
	cfg := s.cfg
	l, ok := s.ropes[L]
	if !ok {
		n := L * (cfg.Heads + cfg.KVHeads) * cfg.HeadDim
		l = ropeKernel.Prepare(
			[]mlx.KernelOutput{{Shape: []int{1, L, cfg.Heads, cfg.HeadDim}, Dtype: s.dt}, {Shape: []int{1, L, cfg.KVHeads, cfg.HeadDim}, Dtype: s.dt}},
			[]mlx.KernelTemplate{{Name: "T", Dtype: s.dt}, {Name: "D", Int: cfg.HeadDim, IsInt: true}, {Name: "HQ", Int: cfg.Heads, IsInt: true},
				{Name: "HK", Int: cfg.KVHeads, IsInt: true}, {Name: "L", Int: L, IsInt: true}},
			[3]int{n, 1, 1}, [3]int{min(256, n), 1, 1})
		if L <= 2 {
			s.ropes[L] = l
		} else {
			defer l.Free()
		}
	}
	out := l.Run(q, k, s.cos, s.sin, mlx.Int(offset))
	return out[0], out[1]
}

// kvCache is one sequence's keys and values, [1, Hkv, capacity, D] a layer,
// grown 256 positions at a time as mlx_lm's KVCache is.
type kvCache struct {
	keys, vals []mlx.Array
	caps       []int // each layer's buffers' capacity
	offset     int
	step       int // positions the buffers grow by
}

const kvStep = 256

func newCache(layers int) *kvCache {
	return &kvCache{keys: make([]mlx.Array, layers), vals: make([]mlx.Array, layers), caps: make([]int, layers), step: kvStep}
}

func (c *kvCache) arrays() []mlx.Array {
	var r []mlx.Array
	for _, set := range [][]mlx.Array{c.keys, c.vals} {
		for _, a := range set {
			if a != 0 {
				r = append(r, a)
			}
		}
	}
	return r
}

func (c *kvCache) free() {
	for _, a := range c.arrays() {
		a.Free()
	}
	c.keys, c.vals = nil, nil
}

// forward reads h [1, L, hidden] at the cache's offset and returns the
// normed hidden states, [1, L, hidden]; the cache's arrays are replaced by
// lazy ones and its offset moved on. The arrays made belong to the caller's
// scope, the cache's too: a caller that keeps the cache past its scope
// escapes them and frees the old ones.
func (s *stack) forward(h mlx.Array, L int, c *kvCache) mlx.Array {
	cfg := s.cfg
	for i := range s.layers {
		l := &s.layers[i]
		x := mlx.RMSNorm(h, l.inNorm, cfg.Eps)
		h = mlx.Add(h, s.attention(l, x, L, c, i))
		x = mlx.RMSNorm(h, l.postNorm, cfg.Eps)
		h = mlx.Add(h, l.down.apply(s.swiglu(l.gateUp.apply(x), L)))
	}
	c.offset += L
	return mlx.RMSNorm(h, s.norm, cfg.Eps)
}

func (s *stack) attention(l *layer, x mlx.Array, L int, c *kvCache, i int) mlx.Array {
	cfg := s.cfg
	H, Hkv, D := cfg.Heads, cfg.KVHeads, cfg.HeadDim
	qkv := l.qkv.apply(x)
	q := mlx.Reshape(mlx.Slice(qkv, []int{0, 0, 0}, []int{1, L, H * D}), 1, L, H, D)
	k := mlx.Reshape(mlx.Slice(qkv, []int{0, 0, H * D}, []int{1, L, (H + Hkv) * D}), 1, L, Hkv, D)
	v := mlx.Reshape(mlx.Slice(qkv, []int{0, 0, (H + Hkv) * D}, []int{1, L, (H + 2*Hkv) * D}), 1, L, Hkv, D)
	if l.qNorm != 0 {
		q, k = mlx.RMSNorm(q, l.qNorm, cfg.Eps), mlx.RMSNorm(k, l.kNorm, cfg.Eps)
	}
	q, k = s.rotate(q, k, L, c.offset)
	q, k, v = mlx.Transpose(q, 0, 2, 1, 3), mlx.Transpose(k, 0, 2, 1, 3), mlx.Transpose(v, 0, 2, 1, 3)
	k, v = c.update(i, k, v, L, Hkv, D, s.dt)
	return l.o.apply(mlx.Reshape(mlx.Transpose(attend(q, k, v, L, D), 0, 2, 1, 3), 1, L, H*D))
}

// update writes k and v [1, Hkv, L, D] (of dtype dt) into layer i's buffers
// at the offset and returns the keys and values so far.
func (c *kvCache) update(i int, k, v mlx.Array, L, Hkv, D int, dt mlx.Dtype) (mlx.Array, mlx.Array) {
	prev, end := c.offset, c.offset+L
	if c.caps[i] < end {
		n := (end + c.step - 1) / c.step * c.step
		for _, buf := range []*mlx.Array{&c.keys[i], &c.vals[i]} {
			grown := mlx.Zeros(dt, 1, Hkv, n, D)
			if *buf != 0 && prev > 0 {
				grown = mlx.SliceUpdate(grown, mlx.Slice(*buf, []int{0, 0, 0, 0}, []int{1, Hkv, prev, D}),
					[]int{0, 0, 0, 0}, []int{1, Hkv, prev, D})
			}
			*buf = grown
		}
		c.caps[i] = n
	}
	c.keys[i] = mlx.SliceUpdate(c.keys[i], k, []int{0, 0, prev, 0}, []int{1, Hkv, end, D})
	c.vals[i] = mlx.SliceUpdate(c.vals[i], v, []int{0, 0, prev, 0}, []int{1, Hkv, end, D})
	return mlx.Slice(c.keys[i], []int{0, 0, 0, 0}, []int{1, Hkv, end, D}),
		mlx.Slice(c.vals[i], []int{0, 0, 0, 0}, []int{1, Hkv, end, D})
}

// attend is fused attention over the last L positions, causal when there
// is more than one.
func attend(q, k, v mlx.Array, L, D int) mlx.Array {
	mode := ""
	if L > 1 {
		mode = "causal"
	}
	return mlx.SDPA(q, k, v, float32(math.Pow(float64(D), -0.5)), mode, 0)
}

// talker is the Qwen3-TTS talker with its text projection, codec head and
// code predictor.
type talker struct {
	cfg        *config
	main       stack
	codecEmbed mlx.Array // [vocab, hidden]
	textEmbed  *rowTable // [text vocab, text hidden], read from the file
	fc1, fc2   linear    // the text projection, text hidden → hidden
	head       linear    // the first code's logits
	cp         stack
	cpProj     linear      // talker hidden → predictor hidden, when they differ
	cpEmbed    []mlx.Array // codes 1..15's embeddings, [2048, talker hidden]
	cpHeads    []linear    // codes 1..15's logits
	made       []mlx.Array
}

func loadTalker(w *weights, cfg *config, dir string) (*talker, error) {
	tc := cfg.Talker
	t := &talker{cfg: cfg}
	var err error
	if t.main, err = loadStack(w, "talker.model.", tc.stackConfig, true); err != nil {
		return nil, err
	}
	if t.codecEmbed, err = w.get("talker.model.codec_embedding.weight"); err != nil {
		return nil, err
	}
	if t.textEmbed, err = openRows(dir, "talker.model.text_embedding.weight"); err != nil {
		return nil, err
	}
	for _, x := range []struct {
		dst  *linear
		name string
	}{{&t.fc1, "talker.text_projection.linear_fc1"}, {&t.fc2, "talker.text_projection.linear_fc2"}, {&t.head, "talker.codec_head"}} {
		if *x.dst, err = w.lin(x.name); err != nil {
			return nil, err
		}
	}
	pc := tc.CodePredictor
	if t.cp, err = loadStack(w, "talker.code_predictor.model.", pc.stackConfig, true); err != nil {
		return nil, err
	}
	if pc.Hidden != tc.Hidden {
		if t.cpProj, err = w.lin("talker.code_predictor.small_to_mtp_projection"); err != nil {
			return nil, err
		}
	}
	for i := 0; i < pc.Groups-1; i++ {
		e, err := w.get(fmt.Sprintf("talker.code_predictor.model.codec_embedding.%d.weight", i))
		if err != nil {
			return nil, err
		}
		h, err := w.lin(fmt.Sprintf("talker.code_predictor.lm_head.%d", i))
		if err != nil {
			return nil, err
		}
		t.cpEmbed, t.cpHeads = append(t.cpEmbed, e), append(t.cpHeads, h)
	}
	dt := t.codecEmbed.Dtype()
	t.main.dt, t.cp.dt = dt, dt
	t.main.cos, t.main.sin = ropeTables(tc.HeadDim, tc.RopeTheta, ropeTable, dt)
	t.cp.cos, t.cp.sin = ropeTables(pc.HeadDim, pc.RopeTheta, pc.Groups+1, dt)
	t.made = []mlx.Array{t.main.cos, t.main.sin, t.cp.cos, t.cp.sin}
	t.made = append(append(t.made, t.main.made...), t.cp.made...)
	return t, nil
}

// text is the text projection of token ids: [1, L, hidden].
func (t *talker) text(ids []int32) (mlx.Array, error) {
	e, err := t.textEmbed.lookup(ids, t.main.dt)
	if err != nil {
		return 0, err
	}
	return t.fc2.apply(mlx.SiLU(t.fc1.apply(e))), nil
}

// codec is codec embeddings of ids [1, L]: [1, L, hidden].
func (t *talker) codec(ids mlx.Array) mlx.Array { return rows(t.codecEmbed, ids) }

// step runs the talker over h [1, L, hidden] and returns the last
// position's first-code logits [1, vocab] and hidden state [1, 1, hidden].
// The head reads every position, as mlx-audio's does, so that a prompt's
// last logits come out of the same kernel.
func (t *talker) step(h mlx.Array, L int, c *kvCache) (logits, hidden mlx.Array) {
	out := t.main.forward(h, L, c)
	hid, V := t.main.cfg.Hidden, t.main.cfg.Vocab
	all := t.head.apply(out)
	if L == 1 {
		return mlx.Reshape(all, 1, V), out
	}
	logits = mlx.Reshape(mlx.Slice(all, []int{0, L - 1, 0}, []int{1, L, V}), 1, V)
	hidden = mlx.Slice(out, []int{0, L - 1, 0}, []int{1, L, hid})
	return logits, hidden
}

// predict runs the code predictor over h [1, L, talker hidden] and returns
// code (group+1)'s logits at the last position, [1, vocab].
func (t *talker) predict(h mlx.Array, L int, c *kvCache, group int) mlx.Array {
	if t.cpProj.W != 0 {
		h = t.cpProj.apply(h)
	}
	out := t.cp.forward(h, L, c)
	all := t.cpHeads[group].apply(out)
	V := t.cp.cfg.Vocab
	if L == 1 {
		return mlx.Reshape(all, 1, V)
	}
	return mlx.Reshape(mlx.Slice(all, []int{0, L - 1, 0}, []int{1, L, V}), 1, V)
}
