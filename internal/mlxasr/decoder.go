package mlxasr

import (
	"fmt"
	"math"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// The text decoder: Qwen3 (mlx-audio's TextModel) — RMSNorm, attention with
// RMS-normed queries and keys and rotary positions, a SwiGLU MLP — its output
// tied to the embedding.

type decLayer struct {
	inNorm, postNorm, qNorm, kNorm mlx.Array
	q, k, v, o, gate, up, down     linear
}

type decoder struct {
	cfg    textConfig
	embed  linear
	head   linear // the embedding when tied
	norm   mlx.Array
	layers []decLayer
}

func loadDecoder(w *weights, cfg textConfig) (*decoder, error) {
	d := &decoder{cfg: cfg}
	var err error
	if d.embed, err = w.lin("model.embed_tokens"); err != nil {
		return nil, err
	}
	d.head = d.embed
	if !cfg.Tied {
		if d.head, err = w.lin("lm_head"); err != nil {
			return nil, err
		}
	}
	if d.norm, err = w.get("model.norm.weight"); err != nil {
		return nil, err
	}
	d.layers = make([]decLayer, cfg.Layers)
	for i := range d.layers {
		l := &d.layers[i]
		p := fmt.Sprintf("model.layers.%d.", i)
		for _, x := range []struct {
			dst  *mlx.Array
			name string
		}{
			{&l.inNorm, "input_layernorm.weight"}, {&l.postNorm, "post_attention_layernorm.weight"},
			{&l.qNorm, "self_attn.q_norm.weight"}, {&l.kNorm, "self_attn.k_norm.weight"},
		} {
			if *x.dst, err = w.get(p + x.name); err != nil {
				return nil, err
			}
		}
		for _, x := range []struct {
			dst  *linear
			name string
		}{
			{&l.q, "self_attn.q_proj"}, {&l.k, "self_attn.k_proj"}, {&l.v, "self_attn.v_proj"}, {&l.o, "self_attn.o_proj"},
			{&l.gate, "mlp.gate_proj"}, {&l.up, "mlp.up_proj"}, {&l.down, "mlp.down_proj"},
		} {
			if *x.dst, err = w.lin(p + x.name); err != nil {
				return nil, err
			}
		}
	}
	return d, nil
}

// kvCache is one sequence's keys and values, [1, Hkv, capacity, D] a layer.
type kvCache struct {
	keys, vals []mlx.Array
	offset     int
}

const kvStep = 256

func (d *decoder) newCache() *kvCache {
	return &kvCache{keys: make([]mlx.Array, d.cfg.Layers), vals: make([]mlx.Array, d.cfg.Layers)}
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

// step reads h [1, L, hidden] at the cache's offset and returns the last
// position's logits, [1, vocab], unevaluated; the cache's arrays are replaced
// by lazy ones, to evaluate along with them (cache.arrays()).
func (d *decoder) step(h mlx.Array, c *kvCache) mlx.Array {
	cfg := d.cfg
	L := h.Shape()[1]
	scope := mlx.NewScope()
	defer scope.Close()
	old := c.arrays()
	for i := range d.layers {
		l := &d.layers[i]
		x := mlx.RMSNorm(h, l.inNorm, cfg.Eps)
		h = mlx.Add(h, d.attention(l, x, c, i))
		x = mlx.RMSNorm(h, l.postNorm, cfg.Eps)
		h = mlx.Add(h, l.down.apply(mlx.Multiply(mlx.SiLU(l.gate.apply(x)), l.up.apply(x))))
	}
	h = mlx.Slice(h, []int{0, L - 1, 0}, []int{1, L, cfg.Hidden})
	h = mlx.RMSNorm(h, d.norm, cfg.Eps)
	logits := mlx.Reshape(d.head.apply(h), 1, cfg.Vocab)
	c.offset += L
	scope.Keep(logits)
	scope.Escape(c.arrays()...)
	for _, a := range old {
		a.Free()
	}
	return logits
}

func (d *decoder) attention(l *decLayer, x mlx.Array, c *kvCache, i int) mlx.Array {
	cfg := d.cfg
	L := x.Shape()[1]
	H, Hkv, D := cfg.Heads, cfg.KVHeads, cfg.HeadDim
	q := mlx.Reshape(l.q.apply(x), 1, L, H, D)
	k := mlx.Reshape(l.k.apply(x), 1, L, Hkv, D)
	v := mlx.Reshape(l.v.apply(x), 1, L, Hkv, D)
	q = mlx.Transpose(mlx.RMSNorm(q, l.qNorm, cfg.Eps), 0, 2, 1, 3)
	k = mlx.Transpose(mlx.RMSNorm(k, l.kNorm, cfg.Eps), 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	q = mlx.RoPE(q, D, false, cfg.RopeTheta, 1, c.offset)
	k = mlx.RoPE(k, D, false, cfg.RopeTheta, 1, c.offset)

	prev, end := c.offset, c.offset+L
	for _, kv := range []struct {
		buf *mlx.Array
		new mlx.Array
	}{{&c.keys[i], k}, {&c.vals[i], v}} {
		if *kv.buf == 0 || (*kv.buf).Shape()[2] < end {
			grown := mlx.Zeros(kv.new.Dtype(), 1, Hkv, (end+kvStep-1)/kvStep*kvStep, D)
			if *kv.buf != 0 && prev > 0 {
				grown = mlx.SliceUpdate(grown, mlx.Slice(*kv.buf, []int{0, 0, 0, 0}, []int{1, Hkv, prev, D}),
					[]int{0, 0, 0, 0}, []int{1, Hkv, prev, D})
			}
			*kv.buf = grown
		}
		*kv.buf = mlx.SliceUpdate(*kv.buf, kv.new, []int{0, 0, prev, 0}, []int{1, Hkv, end, D})
	}
	k = mlx.Slice(c.keys[i], []int{0, 0, 0, 0}, []int{1, Hkv, end, D})
	v = mlx.Slice(c.vals[i], []int{0, 0, 0, 0}, []int{1, Hkv, end, D})
	mode := ""
	if L > 1 {
		mode = "causal"
	}
	o := mlx.SDPA(q, k, v, float32(1/math.Sqrt(float64(D))), mode, 0)
	return l.o.apply(mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), 1, L, H*D))
}
