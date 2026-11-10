package mlxtts

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// The speech tokenizer's decoder (mlx-audio's speech_tokenizer.py): codes
// [T, 16] to 24 kHz audio. The codebooks are looked up and summed, a causal
// convolution and eight transformer layers read them, two transposed
// convolutions and ConvNeXt blocks double the frames twice, and four blocks of
// snake activations, a transposed convolution and dilated residual units
// upsample by 8, 5, 4 and 3: 1,920 samples a frame. Everything is causal, so
// the same pass run a piece at a time — convolutions keeping their last
// inputs, the transposed ones the overlap they spill, the transformer its
// keys and values — gives the audio a whole pass gives: that is how speech is
// streamed. A whole decode is a piece from a fresh state.

const sampleRate = 24000

type codecConfig struct {
	Latent        int     `json:"latent_dim"`
	CodebookDim   int     `json:"codebook_dim"`
	CodebookSize  int     `json:"codebook_size"`
	DecoderDim    int     `json:"decoder_dim"`
	Hidden        int     `json:"hidden_size"`
	FFN           int     `json:"intermediate_size"`
	HeadDim       int     `json:"head_dim"`
	Heads         int     `json:"num_attention_heads"`
	KVHeads       int     `json:"num_key_value_heads"`
	Layers        int     `json:"num_hidden_layers"`
	Quantizers    int     `json:"num_quantizers"`
	Semantic      int     `json:"num_semantic_quantizers"`
	Eps           float32 `json:"rms_norm_eps"`
	RopeTheta     float32 `json:"rope_theta"`
	UpsampleRates []int   `json:"upsample_rates"`
	UpsamplingRat []int   `json:"upsampling_ratios"`
}

// conv is a causal 1-D convolution, weight [out, K, in/groups] as MLX keeps
// it, its inputs padded on the left by (K-1)·dilation.
type conv struct {
	w, b             mlx.Array
	k, dilation, grp int
}

func (c conv) pad() int { return (c.k - 1) * c.dilation }

// snake is SnakeBeta, x + 1/(e^β+ε)·sin²(x·e^α), its per-channel factors
// worked out once as mlx-audio works them out each call.
type snake struct{ alpha, inv mlx.Array }

func (s snake) apply(x mlx.Array) mlx.Array {
	sq := mlx.Power(mlx.Sin(mlx.Multiply(x, s.alpha)), mlx.ScalarAs(2, x.Dtype()))
	return mlx.Add(x, mlx.Multiply(s.inv, sq))
}

type residualUnit struct {
	act1, act2   snake
	conv1, conv2 conv
}

type block struct {
	act        snake
	up         mlx.Array // the transposed convolution's weight [out, 2r, in]
	upB        mlx.Array
	rate       int
	units      [3]residualUnit
	out, inDim int
}

type convNeXt struct {
	dw           conv
	normW, normB mlx.Array
	pw1, pw2     linear
	gamma        mlx.Array
}

type codecLayer struct {
	inNorm, postNorm    mlx.Array
	q, k, v, o          linear
	gate, up, down      linear
	attnScale, mlpScale mlx.Array
}

type codec struct {
	cfg       codecConfig
	codebooks []mlx.Array // [size, dim/2] each: the first the semantic one
	firstOut  mlx.Array   // output projections [dim, 1, dim/2]
	restOut   mlx.Array
	preConv   conv
	inProj    linear
	layers    []codecLayer
	norm      mlx.Array
	outProj   linear
	cos, sin  mlx.Array
	upT       []mlx.Array // the frame-doubling transposed convolutions [out, f, in]
	upTB      []mlx.Array
	upRatio   []int
	upNeXt    []convNeXt
	dec0      conv
	blocks    []block
	outAct    snake
	outConv   conv
	made      []mlx.Array
	perFrame  int
}

// mlxShaped is mlx-audio's check_array_shape_qwen3: whether a convolution's
// weight is already as MLX keeps it, [out, K, in], rather than as PyTorch
// does — [out, in, K] for a convolution, [in, out, K] for a transposed one.
func mlxShaped(a mlx.Array) bool {
	sh := a.Shape()
	if len(sh) != 3 {
		return false
	}
	d2, d3 := sh[1], sh[2]
	switch {
	case d2 == 1:
		return d3 > 64
	case d3 == 1:
		return d2 <= 64
	}
	return d2 < d3
}

func loadCodec(w *weights, dir string) (cd *codec, err error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var top struct {
		Decoder codecConfig `json:"decoder_config"`
	}
	if err := json.Unmarshal(b, &top); err != nil {
		return nil, fmt.Errorf("%s/config.json: %w", dir, err)
	}
	cfg := top.Decoder
	cd = &codec{cfg: cfg, perFrame: 1}
	for _, r := range append(append([]int{}, cfg.UpsampleRates...), cfg.UpsamplingRat...) {
		cd.perFrame *= r
	}
	sc := mlx.NewScope()
	defer sc.Close()
	// made keeps an array computed here past the scope: the model frees it.
	made := func(a mlx.Array) mlx.Array {
		mlx.Eval(a)
		sc.Escape(a)
		cd.made = append(cd.made, a)
		return a
	}
	// A weight laid out again, or worked out of others, is kept in its new
	// form only: the file's is let go.
	convW := func(name string) (mlx.Array, error) {
		a, err := w.get(name)
		if err != nil || mlxShaped(a) {
			return a, err
		}
		delete(w.taken, name)
		return made(mlx.Contiguous(mlx.Transpose(a, 0, 2, 1))), nil
	}
	tconvW := func(name string) (mlx.Array, error) {
		a, err := w.get(name)
		if err != nil || mlxShaped(a) {
			return a, err
		}
		delete(w.taken, name)
		return made(mlx.Contiguous(mlx.Transpose(a, 1, 2, 0))), nil
	}
	loadConv := func(prefix string, k, dilation, groups int) (conv, error) {
		c := conv{k: k, dilation: dilation, grp: groups}
		var err error
		if c.w, err = convW(prefix + ".weight"); err != nil {
			return c, err
		}
		c.b, err = w.get(prefix + ".bias")
		return c, err
	}
	loadSnake := func(prefix string) (snake, error) {
		a, err := w.get(prefix + ".alpha")
		if err != nil {
			return snake{}, err
		}
		bt, err := w.get(prefix + ".beta")
		if err != nil {
			return snake{}, err
		}
		inv := mlx.Divide(mlx.ScalarAs(1, mlx.Float32), mlx.AddScalar(mlx.Exp(bt), 1e-9))
		sn := snake{alpha: made(mlx.Exp(a)), inv: made(inv)}
		delete(w.taken, prefix+".alpha")
		delete(w.taken, prefix+".beta")
		return sn, nil
	}

	// The codebooks: each an embedding sum over its use, as sanitize makes them.
	for _, set := range []struct {
		name string
		n    int
	}{{"rvq_first", cfg.Semantic}, {"rvq_rest", cfg.Quantizers - cfg.Semantic}} {
		for i := 0; i < set.n; i++ {
			p := fmt.Sprintf("decoder.quantizer.%s.vq.layers.%d._codebook.", set.name, i)
			sum, err := w.get(p + "embedding_sum")
			if err != nil {
				return nil, err
			}
			use, err := w.get(p + "cluster_usage")
			if err != nil {
				return nil, err
			}
			n := use.Shape()[0]
			cd.codebooks = append(cd.codebooks, made(mlx.Divide(sum, mlx.Maximum(mlx.Reshape(use, n, 1), mlx.ScalarAs(1e-5, mlx.Float32)))))
			delete(w.taken, p+"embedding_sum")
			delete(w.taken, p+"cluster_usage")
		}
	}
	if cd.firstOut, err = convW("decoder.quantizer.rvq_first.output_proj.weight"); err != nil {
		return nil, err
	}
	if cd.restOut, err = convW("decoder.quantizer.rvq_rest.output_proj.weight"); err != nil {
		return nil, err
	}
	if cd.preConv, err = loadConv("decoder.pre_conv.conv", 3, 1, 1); err != nil {
		return nil, err
	}
	p := "decoder.pre_transformer."
	if cd.inProj, err = w.lin(p + "input_proj"); err != nil {
		return nil, err
	}
	if cd.outProj, err = w.lin(p + "output_proj"); err != nil {
		return nil, err
	}
	if cd.norm, err = w.get(p + "norm.weight"); err != nil {
		return nil, err
	}
	cd.layers = make([]codecLayer, cfg.Layers)
	for i := range cd.layers {
		l := &cd.layers[i]
		lp := fmt.Sprintf("%slayers.%d.", p, i)
		for _, x := range []struct {
			dst  *mlx.Array
			name string
		}{{&l.inNorm, "input_layernorm.weight"}, {&l.postNorm, "post_attention_layernorm.weight"},
			{&l.attnScale, "self_attn_layer_scale.scale"}, {&l.mlpScale, "mlp_layer_scale.scale"}} {
			if *x.dst, err = w.get(lp + x.name); err != nil {
				return nil, err
			}
		}
		for _, x := range []struct {
			dst  *linear
			name string
		}{{&l.q, "self_attn.q_proj"}, {&l.k, "self_attn.k_proj"}, {&l.v, "self_attn.v_proj"}, {&l.o, "self_attn.o_proj"},
			{&l.gate, "mlp.gate_proj"}, {&l.up, "mlp.up_proj"}, {&l.down, "mlp.down_proj"}} {
			if *x.dst, err = w.lin(lp + x.name); err != nil {
				return nil, err
			}
		}
	}
	for i, f := range cfg.UpsamplingRat {
		up := fmt.Sprintf("decoder.upsample.%d.", i)
		t, err := tconvW(up + "0.conv.weight")
		if err != nil {
			return nil, err
		}
		tb, err := w.get(up + "0.conv.bias")
		if err != nil {
			return nil, err
		}
		var nx convNeXt
		if nx.dw, err = loadConv(up+"1.dwconv.conv", 7, 1, cfg.Latent); err != nil {
			return nil, err
		}
		for _, x := range []struct {
			dst  *mlx.Array
			name string
		}{{&nx.normW, "1.norm.weight"}, {&nx.normB, "1.norm.bias"}, {&nx.gamma, "1.gamma"}} {
			if *x.dst, err = w.get(up + x.name); err != nil {
				return nil, err
			}
		}
		if nx.pw1, err = w.lin(up + "1.pwconv1"); err != nil {
			return nil, err
		}
		if nx.pw2, err = w.lin(up + "1.pwconv2"); err != nil {
			return nil, err
		}
		cd.upT, cd.upTB, cd.upRatio, cd.upNeXt = append(cd.upT, t), append(cd.upTB, tb), append(cd.upRatio, f), append(cd.upNeXt, nx)
	}
	if cd.dec0, err = loadConv("decoder.decoder.0.conv", 7, 1, 1); err != nil {
		return nil, err
	}
	for i, r := range cfg.UpsampleRates {
		bp := fmt.Sprintf("decoder.decoder.%d.block.", i+1)
		bl := block{rate: r, inDim: cfg.DecoderDim >> i, out: cfg.DecoderDim >> (i + 1)}
		if bl.act, err = loadSnake(bp + "0"); err != nil {
			return nil, err
		}
		if bl.up, err = tconvW(bp + "1.conv.weight"); err != nil {
			return nil, err
		}
		if bl.upB, err = w.get(bp + "1.conv.bias"); err != nil {
			return nil, err
		}
		for u, d := range []int{1, 3, 9} {
			up := fmt.Sprintf("%s%d.", bp, u+2)
			ru := &bl.units[u]
			if ru.act1, err = loadSnake(up + "act1"); err != nil {
				return nil, err
			}
			if ru.act2, err = loadSnake(up + "act2"); err != nil {
				return nil, err
			}
			if ru.conv1, err = loadConv(up+"conv1.conv", 7, d, 1); err != nil {
				return nil, err
			}
			if ru.conv2, err = loadConv(up+"conv2.conv", 1, 1, 1); err != nil {
				return nil, err
			}
		}
		cd.blocks = append(cd.blocks, bl)
	}
	n := len(cfg.UpsampleRates) + 1
	if cd.outAct, err = loadSnake(fmt.Sprintf("decoder.decoder.%d", n)); err != nil {
		return nil, err
	}
	if cd.outConv, err = loadConv(fmt.Sprintf("decoder.decoder.%d.conv", n+1), 7, 1, 1); err != nil {
		return nil, err
	}
	cd.cos, cd.sin = ropeTables(cfg.HeadDim, cfg.RopeTheta, ropeTable, mlx.Float32)
	cd.made = append(cd.made, cd.cos, cd.sin)
	return cd, nil
}

// codecState is what a streamed decode carries from one piece to the next:
// each causal convolution's last inputs, each upsampling block's overlap,
// the transformer's keys and values.
type codecState struct {
	bufs     map[*conv]mlx.Array
	overflow map[*block]mlx.Array
	cache    *kvCache
}

func newCodecState(layers int) *codecState {
	return &codecState{bufs: map[*conv]mlx.Array{}, overflow: map[*block]mlx.Array{}, cache: newCache(layers)}
}

// arrays are the state's arrays, for a caller keeping it past a scope.
func (s *codecState) arrays() []mlx.Array {
	r := s.cache.arrays()
	for _, a := range s.bufs {
		r = append(r, a)
	}
	for _, a := range s.overflow {
		r = append(r, a)
	}
	return r
}

func (s *codecState) free() {
	for _, a := range s.arrays() {
		a.Free()
	}
	s.bufs, s.overflow = map[*conv]mlx.Array{}, map[*block]mlx.Array{}
	s.cache = newCache(len(s.cache.keys))
}

// causal runs c over x [1, L, C], the state's last inputs (zeros at first)
// before it.
func (cd *codec) causal(c *conv, x mlx.Array, st *codecState) mlx.Array {
	if p := c.pad(); p > 0 {
		if buf, ok := st.bufs[c]; ok {
			x = mlx.Concatenate(1, buf, x)
		} else {
			sh := x.Shape()
			x = mlx.Concatenate(1, mlx.Zeros(x.Dtype(), sh[0], p, sh[2]), x)
		}
		sh := x.Shape()
		st.bufs[c] = mlx.Slice(x, []int{0, sh[1] - p, 0}, []int{sh[0], sh[1], sh[2]})
	}
	return mlx.Add(mlx.Conv1d(x, c.w, 1, 0, c.dilation, c.grp), c.b)
}

// rmsNorm is the decoder's, in float32 by hand: w·(x·rsqrt(mean(x²)+ε)).
func rmsNorm(x, w mlx.Array, eps float32) mlx.Array {
	v := mlx.Mean(mlx.Power(x, mlx.ScalarAs(2, x.Dtype())), -1, true)
	return mlx.Multiply(w, mlx.Multiply(x, mlx.Rsqrt(mlx.AddScalar(v, eps))))
}

// gelu is mlx.nn.gelu's x·(1 + erf(x/√2))/2.
func gelu(x mlx.Array) mlx.Array {
	e := mlx.Erf(mlx.Divide(x, mlx.ScalarAs(float32(math.Sqrt2), x.Dtype())))
	return mlx.Divide(mlx.Multiply(x, mlx.AddScalar(e, 1)), mlx.ScalarAs(2, x.Dtype()))
}

// transformer is the decoder's: x [1, L, latent] at the cache's offset.
func (cd *codec) transformer(x mlx.Array, c *kvCache) mlx.Array {
	cfg := cd.cfg
	h := cd.inProj.apply(x)
	L := h.Shape()[1]
	H, D := cfg.Heads, cfg.HeadDim
	cos := mlx.Slice(cd.cos, []int{c.offset, 0}, []int{c.offset + L, D})
	sin := mlx.Slice(cd.sin, []int{c.offset, 0}, []int{c.offset + L, D})
	rot := func(t mlx.Array) mlx.Array {
		t1 := mlx.Slice(t, []int{0, 0, 0, 0}, []int{1, H, L, D / 2})
		t2 := mlx.Slice(t, []int{0, 0, 0, D / 2}, []int{1, H, L, D})
		return mlx.Add(mlx.Multiply(t, cos), mlx.Multiply(mlx.Concatenate(3, mlx.Negative(t2), t1), sin))
	}
	for i := range cd.layers {
		l := &cd.layers[i]
		a := rmsNorm(h, l.inNorm, cfg.Eps)
		q := mlx.Transpose(mlx.Reshape(l.q.apply(a), 1, L, H, D), 0, 2, 1, 3)
		k := mlx.Transpose(mlx.Reshape(l.k.apply(a), 1, L, cfg.KVHeads, D), 0, 2, 1, 3)
		v := mlx.Transpose(mlx.Reshape(l.v.apply(a), 1, L, cfg.KVHeads, D), 0, 2, 1, 3)
		q, k = rot(q), rot(k)
		k, v = c.update(i, k, v, L, cfg.KVHeads, D, mlx.Float32)
		o := l.o.apply(mlx.Reshape(mlx.Transpose(attend(q, k, v, L, D), 0, 2, 1, 3), 1, L, H*D))
		h = mlx.Add(h, mlx.Multiply(l.attnScale, o))
		a = rmsNorm(h, l.postNorm, cfg.Eps)
		m := l.down.apply(mlx.Multiply(mlx.SiLU(l.gate.apply(a)), l.up.apply(a)))
		h = mlx.Add(h, mlx.Multiply(l.mlpScale, m))
	}
	c.offset += L
	return cd.outProj.apply(rmsNorm(h, cd.norm, cfg.Eps))
}

// step decodes codes [1, T, 16] (int32) to audio [1, T·1920], continuing st.
func (cd *codec) step(codes mlx.Array, st *codecState) mlx.Array {
	T := codes.Shape()[1]
	q := cd.cfg.Quantizers
	look := func(i int) mlx.Array {
		ids := mlx.Reshape(mlx.Slice(codes, []int{0, 0, i}, []int{1, T, i + 1}), 1, T)
		return rows(cd.codebooks[i], ids)
	}
	first := look(0)
	for i := 1; i < cd.cfg.Semantic; i++ {
		first = mlx.Add(first, look(i))
	}
	rest := look(cd.cfg.Semantic)
	for i := cd.cfg.Semantic + 1; i < q; i++ {
		rest = mlx.Add(rest, look(i))
	}
	h := mlx.Add(mlx.Conv1d(first, cd.firstOut, 1, 0, 1, 1), mlx.Conv1d(rest, cd.restOut, 1, 0, 1, 1))
	h = cd.causal(&cd.preConv, h, st)
	h = cd.transformer(h, st.cache)
	for i := range cd.upT {
		h = mlx.Add(mlx.ConvTranspose1d(h, cd.upT[i], cd.upRatio[i]), cd.upTB[i]) // kernel = stride: nothing spills
		nx := &cd.upNeXt[i]
		y := cd.causal(&nx.dw, h, st)
		y = mlx.LayerNorm(y, nx.normW, nx.normB, 1e-6)
		y = nx.pw2.apply(gelu(nx.pw1.apply(y)))
		h = mlx.Add(h, mlx.Multiply(nx.gamma, y))
	}
	wav := cd.causal(&cd.dec0, h, st)
	for i := range cd.blocks {
		b := &cd.blocks[i]
		wav = b.act.apply(wav)
		// The overlap a piece spills into the next is carried without the
		// bias, which the next piece adds: added on both sides (as mlx-audio's
		// streaming_step does) it would count twice where they meet.
		y := mlx.ConvTranspose1d(wav, b.up, b.rate)
		sh := y.Shape()
		if ov, ok := st.overflow[b]; ok {
			n := ov.Shape()[1]
			head := mlx.Add(mlx.Slice(y, []int{0, 0, 0}, []int{1, n, sh[2]}), ov)
			y = mlx.Concatenate(1, head, mlx.Slice(y, []int{0, n, 0}, sh))
		}
		trim := b.rate // kernel 2r, stride r
		st.overflow[b] = mlx.Slice(y, []int{0, sh[1] - trim, 0}, sh)
		wav = mlx.Add(mlx.Slice(y, []int{0, 0, 0}, []int{1, sh[1] - trim, sh[2]}), b.upB)
		for u := range b.units {
			ru := &b.units[u]
			r := wav
			x := cd.causal(&ru.conv1, ru.act1.apply(wav), st)
			x = cd.causal(&ru.conv2, ru.act2.apply(x), st)
			wav = mlx.Add(x, r)
		}
	}
	wav = cd.causal(&cd.outConv, cd.outAct.apply(wav), st)
	n := wav.Shape()[1]
	wav = mlx.Reshape(wav, 1, n)
	return mlx.Minimum(mlx.Maximum(wav, mlx.ScalarAs(-1, mlx.Float32)), mlx.ScalarAs(1, mlx.Float32))
}
