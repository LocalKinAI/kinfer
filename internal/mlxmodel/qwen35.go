// Package mlxmodel holds the model architectures kinfer runs on MLX, written
// in Go over internal/mlx. Each one is checked against mlx-lm's Python
// implementation of the same architecture, layer by layer, before it serves.
package mlxmodel

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// Qwen35Config is the text part of a Qwen3.5 config.json (model_type
// qwen3_5, text_config qwen3_5_text): gated DeltaNet linear attention in three
// of every four layers, full attention with an output gate in the fourth.
type Qwen35Config struct {
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	HeadDim               int     `json:"head_dim"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	VocabSize             int     `json:"vocab_size"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
	FullAttentionInterval int     `json:"full_attention_interval"`
	LinearNumKeyHeads     int     `json:"linear_num_key_heads"`
	LinearNumValueHeads   int     `json:"linear_num_value_heads"`
	LinearKeyHeadDim      int     `json:"linear_key_head_dim"`
	LinearValueHeadDim    int     `json:"linear_value_head_dim"`
	LinearConvKernelDim   int     `json:"linear_conv_kernel_dim"`
	NumExperts            int     `json:"num_experts"`
	RopeParameters        struct {
		RopeTheta           float32 `json:"rope_theta"`
		PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	} `json:"rope_parameters"`
	EOSTokenID json.RawMessage `json:"eos_token_id"`

	// Quant is how the linear layers are quantized, from the top level of
	// config.json; nil for an unquantized model.
	Quant *Quantization `json:"-"`
}

// EOS is the configured end-of-sequence ids, given as one or as a list.
func (c *Qwen35Config) EOS() []int32 {
	var one int32
	if json.Unmarshal(c.EOSTokenID, &one) == nil {
		return []int32{one}
	}
	var many []int32
	json.Unmarshal(c.EOSTokenID, &many)
	return many
}

// Quantization is MLX's "quantization" entry: the defaults, and per-module
// exceptions keyed by module path.
type Quantization struct {
	GroupSize int
	Bits      int
	Mode      string
	Modules   map[string]Quantization
}

func (q *Quantization) UnmarshalJSON(b []byte) error {
	var all map[string]json.RawMessage
	if err := json.Unmarshal(b, &all); err != nil {
		return err
	}
	q.GroupSize, q.Bits, q.Mode = 64, 4, "affine"
	for k, v := range all {
		switch k {
		case "group_size":
			json.Unmarshal(v, &q.GroupSize)
		case "bits":
			json.Unmarshal(v, &q.Bits)
		case "mode":
			json.Unmarshal(v, &q.Mode)
		default:
			var sub Quantization
			if json.Unmarshal(v, &sub) == nil && v[0] == '{' {
				if q.Modules == nil {
					q.Modules = map[string]Quantization{}
				}
				q.Modules[k] = sub
			}
		}
	}
	return nil
}

// Serves reports whether a config.json's model_type is one this package
// implements.
func Serves(modelType string) bool { return modelType == "qwen3_5" || modelType == "qwen3_5_text" }

// ReadQwen35Config reads dir/config.json, from text_config when the file is
// a multimodal release's.
func ReadQwen35Config(dir string) (*Qwen35Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var outer struct {
		ModelType    string          `json:"model_type"`
		TextConfig   json.RawMessage `json:"text_config"`
		Quantization *Quantization   `json:"quantization"`
	}
	if err := json.Unmarshal(raw, &outer); err != nil {
		return nil, err
	}
	if !Serves(outer.ModelType) {
		return nil, fmt.Errorf("model_type %q is not qwen3_5", outer.ModelType)
	}
	text := raw
	if len(outer.TextConfig) > 0 {
		text = outer.TextConfig
	}
	c := &Qwen35Config{FullAttentionInterval: 4, LinearConvKernelDim: 4, RMSNormEps: 1e-6}
	c.RopeParameters.RopeTheta, c.RopeParameters.PartialRotaryFactor = 1e7, 0.25
	if err := json.Unmarshal(text, c); err != nil {
		return nil, err
	}
	c.Quant = outer.Quantization
	if c.NumExperts > 0 {
		return nil, fmt.Errorf("Qwen3.5 with %d experts (qwen3_5_moe) is not written yet", c.NumExperts)
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	return c, nil
}

// Linear is a weight matrix [out, in], plain or quantized the way MLX's
// QuantizedLinear stores it: packed W, per-group Scales and, in affine mode,
// Biases.
type Linear struct {
	W, Scales, Biases mlx.Array
	Group, Bits       int
	Mode              string
}

func (l Linear) quantized() bool { return l.Scales != 0 }

func (l Linear) arrays() []mlx.Array {
	r := []mlx.Array{l.W}
	for _, a := range []mlx.Array{l.Scales, l.Biases} {
		if a != 0 {
			r = append(r, a)
		}
	}
	return r
}

// linear is x @ w.T.
func linear(x mlx.Array, w Linear) mlx.Array {
	if w.quantized() {
		return mlx.QuantizedMatmul(x, w.W, w.Scales, w.Biases, w.Group, w.Bits, w.Mode)
	}
	return mlx.Matmul(x, mlx.T(w.W))
}

// rows looks up rows of w, as an embedding: ids [B, L] → [B, L, in].
func (l Linear) rows(ids mlx.Array) mlx.Array {
	if !l.quantized() {
		return mlx.Take(l.W, ids, 0)
	}
	var b mlx.Array
	if l.Biases != 0 {
		b = mlx.Take(l.Biases, ids, 0)
	}
	return mlx.Dequantize(mlx.Take(l.W, ids, 0), mlx.Take(l.Scales, ids, 0), b, l.Group, l.Bits, l.Mode)
}

type linearAttn struct {
	inQKV, inZ, inB, inA, outProj Linear
	conv                          mlx.Array // [conv_dim, K, 1]
	dtBias, aLog                  mlx.Array
	norm                          mlx.Array
}

type fullAttn struct {
	q, k, v, o   Linear
	qNorm, kNorm mlx.Array
}

type qwen35Layer struct {
	linear         *linearAttn // nil for a full-attention layer
	full           *fullAttn
	inNorm, inPost mlx.Array
	gate, up, down Linear
}

// Qwen35 is a loaded Qwen3.5 text model.
type Qwen35 struct {
	Config *Qwen35Config

	// OpsDeltaNet runs the DeltaNet recurrence as ordinary ops, one token at
	// a time, instead of the Metal kernel: the reference the kernel is
	// tested against.
	OpsDeltaNet bool

	embed  Linear
	lmHead *Linear // nil when tied to embed
	norm   mlx.Array
	layers []qwen35Layer
	all    []mlx.Array // every weight, for Free
}

func (m *Qwen35) head() Linear {
	if m.lmHead != nil {
		return *m.lmHead
	}
	return m.embed
}

// LoadQwen35 reads config.json and the safetensors files in dir: a Hugging
// Face release, or an MLX conversion of one (mlx_lm.convert), quantized or not.
//
// EditFile is an edited model's weights that differ from its base's, beside
// the base's own files (see kinlab edit): loaded last, each of its tensors
// takes the place of the base's of the same name.
const EditFile = "edit.safetensors"

// applyEdit puts the tensors of the edit file at path in place of raw's. A
// tensor the base does not have, or has in another shape or dtype, is an
// error: an edit that changes nothing it means to is worse than none.
func applyEdit(raw map[string]mlx.Array, path string) error {
	t, err := mlx.LoadSafetensors(path)
	if err != nil {
		return err
	}
	var bad error
	for k, v := range t {
		old, ok := raw[k]
		switch {
		case bad != nil:
		case !ok:
			bad = fmt.Errorf("%s: the base model has no tensor %s", path, k)
		case old.Dtype() != v.Dtype() || !slices.Equal(old.Shape(), v.Shape()):
			bad = fmt.Errorf("%s: %s is %v %v, the base's is %v %v", path, k, v.Dtype(), v.Shape(), old.Dtype(), old.Shape())
		}
		if bad != nil {
			v.Free()
			continue
		}
		old.Free()
		raw[k] = v
	}
	return bad
}

// A Hugging Face release stores what mlx-lm's sanitize changes: conv1d
// weights as [C, 1, K] where MLX's conv1d wants [C, K, 1], and the five
// RMSNorm weights Qwen3.5 centres on zero (the layer norms, the final norm,
// q_norm and k_norm) as offsets from 1. The DeltaNet's gated norm is stored
// as is. An MLX conversion has had both done already; as in mlx-lm, the
// conv1d shape tells which kind of file this is.
func LoadQwen35(dir string) (m *Qwen35, err error) {
	defer mlx.Recover(&err)
	c, err := ReadQwen35Config(dir)
	if err != nil {
		return nil, err
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	var edit string
	for i, f := range files {
		if filepath.Base(f) == EditFile {
			edit, files = f, append(files[:i:i], files[i+1:]...)
			break
		}
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no .safetensors in %s", dir)
	}
	raw := map[string]mlx.Array{}
	for _, f := range files {
		t, err := mlx.LoadSafetensors(f)
		if err != nil {
			return nil, err
		}
		for k, v := range t {
			raw[k] = v
		}
	}
	if edit != "" {
		if err := applyEdit(raw, edit); err != nil {
			for _, v := range raw {
				v.Free()
			}
			return nil, err
		}
	}
	// Weights used as stored are taken out of raw; whatever is left — the
	// sources of the fixed-up weights, and the vision and MTP tensors this
	// text model does not use — is freed. MLX loads lazily, so the unused
	// ones were never read from disk.
	defer func() {
		for _, v := range raw {
			v.Free()
		}
	}()

	hfLayout := false
	for k, v := range raw {
		if strings.HasSuffix(k, "conv1d.weight") {
			if s := v.Shape(); len(s) == 3 && s[2] != 1 {
				hfLayout = true
			}
			break
		}
	}

	m = &Qwen35{Config: c}
	prefixes := []string{"model.language_model.", "language_model.model.", "language_model.", "model.", ""}
	key := func(name string) (string, bool) {
		for _, prefix := range prefixes {
			if _, ok := raw[prefix+name]; ok {
				return prefix + name, true
			}
		}
		return "", false
	}
	must := func(name string) string {
		k, ok := key(name)
		if !ok {
			panic(&mlx.Error{Msg: "weight " + name + " missing"})
		}
		return k
	}
	keep := func(a mlx.Array) mlx.Array {
		m.all = append(m.all, a)
		return a
	}
	take := func(k string) mlx.Array {
		a := raw[k]
		delete(raw, k)
		return keep(a)
	}
	plain := func(name string) mlx.Array { return take(must(name)) }
	centred := func(name string) mlx.Array {
		if !hfLayout {
			return plain(name)
		}
		return keep(mlx.AddScalar(raw[must(name)], 1))
	}
	lin := func(module string) Linear {
		wk := must(module + ".weight")
		l := Linear{W: take(wk)}
		base := strings.TrimSuffix(wk, ".weight")
		if _, ok := raw[base+".scales"]; ok {
			q := Quantization{GroupSize: 64, Bits: 4, Mode: "affine"}
			if c.Quant != nil {
				q = *c.Quant
				if o, ok := c.Quant.Modules[base]; ok {
					q = o
				}
			}
			l.Scales, l.Group, l.Bits, l.Mode = take(base+".scales"), q.GroupSize, q.Bits, q.Mode
			if _, ok := raw[base+".biases"]; ok {
				l.Biases = take(base + ".biases")
			}
		}
		return l
	}

	m.embed = lin("embed_tokens")
	if !c.TieWordEmbeddings {
		if _, ok := key("lm_head.weight"); ok {
			h := lin("lm_head")
			m.lmHead = &h
		}
	}
	m.norm = centred("norm.weight")
	for i := 0; i < c.NumHiddenLayers; i++ {
		p := fmt.Sprintf("layers.%d.", i)
		l := qwen35Layer{
			inNorm: centred(p + "input_layernorm.weight"),
			inPost: centred(p + "post_attention_layernorm.weight"),
			gate:   lin(p + "mlp.gate_proj"),
			up:     lin(p + "mlp.up_proj"),
			down:   lin(p + "mlp.down_proj"),
		}
		if (i+1)%c.FullAttentionInterval != 0 {
			a := p + "linear_attn."
			conv := raw[must(a+"conv1d.weight")]
			if hfLayout {
				conv = mlx.Swapaxes(conv, 1, 2)
			}
			l.linear = &linearAttn{
				inQKV: lin(a + "in_proj_qkv"), inZ: lin(a + "in_proj_z"),
				inB: lin(a + "in_proj_b"), inA: lin(a + "in_proj_a"),
				outProj: lin(a + "out_proj"), conv: keep(mlx.Contiguous(conv)),
				dtBias: plain(a + "dt_bias"), aLog: plain(a + "A_log"), norm: plain(a + "norm.weight"),
			}
		} else {
			a := p + "self_attn."
			l.full = &fullAttn{
				q: lin(a + "q_proj"), k: lin(a + "k_proj"),
				v: lin(a + "v_proj"), o: lin(a + "o_proj"),
				qNorm: centred(a + "q_norm.weight"), kNorm: centred(a + "k_norm.weight"),
			}
		}
		m.layers = append(m.layers, l)
	}
	mlx.Eval(m.all...)
	return m, nil
}

// Free releases the weights.
func (m *Qwen35) Free() {
	for _, a := range m.all {
		a.Free()
	}
	m.all = nil
}

// Qwen35Cache is one sequence's state: a DeltaNet layer keeps its last K-1
// convolution inputs and its [Hv, Dv, Dk] recurrent state; an attention layer
// keeps its keys and values, in buffers that grow kvStep positions at a time
// so a decode step writes one position instead of copying them all.
type Qwen35Cache struct {
	conv, state []mlx.Array // per layer, 0 where unused
	keys, vals  []mlx.Array // [B, Hkv, capacity, D]
	Offset      int
}

const kvStep = 256

// NewCache makes an empty cache for one sequence.
func (m *Qwen35) NewCache() *Qwen35Cache {
	n := len(m.layers)
	return &Qwen35Cache{conv: make([]mlx.Array, n), state: make([]mlx.Array, n),
		keys: make([]mlx.Array, n), vals: make([]mlx.Array, n)}
}

// Arrays are the cache's current arrays, to evaluate along with a Step's
// logits.
func (c *Qwen35Cache) Arrays() []mlx.Array {
	var r []mlx.Array
	for _, set := range [][]mlx.Array{c.conv, c.state, c.keys, c.vals} {
		for _, a := range set {
			if a != 0 {
				r = append(r, a)
			}
		}
	}
	return r
}

// Clone is a second cache at the same point, sharing c's arrays rather than
// copying them. Either can read on: what it writes are new arrays, and a
// buffer the other still holds is copied before it is written to.
func (c *Qwen35Cache) Clone() *Qwen35Cache {
	d := &Qwen35Cache{Offset: c.Offset}
	for _, p := range []struct{ dst, src *[]mlx.Array }{
		{&d.conv, &c.conv}, {&d.state, &c.state}, {&d.keys, &c.keys}, {&d.vals, &c.vals},
	} {
		*p.dst = make([]mlx.Array, len(*p.src))
		for i, a := range *p.src {
			if a != 0 {
				(*p.dst)[i] = a.Ref()
			}
		}
	}
	return d
}

// Free releases the cache's arrays.
func (c *Qwen35Cache) Free() {
	for _, a := range c.Arrays() {
		a.Free()
	}
	*c = Qwen35Cache{conv: make([]mlx.Array, len(c.conv)), state: make([]mlx.Array, len(c.conv)),
		keys: make([]mlx.Array, len(c.conv)), vals: make([]mlx.Array, len(c.conv))}
}

// Forward runs tokens [B, L] through the model, advancing cache, and returns
// the logits of the last position, [B, vocab], evaluated. When trace is
// non-nil it is called with each layer's output.
func (m *Qwen35) Forward(tokens [][]int32, cache *Qwen35Cache, trace func(layer int, h mlx.Array)) (logits mlx.Array, err error) {
	defer mlx.Recover(&err)
	B, L := len(tokens), len(tokens[0])
	flat := make([]int32, 0, B*L)
	for _, t := range tokens {
		flat = append(flat, t...)
	}
	ids := mlx.FromInt32(flat, B, L)
	logits = m.Step(ids, cache, trace)
	ids.Free()
	mlx.Eval(append([]mlx.Array{logits}, cache.Arrays()...)...)
	return logits, nil
}

// Step records one forward pass of tokens [B, L] int32 and returns the last
// position's logits, [B, vocab] — without running anything. The cache's
// arrays are replaced by lazy ones too; evaluate them with the logits
// (cache.Arrays()). Because tokens can itself be unevaluated — the argmax or
// sample of the previous step's logits — the next step's graph can be
// recorded while the GPU still runs this one, which is how decoding keeps the
// GPU busy. Failed ops panic with *mlx.Error.
func (m *Qwen35) Step(tokens mlx.Array, cache *Qwen35Cache, trace func(layer int, h mlx.Array)) mlx.Array {
	return m.step(tokens, cache, trace, false)
}

// Logits reads tokens from the start and returns the logits at every
// position, [len(tokens), vocab], evaluated: what the model predicts after
// each token, for comparing two models' predictions on the same text.
func (m *Qwen35) Logits(tokens []int32) (logits mlx.Array, err error) {
	c := m.NewCache()
	defer c.Free()
	return m.ReadAll(tokens, c)
}

// ReadAll is Forward for one sequence, returning the logits at every
// position, [len(tokens), vocab], evaluated: a long text is read a chunk at a
// time with the same cache.
func (m *Qwen35) ReadAll(tokens []int32, cache *Qwen35Cache) (logits mlx.Array, err error) {
	defer mlx.Recover(&err)
	ids := mlx.FromInt32(tokens, 1, len(tokens))
	logits = m.step(ids, cache, nil, true)
	ids.Free()
	mlx.Eval(append([]mlx.Array{logits}, cache.Arrays()...)...)
	return logits, nil
}

func (m *Qwen35) step(tokens mlx.Array, cache *Qwen35Cache, trace func(layer int, h mlx.Array), every bool) mlx.Array {
	s := tokens.Shape()
	B, L := s[0], s[1]
	scope := mlx.NewScope()
	defer scope.Close()
	old := cache.Arrays()

	h := m.embed.rows(tokens)
	for i := range m.layers {
		l := &m.layers[i]
		x := mlx.RMSNorm(h, l.inNorm, m.Config.RMSNormEps)
		var r mlx.Array
		if l.linear != nil {
			r = m.deltaNet(l.linear, x, cache, i)
		} else {
			r = m.attention(l.full, x, cache, i)
		}
		h = mlx.Add(h, r)
		x = mlx.RMSNorm(h, l.inPost, m.Config.RMSNormEps)
		h = mlx.Add(h, linear(mlx.Multiply(mlx.SiLU(linear(x, l.gate)), linear(x, l.up)), l.down))
		if trace != nil {
			trace(i, h)
		}
	}
	rows := B * L
	if !every {
		h, rows = mlx.Slice(h, []int{0, L - 1, 0}, []int{B, L, m.Config.HiddenSize}), B
	}
	h = mlx.RMSNorm(h, m.norm, m.Config.RMSNormEps)
	logits := mlx.Reshape(linear(h, m.head()), rows, m.Config.VocabSize)

	cache.Offset += L
	scope.Keep(logits)
	scope.Escape(cache.Arrays()...)
	// The graph holds what it still needs of the old cache; our handles to it
	// can go now, which also lets MLX update the KV buffers in place.
	for _, a := range old {
		a.Free()
	}
	return logits
}

func (m *Qwen35) deltaNet(w *linearAttn, x mlx.Array, cache *Qwen35Cache, i int) mlx.Array {
	c := m.Config
	s := x.Shape()
	B, L := s[0], s[1]
	Hk, Hv, Dk, Dv := c.LinearNumKeyHeads, c.LinearNumValueHeads, c.LinearKeyHeadDim, c.LinearValueHeadDim
	keyDim, valDim := Hk*Dk, Hv*Dv
	convDim := 2*keyDim + valDim
	K := c.LinearConvKernelDim

	qkv := linear(x, w.inQKV)
	z := mlx.Reshape(linear(x, w.inZ), B, L, Hv, Dv)
	b := linear(x, w.inB)
	a := linear(x, w.inA)

	convState := cache.conv[i]
	if convState == 0 {
		convState = mlx.Zeros(x.Dtype(), B, K-1, convDim)
	}
	convIn := mlx.Concatenate(1, convState, qkv)
	cache.conv[i] = mlx.Contiguous(mlx.Slice(convIn, []int{0, L, 0}, []int{B, L + K - 1, convDim}))
	convOut := mlx.SiLU(mlx.Conv1d(convIn, w.conv, 1, 0, 1, convDim))

	parts := mlx.Split(convOut, -1, keyDim, 2*keyDim)
	q := mlx.Reshape(parts[0], B, L, Hk, Dk)
	k := mlx.Reshape(parts[1], B, L, Hk, Dk)
	v := mlx.Reshape(parts[2], B, L, Hv, Dv)

	// normalize_qk: L2-normalise q and k as an RMS norm, with eps scaled to
	// match, and fold the Dk^-0.5 readout scale into q.
	inv := float32(1 / math.Sqrt(float64(Dk)))
	eps := 1e-6 * inv * inv
	q = mlx.MulScalar(mlx.RMSNorm(q, 0, eps), inv*inv)
	k = mlx.MulScalar(mlx.RMSNorm(k, 0, eps), inv)

	beta := mlx.Sigmoid(b)
	g := mlx.Exp(mlx.Multiply(mlx.Negative(mlx.Exp(mlx.Astype(w.aLog, mlx.Float32))), mlx.Softplus(mlx.Add(a, w.dtBias))))

	state := cache.state[i]
	if state == 0 {
		state = mlx.Zeros(mlx.Float32, B, Hv, Dv, Dk)
	}
	var y mlx.Array
	if Dk%32 == 0 && !m.OpsDeltaNet {
		y, state = gatedDeltaFast(q, k, v, g, beta, state, 0)
	} else {
		y, state = gatedDelta(q, k, v, g, beta, state)
	}
	cache.state[i] = state

	// RMSNormGated: rms_norm(y) * silu(z), the product in float32.
	n := mlx.RMSNorm(y, w.norm, c.RMSNormEps)
	o := mlx.Astype(mlx.Multiply(mlx.SiLU(mlx.Astype(z, mlx.Float32)), mlx.Astype(n, mlx.Float32)), y.Dtype())
	return linear(mlx.Reshape(o, B, L, valDim), w.outProj)
}

// gatedDelta runs the gated delta rule one step at a time, as mlx-lm's
// gated_delta_ops: q, k [B, L, Hk, Dk], v [B, L, Hv, Dv], g, beta [B, L, Hv],
// state [B, Hv, Dv, Dk] float32.
func gatedDelta(q, k, v, g, beta, state mlx.Array) (mlx.Array, mlx.Array) {
	s := q.Shape()
	B, L, Hk := s[0], s[1], s[2]
	vs := v.Shape()
	Hv, Dv, Dk := vs[2], vs[3], s[3]
	if r := Hv / Hk; r > 1 {
		q, k = mlx.Repeat(q, r, 2), mlx.Repeat(k, r, 2)
	}
	at := func(a mlx.Array, t int) mlx.Array {
		sh := a.Shape()
		start := make([]int, len(sh))
		stop := append([]int(nil), sh...)
		start[1], stop[1] = t, t+1
		return mlx.Squeeze(mlx.Slice(a, start, stop), 1)
	}
	ys := make([]mlx.Array, L)
	for t := 0; t < L; t++ {
		qt, kt, vt := at(q, t), at(k, t), at(v, t) // [B, H, D]
		gt, bt := at(g, t), at(beta, t)            // [B, H]
		state = mlx.Multiply(state, mlx.Reshape(gt, B, Hv, 1, 1))
		kRow := mlx.Reshape(kt, B, Hv, 1, Dk)
		kvMem := mlx.Sum(mlx.Multiply(state, kRow), -1, false)                    // [B, H, Dv]
		delta := mlx.Multiply(mlx.Subtract(vt, kvMem), mlx.Reshape(bt, B, Hv, 1)) // [B, H, Dv]
		state = mlx.Add(state, mlx.Multiply(kRow, mlx.Reshape(delta, B, Hv, Dv, 1)))
		y := mlx.Sum(mlx.Multiply(state, mlx.Reshape(qt, B, Hv, 1, Dk)), -1, false)
		ys[t] = mlx.Astype(y, q.Dtype())
	}
	return mlx.Stack(1, ys...), state
}

func (m *Qwen35) attention(w *fullAttn, x mlx.Array, cache *Qwen35Cache, i int) mlx.Array {
	c := m.Config
	s := x.Shape()
	B, L := s[0], s[1]
	H, Hkv, D := c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim

	qg := mlx.Split(mlx.Reshape(linear(x, w.q), B, L, H, 2*D), -1, D)
	q, gate := qg[0], mlx.Reshape(qg[1], B, L, H*D)
	k := mlx.Reshape(linear(x, w.k), B, L, Hkv, D)
	v := mlx.Reshape(linear(x, w.v), B, L, Hkv, D)

	q = mlx.Transpose(mlx.RMSNorm(q, w.qNorm, c.RMSNormEps), 0, 2, 1, 3)
	k = mlx.Transpose(mlx.RMSNorm(k, w.kNorm, c.RMSNormEps), 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	dims := int(float32(D) * c.RopeParameters.PartialRotaryFactor)
	q = mlx.RoPE(q, dims, false, c.RopeParameters.RopeTheta, 1, cache.Offset)
	k = mlx.RoPE(k, dims, false, c.RopeParameters.RopeTheta, 1, cache.Offset)
	prev, end := cache.Offset, cache.Offset+L
	for _, kv := range []struct {
		buf *mlx.Array
		new mlx.Array
	}{{&cache.keys[i], k}, {&cache.vals[i], v}} {
		if *kv.buf == 0 || (*kv.buf).Shape()[2] < end {
			grown := mlx.Zeros(kv.new.Dtype(), B, Hkv, (end+kvStep-1)/kvStep*kvStep, D)
			if *kv.buf != 0 && prev > 0 {
				grown = mlx.SliceUpdate(grown, mlx.Slice(*kv.buf, []int{0, 0, 0, 0}, []int{B, Hkv, prev, D}),
					[]int{0, 0, 0, 0}, []int{B, Hkv, prev, D})
			}
			*kv.buf = grown
		}
		*kv.buf = mlx.SliceUpdate(*kv.buf, kv.new, []int{0, 0, prev, 0}, []int{B, Hkv, end, D})
	}
	k = mlx.Slice(cache.keys[i], []int{0, 0, 0, 0}, []int{B, Hkv, end, D})
	v = mlx.Slice(cache.vals[i], []int{0, 0, 0, 0}, []int{B, Hkv, end, D})

	mode := ""
	if L > 1 {
		mode = "causal"
	}
	o := mlx.SDPA(q, k, v, float32(1/math.Sqrt(float64(D))), mode, 0)
	o = mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), B, L, H*D)
	return linear(mlx.Multiply(o, mlx.Sigmoid(gate)), w.o)
}
