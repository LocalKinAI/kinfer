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
// of every four layers, full attention with an output gate in the fourth. The
// MoE models (qwen3_5_moe) have experts in place of each layer's MLP.
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
	NumExpertsPerTok      int     `json:"num_experts_per_tok"`
	NormTopkProb          *bool   `json:"norm_topk_prob"`
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
func Serves(modelType string) bool {
	switch modelType {
	case "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text":
		return true
	}
	return false
}

// ExpertsReadable says whether an MoE model whose tensors have these names
// keeps its experts as LoadQwen35 reads them, stacked as MLX's switch_mlp:
// mlx_lm.convert's files and Ollama's do, Hugging Face's own do not. No names
// is no answer, and no error.
func ExpertsReadable(names []string) error {
	others := false
	for _, n := range names {
		if strings.Contains(n, ".switch_mlp.") {
			return nil
		}
		others = others || strings.Contains(n, ".experts.")
	}
	if others {
		return fmt.Errorf("its experts are stored one by one or as gate_up_proj, as Hugging Face's own files keep them, which kinfer does not read; an MLX conversion of it (mlx_lm.convert, mlx-community's) keeps them stacked as switch_mlp, which it does")
	}
	return nil
}

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
	if c.NumExperts > 0 && (c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts) {
		return nil, fmt.Errorf("%d experts with %d a token", c.NumExperts, c.NumExpertsPerTok)
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	return c, nil
}

// Linear is a weight matrix [out, in], plain or quantized the way MLX's
// QuantizedLinear stores it: packed W, per-group Scales and, in affine mode,
// Biases. An NVFP4 weight as NVIDIA's ModelOpt exports it, and Ollama stores
// it, has a scale of its own besides, Global: the weight is W·Scales·Global.
type Linear struct {
	W, Scales, Biases mlx.Array
	Group, Bits       int
	Mode              string
	Global            float32
}

func (l Linear) quantized() bool { return l.Scales != 0 }

// global applies the weight's own scale to y, a product with it, in float32:
// rounded to y's bfloat16 first, the scale would be off by as much as 0.4% in
// every layer. MLX's dequantize takes a global scale too, but means by it the
// weight's largest value, 2688 times this one.
func (l Linear) global(y mlx.Array) mlx.Array {
	if l.Global == 0 {
		return y
	}
	return mlx.Astype(mlx.Multiply(y, mlx.ScalarAs(l.Global, mlx.Float32)), y.Dtype())
}

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
		return w.global(mlx.QuantizedMatmul(x, w.W, w.Scales, w.Biases, w.Group, w.Bits, w.Mode))
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
	return l.global(mlx.Dequantize(mlx.Take(l.W, ids, 0), mlx.Take(l.Scales, ids, 0), b, l.Group, l.Bits, l.Mode))
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
	moe            *moeBlock // in place of gate, up and down in an MoE model
}

// moeBlock is an MoE model's MLP: for each token a router picks k of the
// experts, whose SwiGLUs' outputs are summed by the router's weights for
// them, and one more, the shared expert, every token goes through, its output
// scaled by a sigmoid gate of its own.
type moeBlock struct {
	router, sharedGate Linear
	gate, up, down     Linear // the experts', stacked: [experts, out, in]
	sGate, sUp, sDown  Linear // the shared expert's
	k, experts         int
	normed             bool // the k weights scaled to sum to one
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
	mtp    *mtpHead    // nil when the model has none
	all    []mlx.Array // every weight, for Free
}

// mtpHead is a model's multi-token-prediction head, as Qwen3.5 ships it: the
// embedding of a token and the model's hidden state at the position before it
// — each RMS-normed, side by side — projected back to the hidden size, then
// one full-attention decoder block with a cache of its own, a norm, and the
// model's own output head. What comes out is the token after the next, which
// the model then checks in one pass with the token it is sure of.
type mtpHead struct {
	fc                 Linear
	enorm, hnorm, norm mlx.Array
	block              qwen35Layer
}

// CanDraft reports whether the model carries an MTP head. mlx-lm's
// conversions drop it; Hugging Face's releases and Ollama's keep it.
func (m *Qwen35) CanDraft() bool { return m.mtp != nil }

func (m *Qwen35) head() Linear {
	if m.lmHead != nil {
		return *m.lmHead
	}
	return m.embed
}

// LoadQwen35 reads config.json and the safetensors files in dir: a Hugging
// Face release, or an MLX conversion of one (mlx_lm.convert), quantized or not,
// or one quantized to NVFP4 as Ollama keeps it.
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
		// NVFP4 as ModelOpt exports it and Ollama stores it: 4-bit floats
		// eight to a uint32, an FP8 scale for each 16 of them, and one scale
		// for the whole weight.
		if s, ok := raw[wk+".scale"]; ok {
			ws, ss := l.W.Shape(), s.Shape()
			if n := len(ws); l.W.Dtype() != mlx.Uint32 || s.Dtype() != mlx.Uint8 || n < 2 || len(ss) != n || ss[n-1]*16 != ws[n-1]*8 {
				panic(&mlx.Error{Msg: fmt.Sprintf("%s is %v %v with a %v %v scale, not NVFP4", wk, l.W.Dtype(), ws, s.Dtype(), ss)})
			}
			l.Scales, l.Group, l.Bits, l.Mode = take(wk+".scale"), 16, 4, "nvfp4"
			if g, ok := raw[wk+".global_scale"]; ok {
				v := g.Float32s()
				if len(v) != 1 {
					panic(&mlx.Error{Msg: fmt.Sprintf("%s has %d global scales; one for the whole weight is what is read", wk, len(v))})
				}
				l.Global = v[0]
			}
			return l
		}
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

	// mlp reads a layer's MLP at p: a SwiGLU, or in an MoE model its experts,
	// stacked as MLX's SwitchGLU keeps them (mlx_lm.convert's and Ollama's
	// files; Hugging Face's own keep them otherwise).
	mlp := func(p string) (l qwen35Layer) {
		if _, ok := key(p + "switch_mlp.gate_proj.weight"); !ok {
			if _, dense := key(p + "gate_proj.weight"); !dense && c.NumExperts > 0 {
				panic(&mlx.Error{Msg: "the experts at " + p + " are not stacked as MLX's switch_mlp; an MLX conversion of the model (mlx_lm.convert) is"})
			}
			l.gate, l.up, l.down = lin(p+"gate_proj"), lin(p+"up_proj"), lin(p+"down_proj")
			return l
		}
		l.moe = &moeBlock{
			router: lin(p + "gate"), sharedGate: lin(p + "shared_expert_gate"),
			gate: lin(p + "switch_mlp.gate_proj"), up: lin(p + "switch_mlp.up_proj"), down: lin(p + "switch_mlp.down_proj"),
			sGate: lin(p + "shared_expert.gate_proj"), sUp: lin(p + "shared_expert.up_proj"), sDown: lin(p + "shared_expert.down_proj"),
			k: c.NumExpertsPerTok, experts: c.NumExperts, normed: c.NormTopkProb == nil || *c.NormTopkProb,
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
		l := mlp(p + "mlp.")
		l.inNorm = centred(p + "input_layernorm.weight")
		l.inPost = centred(p + "post_attention_layernorm.weight")
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
	if _, ok := key("mtp.fc.weight"); ok {
		a := "mtp.layers.0.self_attn."
		block := mlp("mtp.layers.0.mlp.")
		block.inNorm = centred("mtp.layers.0.input_layernorm.weight")
		block.inPost = centred("mtp.layers.0.post_attention_layernorm.weight")
		block.full = &fullAttn{
			q: lin(a + "q_proj"), k: lin(a + "k_proj"), v: lin(a + "v_proj"), o: lin(a + "o_proj"),
			qNorm: centred(a + "q_norm.weight"), kNorm: centred(a + "k_norm.weight"),
		}
		m.mtp = &mtpHead{
			fc:    lin("mtp.fc"),
			enorm: centred("mtp.pre_fc_norm_embedding.weight"), hnorm: centred("mtp.pre_fc_norm_hidden.weight"),
			norm:  centred("mtp.norm.weight"),
			block: block,
		}
	}
	mlx.Eval(m.all...)
	return m, nil
}

// Bytes is the size of the weights as loaded.
func (m *Qwen35) Bytes() int64 {
	var n int64
	for _, a := range m.all {
		n += int64(a.Nbytes())
	}
	return n
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

	// Drafting, when true, has every pass also feed the MTP head, so it can
	// propose the tokens after the next (Draft). The head keeps a key and
	// value cache of its own, at headOffset, and headH is the model's normed
	// hidden state at the last position read, which the next token is paired
	// with; 0 before the first.
	Drafting           bool
	headKeys, headVals mlx.Array
	headOffset         int
	headH              mlx.Array

	// steps and convIns are, after Verify, each DeltaNet layer's state after
	// every token of the pass and the convolution inputs it read, from which
	// Keep cuts the cache back to the last token the reply kept.
	keepSteps      bool
	steps, convIns []mlx.Array
}

const kvStep = 256

// NewCache makes an empty cache for one sequence.
func (m *Qwen35) NewCache() *Qwen35Cache {
	n := len(m.layers)
	return &Qwen35Cache{conv: make([]mlx.Array, n), state: make([]mlx.Array, n),
		keys: make([]mlx.Array, n), vals: make([]mlx.Array, n), steps: make([]mlx.Array, n), convIns: make([]mlx.Array, n)}
}

// Arrays are the cache's current arrays, to evaluate along with a Step's
// logits.
func (c *Qwen35Cache) Arrays() []mlx.Array {
	var r []mlx.Array
	for _, set := range [][]mlx.Array{c.conv, c.state, c.keys, c.vals, c.steps, c.convIns, {c.headKeys, c.headVals, c.headH}} {
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
	d := &Qwen35Cache{Offset: c.Offset, Drafting: c.Drafting, headOffset: c.headOffset,
		steps: make([]mlx.Array, len(c.conv)), convIns: make([]mlx.Array, len(c.conv))}
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
	for _, p := range []struct{ dst, src *mlx.Array }{{&d.headKeys, &c.headKeys}, {&d.headVals, &c.headVals}, {&d.headH, &c.headH}} {
		if *p.src != 0 {
			*p.dst = (*p.src).Ref()
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
		keys: make([]mlx.Array, len(c.conv)), vals: make([]mlx.Array, len(c.conv)),
		steps: make([]mlx.Array, len(c.conv)), convIns: make([]mlx.Array, len(c.conv)), Drafting: c.Drafting}
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
	logits, _ := m.step(tokens, cache, trace, false)
	return logits
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
	logits, _ = m.step(ids, cache, nil, true)
	ids.Free()
	mlx.Eval(append([]mlx.Array{logits}, cache.Arrays()...)...)
	return logits, nil
}

// Verify builds the pass that reads tok — the token the reply is sure of —
// and the drafts after it, [1, n] as Draft gave them or 0, for one sequence.
// It returns, unevaluated, the logits after each token, [n+1, vocab], and the
// model's normed hidden state after each, [n+1, hidden], from which the head
// drafts on (Keep); the caller evaluates them with the cache's Arrays. Built
// while the drafts are still being worked out, the pass costs no time of its
// own between them and its evaluation.
func (m *Qwen35) Verify(tok int32, drafts mlx.Array, cache *Qwen35Cache) (logits, hidden mlx.Array, err error) {
	defer mlx.Recover(&err)
	ids := mlx.FromInt32([]int32{tok}, 1, 1)
	if drafts != 0 {
		all := mlx.Concatenate(1, ids, drafts)
		ids.Free()
		ids = all
	}
	cache.keepSteps = true
	logits, hidden = m.step(ids, cache, nil, true)
	cache.keepSteps = false
	ids.Free()
	return logits, hidden, nil
}

// step records a pass and returns the logits — of every position when every,
// of the last otherwise — and, when every, the normed hidden states they came
// from, [B*L, hidden]; 0 otherwise.
func (m *Qwen35) step(tokens mlx.Array, cache *Qwen35Cache, trace func(layer int, h mlx.Array), every bool) (mlx.Array, mlx.Array) {
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
		h = mlx.Add(h, l.mlp(x))
		if trace != nil {
			trace(i, h)
		}
	}
	H, rows := m.Config.HiddenSize, B*L
	if m.mtp != nil && cache.Drafting {
		// The head reads every position, so every position is normed.
		all := mlx.RMSNorm(h, m.norm, m.Config.RMSNormEps)
		m.feedHead(tokens, all, cache)
		h = all
		if !every {
			h, rows = mlx.Slice(all, []int{0, L - 1, 0}, []int{B, L, H}), B
		}
	} else {
		if !every {
			h, rows = mlx.Slice(h, []int{0, L - 1, 0}, []int{B, L, H}), B
		}
		h = mlx.RMSNorm(h, m.norm, m.Config.RMSNormEps)
	}
	logits := mlx.Reshape(linear(h, m.head()), rows, m.Config.VocabSize)
	var hidden mlx.Array
	if every {
		hidden = mlx.Reshape(h, rows, H)
		scope.Keep(hidden)
	}

	cache.Offset += L
	scope.Keep(logits)
	scope.Escape(cache.Arrays()...)
	// The graph holds what it still needs of the old cache; our handles to it
	// can go now, which also lets MLX update the KV buffers in place.
	for _, a := range old {
		a.Free()
	}
	return logits, hidden
}

// feedHead reads into the MTP head each token of a pass paired with the
// model's normed hidden state at the position before it: the pass's own for
// all but the first, which takes the last of the pass before — zeros at the
// very start, as llama.cpp's draft context has it. The head is then where the
// model is, and drafts from the last position's state.
func (m *Qwen35) feedHead(tokens, all mlx.Array, cache *Qwen35Cache) {
	s := all.Shape()
	B, L, H := s[0], s[1], s[2]
	before := cache.headH
	if before == 0 {
		before = mlx.Zeros(all.Dtype(), B, 1, H)
	}
	if L > 1 {
		before = mlx.Concatenate(1, before, mlx.Slice(all, []int{0, 0, 0}, []int{B, L - 1, H}))
	}
	m.headRead(tokens, before, cache)
	cache.headH = mlx.Contiguous(mlx.Slice(all, []int{0, L - 1, 0}, []int{B, L, H}))
}

// headRead is one pass of the MTP head over tokens [B, L] and the hidden
// states paired with them, [B, L, hidden], at the head's own positions; it
// returns the head's normed output, [B, L, hidden], from which both its
// logits and its next draft's paired state come.
func (m *Qwen35) headRead(tokens, before mlx.Array, cache *Qwen35Cache) mlx.Array {
	t, eps := m.mtp, m.Config.RMSNormEps
	e := m.embed.rows(tokens)
	x := linear(mlx.Concatenate(-1, mlx.RMSNorm(e, t.enorm, eps), mlx.RMSNorm(before, t.hnorm, eps)), t.fc)
	kv := &Qwen35Cache{keys: []mlx.Array{cache.headKeys}, vals: []mlx.Array{cache.headVals}, Offset: cache.headOffset}
	x = mlx.Add(x, m.attention(t.block.full, mlx.RMSNorm(x, t.block.inNorm, eps), kv, 0))
	cache.headKeys, cache.headVals = kv.keys[0], kv.vals[0]
	y := mlx.RMSNorm(x, t.block.inPost, eps)
	x = mlx.Add(x, t.block.mlp(y))
	cache.headOffset += tokens.Shape()[1]
	return mlx.RMSNorm(x, t.norm, eps)
}

// Draft proposes the n tokens after tok, each the head's most likely, for a
// sequence the model has read up to the token before tok: the head pairs
// tok with the model's state there, then each draft with its own output. It
// starts working them out and returns them before they are, [1, n], for the
// pass that checks them (Verify) to be built meanwhile; 0 when it has none.
// The caller frees them. The cache is left as it was — the head's positions
// written while drafting are written again by the pass that checks them.
func (m *Qwen35) Draft(cache *Qwen35Cache, tok int32, n int) (drafts mlx.Array, err error) {
	defer mlx.Recover(&err)
	if m.mtp == nil || !cache.Drafting || cache.headH == 0 || n <= 0 {
		return 0, nil
	}
	scope := mlx.NewScope()
	defer scope.Close()
	keys, vals, off := cache.headKeys, cache.headVals, cache.headOffset
	t, h := mlx.FromInt32([]int32{tok}, 1, 1), cache.headH
	toks := make([]mlx.Array, n)
	for i := range n {
		h = m.headRead(t, h, cache)
		t = mlx.Astype(mlx.Argmax(linear(h, m.head()), -1, false), mlx.Int32)
		toks[i] = t
	}
	drafts = mlx.Concatenate(1, toks...)
	scope.Escape(drafts)
	mlx.AsyncEval(drafts)
	// The buffers the drafts are written into belong to their graph; the
	// cache keeps the ones it had, which are not written in place while
	// these handles hold them.
	cache.headKeys, cache.headVals, cache.headOffset = keys, vals, off
	return drafts, nil
}

// Mark is where a cache was before a pass that checks drafts.
type Mark struct{ offset, headOffset int }

// Mark marks where c is now.
func (c *Qwen35Cache) Mark() Mark { return Mark{c.Offset, c.headOffset} }

// Keep takes back what the Verify pass after mk read past its i-th token, the
// last the reply kept. The DeltaNet layers get their state and convolution
// inputs as they were after it, which the pass left for this; the attention
// layers and the head go back to just after it, their buffers written again
// from there; and the head drafts from the model's state after it, hidden's
// i-th row. With every token kept there is nothing to take back. What it
// takes back is left unevaluated, for the next pass to work out with its own.
func (c *Qwen35Cache) Keep(mk Mark, i int, hidden mlx.Array) (err error) {
	defer mlx.Recover(&err)
	read := c.Offset - mk.offset
	if i < read-1 {
		scope := mlx.NewScope()
		defer scope.Close()
		for l, st := range c.steps {
			if st == 0 {
				continue
			}
			s := st.Shape()
			c.state[l].Free()
			c.state[l] = mlx.Contiguous(mlx.Reshape(mlx.Slice(st, []int{0, i, 0, 0, 0}, []int{s[0], i + 1, s[2], s[3], s[4]}), s[0], s[2], s[3], s[4]))
			in := c.convIns[l].Shape()
			K := in[1] - read + 1
			c.conv[l].Free()
			c.conv[l] = mlx.Contiguous(mlx.Slice(c.convIns[l], []int{0, i + 1, 0}, []int{in[0], i + K, in[2]}))
		}
		H := hidden.Shape()[1]
		if c.headH != 0 {
			c.headH.Free()
		}
		c.headH = mlx.Reshape(mlx.Contiguous(mlx.Slice(hidden, []int{i, 0}, []int{i + 1, H})), 1, 1, H)
		kept := []mlx.Array{c.headH}
		for l, st := range c.steps {
			if st != 0 {
				kept = append(kept, c.state[l], c.conv[l])
			}
		}
		scope.Escape(kept...)
		c.Offset, c.headOffset = mk.offset+i+1, mk.headOffset+i+1
	}
	for l := range c.steps {
		for _, a := range []*mlx.Array{&c.steps[l], &c.convIns[l]} {
			if *a != 0 {
				(*a).Free()
				*a = 0
			}
		}
	}
	return nil
}

// HeadReady reports whether c's head has read everything the model has, so
// that drafting from it is drafting from the whole conversation: false for a
// cache whose prompt was read, or pooled, without it.
func (c *Qwen35Cache) HeadReady() bool {
	return c.Drafting && c.headH != 0 && c.headOffset == c.Offset
}

// mlp is the layer's feed-forward part, x [B, L, hidden].
func (l *qwen35Layer) mlp(x mlx.Array) mlx.Array {
	if l.moe != nil {
		return l.moe.forward(x)
	}
	return swiGLU(x, l.gate, l.up, l.down)
}

func swiGLU(x mlx.Array, gate, up, down Linear) mlx.Array {
	return linear(mlx.Multiply(mlx.SiLU(linear(x, gate)), linear(x, up)), down)
}

// sortAt is how many token-expert pairs a pass has before they are sorted by
// expert, as mlx-lm's SwitchGLU sorts them, so that each expert's weights
// are read once for all the rows that go through it.
const sortAt = 64

// forward is mlx-lm's Qwen3NextSparseMoeBlock: the router's softmax in
// float32, the top k, and the SwitchGLU of those experts.
func (e *moeBlock) forward(x mlx.Array) mlx.Array {
	s := x.Shape()
	B, L, D := s[0], s[1], s[2]
	N, k := B*L, e.k
	probs := mlx.Softmax(mlx.Astype(linear(x, e.router), mlx.Float32), -1)
	inds := mlx.Slice(mlx.Argpartition(probs, e.experts-k, -1), []int{0, 0, e.experts - k}, []int{B, L, e.experts})
	w := mlx.TakeAlongAxis(probs, inds, -1)
	if e.normed {
		w = mlx.Divide(w, mlx.Sum(w, -1, true))
	}
	var y mlx.Array
	if N*k >= sortAt {
		flat := mlx.Reshape(inds, N*k)
		order := mlx.Argsort(flat, 0)
		rows := mlx.FloorDivide(order, mlx.ScalarAs(float32(k), order.Dtype()))
		y = e.glu(mlx.Take(mlx.Reshape(x, N, 1, D), rows, 0), mlx.Take(flat, order, 0), true) // [N*k, 1, D]
		y = mlx.Reshape(mlx.Take(y, mlx.Argsort(order, 0), 0), B, L, k, D)
	} else {
		y = mlx.Squeeze(e.glu(mlx.ExpandDims(mlx.ExpandDims(x, 2), 2), inds, false), 3) // [B, L, k, 1, D] → [B, L, k, D]
	}
	y = mlx.Sum(mlx.Multiply(y, mlx.ExpandDims(mlx.Astype(w, y.Dtype()), 3)), 2, false)
	shared := mlx.Multiply(mlx.Sigmoid(linear(x, e.sharedGate)), swiGLU(x, e.sGate, e.sUp, e.sDown))
	return mlx.Add(y, shared)
}

// glu runs each of x's rows, [..., M, hidden], through the expert idx names
// for it; sorted when idx is in ascending order.
func (e *moeBlock) glu(x, idx mlx.Array, sorted bool) mlx.Array {
	g, u := gather(x, e.gate, idx, sorted), gather(x, e.up, idx, sorted)
	return gather(mlx.Multiply(mlx.SiLU(g), u), e.down, idx, sorted)
}

func gather(x mlx.Array, w Linear, idx mlx.Array, sorted bool) mlx.Array {
	if w.quantized() {
		return w.global(mlx.GatherQMM(x, w.W, w.Scales, w.Biases, idx, w.Group, w.Bits, w.Mode, sorted))
	}
	return mlx.GatherMM(x, mlx.Swapaxes(w.W, 1, 2), idx, sorted)
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
	switch {
	case cache.keepSteps:
		y, state, cache.steps[i] = gatedDeltaSteps(q, k, v, g, beta, state)
		cache.convIns[i] = convIn
	case Dk%32 == 0 && !m.OpsDeltaNet:
		y, state = gatedDeltaFast(q, k, v, g, beta, state, 0)
	default:
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
