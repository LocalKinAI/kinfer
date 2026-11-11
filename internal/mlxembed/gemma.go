// Package mlxembed serves text embedding models on MLX that llama.cpp cannot
// read: EmbeddingGemma 2 (embeddinggemma-2 in Ollama's library), which Ollama
// keeps as safetensors quantized to NVFP4, not as a GGUF.
//
// The model is Gemma 4's text stack run as an encoder: every layer attends both
// ways, sliding layers over a band of ±(sliding_window+1) tokens and full ones
// over everything; each layer gets an input of its own projected from the
// token embeddings (per-layer embeddings, projection only); the last hidden
// state is normed and projected to the embedding's width per token, then
// averaged over the tokens and scaled to length 1 — sentence-transformers'
// Pooling (mean) and Normalize modules. Written after Ollama's
// mlxrunner/model/gemma4embedding, whose vectors these are measured against.
// The audio and vision towers the release carries are not loaded: text only.
package mlxembed

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/mlxmodel"
	"github.com/LocalKinAI/kinfer/internal/tokenizer"
)

const architecture = "EmbeddingGemma2Model"

// IsModelDir reports a directory holding a model this package serves.
func IsModelDir(dir string) bool {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		Architectures []string `json:"architectures"`
	}
	return json.Unmarshal(b, &c) == nil && slices.Contains(c.Architectures, architecture)
}

type textConfig struct {
	HiddenSize         int      `json:"hidden_size"`
	Layers             int      `json:"num_hidden_layers"`
	Heads              int      `json:"num_attention_heads"`
	KVHeads            int      `json:"num_key_value_heads"`
	HeadDim            int      `json:"head_dim"`
	Eps                float32  `json:"rms_norm_eps"`
	SlidingWindow      int      `json:"sliding_window"`
	LayerTypes         []string `json:"layer_types"`
	PerLayerInput      int      `json:"hidden_size_per_layer_input"`
	VocabPerLayerInput int      `json:"vocab_size_per_layer_input"`
	EmbeddingDim       int      `json:"embedding_dim"`
	VocabSize          int      `json:"vocab_size"`
	BOS                int32    `json:"bos_token_id"`
	EOS                int32    `json:"eos_token_id"`
	PerLayer           map[string]struct {
		HeadDim int `json:"head_dim"`
		KVHeads int `json:"num_key_value_heads"`
	} `json:"per_layer_config"`
	Rope map[string]struct {
		Theta   float32 `json:"rope_theta"`
		Type    string  `json:"rope_type"`
		Partial float32 `json:"partial_rotary_factor"`
	} `json:"rope_parameters"`
}

// Model is one loaded EmbeddingGemma 2.
type Model struct {
	cfg textConfig
	tok *tokenizer.Tokenizer

	embed    mlxmodel.Linear // [vocab, hidden], looked up by rows
	pleProj  mlxmodel.Linear // hidden → layers × per-layer input
	pleNorm  mlx.Array
	layers   []layer
	norm     mlx.Array
	proj     mlxmodel.Linear // hidden → embedding_dim
	baseFull float32
	baseSlid float32

	all []mlx.Array
}

type layer struct {
	q, k, v, o          mlxmodel.Linear
	qNorm, kNorm        mlx.Array
	in, postAttn        mlx.Array
	preFF, postFF       mlx.Array
	gate, up, down      mlxmodel.Linear
	pleGate, plePro     mlxmodel.Linear
	plePost             mlx.Array
	scalar              mlx.Array
	sliding             bool
	headDim, kvHeads    int
	hasPLE, hasPLENorms bool
}

// towers are the parts of the release this package does not read.
var towers = []string{"audio_tower", "vision_tower", "embed_audio", "embed_vision"}

func isTower(name string) bool {
	name = strings.TrimPrefix(name, "model.")
	for _, t := range towers {
		if strings.HasPrefix(name, t) {
			return true
		}
	}
	return false
}

// Load reads the model in dir. mlx.Open must have been called.
func Load(dir string) (m *Model, err error) {
	defer mlx.Recover(&err)
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c struct {
		TextConfig textConfig `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("%s/config.json: %w", dir, err)
	}
	cfg := c.TextConfig
	if cfg.HeadDim == 0 {
		cfg.HeadDim = 256
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-6
	}
	switch {
	case cfg.HiddenSize == 0 || cfg.Layers == 0 || cfg.Heads == 0 || cfg.KVHeads == 0 || cfg.EmbeddingDim == 0:
		return nil, fmt.Errorf("%s/config.json: text_config lacks the sizes an EmbeddingGemma 2 has", dir)
	case cfg.VocabPerLayerInput > 0:
		// The token half of per-layer embeddings: none of the releases has it,
		// and without it the vectors would be quietly wrong.
		return nil, fmt.Errorf("%s: per-layer embeddings looked up by token (vocab_size_per_layer_input %d) are not supported", dir, cfg.VocabPerLayerInput)
	}
	for kind, r := range cfg.Rope {
		if (r.Type != "" && r.Type != "default") || (r.Partial != 0 && r.Partial != 1) {
			return nil, fmt.Errorf("%s: %s rotary embedding %q (partial %g) is not supported", dir, kind, r.Type, r.Partial)
		}
	}

	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}

	files, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	t := map[string]mlx.Array{}
	for _, f := range files {
		// Ollama keeps a tensor to a file, named for it: the towers are
		// left on the disk.
		if isTower(strings.TrimSuffix(filepath.Base(f), ".safetensors")) {
			continue
		}
		got, err := mlx.LoadSafetensors(f)
		if err != nil {
			return nil, err
		}
		for k, v := range got {
			if isTower(k) {
				v.Free()
				continue
			}
			t[k] = v
		}
	}
	m = &Model{cfg: cfg, tok: tok, baseFull: 1e6, baseSlid: 1e4}
	if r, ok := cfg.Rope["full_attention"]; ok && r.Theta > 0 {
		m.baseFull = r.Theta
	}
	if r, ok := cfg.Rope["sliding_attention"]; ok && r.Theta > 0 {
		m.baseSlid = r.Theta
	}
	for _, a := range t {
		m.all = append(m.all, a)
	}
	defer func() {
		if err != nil {
			m.Free()
			m = nil
		}
	}()

	prefix := ""
	for _, p := range []string{"language_model.", "model.language_model."} {
		if _, ok := t[p+"embed_tokens.weight"]; ok {
			prefix = p
			break
		}
	}
	if prefix == "" {
		return nil, fmt.Errorf("%s: no language_model.embed_tokens.weight", dir)
	}
	missing := ""
	arr := func(k string) mlx.Array {
		a, ok := t[prefix+k]
		if !ok && missing == "" {
			missing = prefix + k
		}
		return a
	}
	lin := func(k string) mlxmodel.Linear {
		wk := prefix + k + ".weight"
		w, ok := t[wk]
		if !ok {
			if missing == "" {
				missing = wk
			}
			return mlxmodel.Linear{}
		}
		l := mlxmodel.Linear{W: w}
		// NVFP4 as ModelOpt exports it and Ollama stores it; see
		// mlxmodel.Linear.
		if s, ok := t[wk+".scale"]; ok {
			l.Scales, l.Group, l.Bits, l.Mode = s, 16, 4, "nvfp4"
			if g, ok := t[wk+".global_scale"]; ok {
				v := g.Float32s()
				if len(v) != 1 {
					panic(&mlx.Error{Msg: fmt.Sprintf("%s has %d global scales", wk, len(v))})
				}
				l.Global = v[0]
			}
		} else if _, ok := t[prefix+k+".scales"]; ok {
			// MLX's affine quantization: no release of this model ships it.
			panic(&mlx.Error{Msg: fmt.Sprintf("%s is quantized the MLX way, which is not supported here", wk)})
		}
		return l
	}

	m.embed = lin("embed_tokens")
	m.norm = arr("norm.weight")
	m.proj = lin("embedding_projection")
	if cfg.PerLayerInput > 0 {
		m.pleProj = lin("ple.per_layer_model_projection")
		m.pleNorm = arr("ple.per_layer_projection_norm.weight")
	}
	for i := 0; i < cfg.Layers; i++ {
		p := fmt.Sprintf("layers.%d.", i)
		l := layer{
			q: lin(p + "self_attn.q_proj"), k: lin(p + "self_attn.k_proj"),
			v: lin(p + "self_attn.v_proj"), o: lin(p + "self_attn.o_proj"),
			qNorm: arr(p + "self_attn.q_norm.weight"), kNorm: arr(p + "self_attn.k_norm.weight"),
			in: arr(p + "input_layernorm.weight"), postAttn: arr(p + "post_attention_layernorm.weight"),
			preFF: arr(p + "pre_feedforward_layernorm.weight"), postFF: arr(p + "post_feedforward_layernorm.weight"),
			gate: lin(p + "mlp.gate_proj"), up: lin(p + "mlp.up_proj"), down: lin(p + "mlp.down_proj"),
			scalar:  arr(p + "layer_scalar"),
			sliding: i < len(cfg.LayerTypes) && cfg.LayerTypes[i] == "sliding_attention",
			headDim: cfg.HeadDim, kvHeads: cfg.KVHeads,
		}
		if o, ok := cfg.PerLayer[fmt.Sprintf("%02d", i)]; ok {
			if o.HeadDim > 0 {
				l.headDim = o.HeadDim
			}
			if o.KVHeads > 0 {
				l.kvHeads = o.KVHeads
			}
		}
		if cfg.PerLayerInput > 0 {
			l.hasPLE = true
			l.pleGate = lin(p + "ple_block.per_layer_input_gate")
			l.plePro = lin(p + "ple_block.per_layer_projection")
			l.plePost = arr(p + "ple_block.post_per_layer_input_norm.weight")
		}
		m.layers = append(m.layers, l)
	}
	if missing != "" {
		return nil, fmt.Errorf("%s: no tensor %s", dir, missing)
	}
	if sh := m.embed.W.Shape(); len(sh) != 2 || (m.embed.Scales == 0 && sh[1] != cfg.HiddenSize) {
		return nil, fmt.Errorf("%s: embed_tokens is %v, want [vocab, %d]", dir, sh, cfg.HiddenSize)
	}
	return m, nil
}

// Free releases the weights.
func (m *Model) Free() {
	for _, a := range m.all {
		a.Free()
	}
	m.all = nil
}

// Tokens is text as the model reads it: <bos>, the text, <eos>, which is what
// the release's post-processor adds and what Ollama sends.
func (m *Model) Tokens(text string) []int32 {
	ids := append([]int32{m.cfg.BOS}, m.tok.Encode(text, true)...)
	return append(ids, m.cfg.EOS)
}

// VocabSize is the number of token ids the model reads.
func (m *Model) VocabSize() int { return max(m.cfg.VocabSize, m.tok.VocabSize()) }

// Embed is the vector of one token sequence, length 1.
func (m *Model) Embed(ids []int32) (vec []float32, err error) {
	defer mlx.Recover(&err)
	sc := mlx.NewScope()
	defer sc.Close()
	c := m.cfg
	n := len(ids)

	h := m.embed.Rows(mlx.FromInt32(ids, 1, n)) // [1, n, hidden]
	h = mlx.MulScalar(h, float32(math.Sqrt(float64(c.HiddenSize))))

	var ple mlx.Array
	if c.PerLayerInput > 0 {
		ple = m.pleProj.Apply(h)
		ple = mlx.MulScalar(ple, float32(1/math.Sqrt(float64(c.HiddenSize))))
		ple = mlx.Reshape(ple, 1, n, c.Layers, c.PerLayerInput)
		ple = mlx.RMSNorm(ple, m.pleNorm, c.Eps)
	}

	// Sliding layers see keys at most sliding_window+1 positions away, either
	// side; a sequence no longer than that needs no mask.
	var band mlx.Array
	if w := c.SlidingWindow + 1; c.SlidingWindow > 0 && n > w {
		pos := mlx.Arange(0, float64(n), 1, mlx.Int32)
		d := mlx.Abs(mlx.Subtract(mlx.ExpandDims(pos, 1), mlx.ExpandDims(pos, 0)))
		band = mlx.Less(d, mlx.ScalarAs(float32(w+1), mlx.Int32))
	}

	for i := range m.layers {
		var in mlx.Array
		if ple != 0 {
			in = mlx.Squeeze(mlx.Slice(ple, []int{0, 0, i, 0}, []int{1, n, i + 1, c.PerLayerInput}), 2)
		}
		h = m.layers[i].forward(h, in, band, m, n)
	}

	out := m.proj.Apply(mlx.RMSNorm(h, m.norm, c.Eps)) // [1, n, dim]
	pooled := mlx.Mean(mlx.Astype(mlx.Squeeze(out, 0), mlx.Float32), 0, false)
	length := mlx.Sqrt(mlx.Sum(mlx.Square(pooled), 0, true))
	pooled = mlx.Divide(pooled, mlx.Maximum(length, mlx.Scalar(1e-12)))
	pooled.Eval()
	return pooled.Float32s(), nil
}

func (l *layer) forward(x, ple, band mlx.Array, m *Model, n int) mlx.Array {
	eps := m.cfg.Eps
	a := l.attend(mlx.RMSNorm(x, l.in, eps), band, m, n)
	h := mlx.Add(x, mlx.RMSNorm(a, l.postAttn, eps))

	f := mlx.RMSNorm(h, l.preFF, eps)
	f = l.down.Apply(mlx.Multiply(geluTanh(l.gate.Apply(f)), l.up.Apply(f)))
	h = mlx.Add(h, mlx.RMSNorm(f, l.postFF, eps))

	if l.hasPLE && ple != 0 {
		g := mlx.Multiply(geluTanh(l.pleGate.Apply(h)), ple)
		h = mlx.Add(h, mlx.RMSNorm(l.plePro.Apply(g), l.plePost, eps))
	}
	return mlx.Multiply(h, l.scalar)
}

func (l *layer) attend(x, band mlx.Array, m *Model, n int) mlx.Array {
	c, eps := m.cfg, m.cfg.Eps
	heads := func(p mlxmodel.Linear, count int) mlx.Array {
		return mlx.Transpose(mlx.Reshape(p.Apply(x), 1, n, count, l.headDim), 0, 2, 1, 3)
	}
	q := mlx.RMSNorm(heads(l.q, c.Heads), l.qNorm, eps)
	k := mlx.RMSNorm(heads(l.k, l.kvHeads), l.kNorm, eps)
	v := mlx.RMSNorm(heads(l.v, l.kvHeads), 0, eps) // a norm with no weight

	base, mask := m.baseFull, mlx.Array(0)
	if l.sliding {
		base, mask = m.baseSlid, band
	}
	q = mlx.RoPE(q, l.headDim, false, base, 1, 0)
	k = mlx.RoPE(k, l.headDim, false, base, 1, 0)

	// The q and k norms keep the scores' size; there is no 1/√d.
	o := mlx.SDPA(q, k, v, 1, "", mask)
	return l.o.Apply(mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), 1, n, c.Heads*l.headDim))
}

// geluTanh is GELU's tanh approximation, in x's own dtype, as Ollama and
// mlx_lm compute it.
func geluTanh(x mlx.Array) mlx.Array {
	dt := x.Dtype()
	inner := mlx.Add(x, mlx.Multiply(mlx.ScalarAs(0.044715, dt), mlx.Multiply(mlx.Multiply(x, x), x)))
	t := mlx.Tanh(mlx.Multiply(mlx.ScalarAs(float32(math.Sqrt(2/math.Pi)), dt), inner))
	return mlx.Multiply(mlx.Multiply(mlx.ScalarAs(0.5, dt), x), mlx.Add(mlx.ScalarAs(1, dt), t))
}
