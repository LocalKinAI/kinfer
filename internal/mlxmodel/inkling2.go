package mlxmodel

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// Inkling2 is Thinking Machines' Inkling 2 Mini, the text model: a looped
// transformer whose feed-forward layers are mixtures of experts.
//
// What is unlike the other models here:
//
//   - The middle layers run several times with the same weights (loop): 24
//     layers of weights are 48 layer applications a token. A layer's later
//     passes also read what the pass before left (messages), through gates.
//   - There is no single residual stream. Every sixth layer closes a block,
//     and each layer's inputs are a mixture of the blocks so far and the one
//     being written, weighted per token by learned queries (mix).
//   - Attention has no rotary positions: a query adds a bias of its own for
//     each distance back, up to 1,024 tokens, and a sink it may attend to
//     instead of any token. Most layers see only the last 1,024 tokens; every
//     sixth sees everything.
//   - Keys, values and each layer's two outputs pass through a convolution
//     over the last four tokens.
//
// The structure follows the checkpoint as Thinking Machines publishes it and
// as Ollama's MLX runner reads it; this is not Ollama's code. The experts are
// stored in NVFP4 or MXFP8 and are multiplied as stored. The publisher's
// reference also rounds each expert's input to NVFP4, as the hardware the
// checkpoint was made for does; here the inputs stay as they are.
type Inkling2 struct {
	Config *Inkling2Config
	plan   []inkVisit

	// DenseBits is the bits the attention's projections and the output
	// layer were packed to when the model was loaded: 0 for as published.
	DenseBits int
	unembedQ  Linear

	// eyes is the vision tower, nil when the model has none; spans is the
	// pictures of the pass being recorded (ForwardSpans).
	eyes  *inkEyes
	spans []Inkling2Span

	embed, unembed          mlx.Array
	embedNorm, outNorm      mlx.Array
	outQuery                mlx.Array
	layers                  []inkLayer
	all                     []mlx.Array
	positions               map[int]mlx.Array
	bytes                   int64
	logitCap, routeScale    float32
	gateCap, upCap          float32
	normEps                 float32
	hidden, experts, topK   int
	width, taps, blockSize  int
	relative, relativeRange int
}

// Inkling2Config is config.json's text_config.
type Inkling2Config struct {
	NumLayers     int   `json:"num_hidden_layers"`
	LocalLayers   []int `json:"local_layer_ids"`
	BlockSize     int   `json:"n_layers_per_attn_res_block"`
	HiddenSize    int   `json:"hidden_size"`
	VocabSize     int   `json:"vocab_size"`
	UnpaddedVocab int   `json:"unpadded_vocab_size"`
	Context       int   `json:"model_max_length"`
	HeadDim       int   `json:"head_dim"`
	Heads         int   `json:"num_attention_heads"`
	KVHeads       int   `json:"num_key_value_heads"`
	LocalHeadDim  int   `json:"swa_head_dim"`
	LocalHeads    int   `json:"swa_num_attention_heads"`
	LocalKVHeads  int   `json:"swa_num_key_value_heads"`
	RelativeDim   int   `json:"d_rel"`
	RelativeRange int   `json:"rel_extent"`
	Window        int   `json:"sliding_window_size"`
	ConvTaps      int   `json:"sconv_kernel_size"`
	Experts       int   `json:"n_routed_experts"`
	SharedExperts int   `json:"n_shared_experts"`
	TopK          int   `json:"num_experts_per_tok"`
	ExpertWidth   int   `json:"intermediate_size"`
	Loop          *struct {
		Variable bool `json:"is_varloop"`
		Start    int  `json:"core_start"`
		End      int  `json:"core_end"`
		Passes   int  `json:"num_passes"`
		GateRank int  `json:"gate_rank"`
	} `json:"loop_config"`

	NormEps      float32 `json:"rms_norm_eps"`
	QKClip       float32 `json:"qk_l2_clip_threshold"`
	RelativeClip float32 `json:"rel_l2_clip_threshold"`
	LogFloor     float32 `json:"log_scaling_n_floor"`
	LogAlpha     float32 `json:"log_scaling_alpha"`
	RouteScale   float32 `json:"route_scale"`
	GateCap      float32 `json:"mlp_gate_softcap"`
	UpCap        float32 `json:"mlp_up_softcap"`
	LogitCap     float32 `json:"final_logit_softcapping"`
	GateBias     bool    `json:"use_gate_bias"`
	NormValues   bool    `json:"attn_res_normalize_values"`

	EOS json.RawMessage `json:"-"`
}

// inkVisit is one application of a layer: which layer's weights, and which
// pass of the loop it is. Each visit has keys, values and convolution
// histories of its own.
type inkVisit struct {
	layer, pass int
}

type inkConv struct {
	rows  []mlx.Array // per tap, float32: [C], or [1, 1, heads, C/heads] when the taps are also projected
	taps  mlx.Array   // the same, [taps, C]
	heads int         // heads the projected taps are per; 0 for fixed taps
}

type inkBank struct {
	w, scales   mlx.Array
	mode        string
	group, bits int
	outer       mlx.Array // [experts] float32, each expert's own scale; 0 when it has none
}

type inkAttention struct {
	// Projections, transposed for x @ w: of the position's own input — its
	// query, relative vector, sink and gate, side by side in fromQ — of the
	// key's and the value's, each with its convolution's taps after it where
	// the layer has them, and all of those side by side in fromAll for a
	// layer whose three inputs are one.
	fromQ, fromK, fromV, fromAll mlx.Array
	o                            mlx.Array // [in, out]
	// The same five packed to DenseBits bits, when the model was loaded so:
	// all, of the query's input, the key's, the value's, and the output.
	packed                  [5]Linear
	positions               mlx.Array // [d_rel, range]
	qScale, relScale        mlx.Array // float32
	kConv, vConv            inkConv
	headDim, heads, kvHeads int
	window                  int
}

type inkGate struct{ down, both mlx.Array } // [2*hidden, rank]; [rank, 2*hidden] float32: gain, then forget

type inkLayer struct {
	attn       inkAttention
	attnNorm   [3]mlx.Array // float32
	attnNorms  mlx.Array    // the same, one a query of attnQuery: [1, hidden] or [3, hidden]
	attnQuery  mlx.Array    // [hidden, 1] or [hidden, 3], float32, scaled; 0 for the first layer
	attnOut    inkConv
	router     mlx.Array // [hidden, experts] float32
	routerRows mlx.Array // the same as the checkpoint has it, [experts, hidden]
	bias       mlx.Array
	gateUp     inkBank
	down       inkBank
	mlpNorm    mlx.Array
	mlpQuery   mlx.Array
	mlpOut     inkConv
	gates      [4]inkGate
	messages   mlx.Array // [hidden, 4] float32, scaled
	local      bool
}

const inkResidualEps = 1e-6

// ServesInkling2 reports whether a config.json's architecture is this one.
func ServesInkling2(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		ModelType string `json:"model_type"`
	}
	return json.Unmarshal(raw, &c) == nil && c.ModelType == "inkling2_loop"
}

func readInkling2Config(dir string) (*Inkling2Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var top struct {
		Text Inkling2Config  `json:"text_config"`
		EOS  json.RawMessage `json:"eos_token_id"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("%s/config.json: %w", dir, err)
	}
	c := &top.Text
	c.EOS = top.EOS
	switch {
	case c.HiddenSize <= 0 || c.NumLayers <= 0 || c.BlockSize <= 0 || c.UnpaddedVocab <= 0 || c.UnpaddedVocab > c.VocabSize:
		return nil, fmt.Errorf("%s/config.json: no Inkling 2 text model in it", dir)
	case c.SharedExperts != 0 || !c.NormValues:
		return nil, fmt.Errorf("%s: an Inkling 2 with shared experts or unnormalized residual values is not read", dir)
	case c.Loop != nil && (c.Loop.Variable || c.Loop.Start%c.BlockSize != 0 || c.Loop.End%c.BlockSize != 0 || c.Loop.Passes < 2):
		return nil, fmt.Errorf("%s: an Inkling 2 loop of this kind is not read", dir)
	}
	return c, nil
}

// EOSTokens is the tokens config.json ends a reply with.
func (c *Inkling2Config) EOSTokens() []int32 {
	var one int32
	if json.Unmarshal(c.EOS, &one) == nil {
		return []int32{one}
	}
	var many []int32
	json.Unmarshal(c.EOS, &many)
	return many
}

func (c *Inkling2Config) visits() []inkVisit {
	end := c.NumLayers
	if c.Loop != nil {
		end = c.Loop.End
	}
	var plan []inkVisit
	for l := 0; l < end; l++ {
		plan = append(plan, inkVisit{layer: l})
	}
	if c.Loop != nil {
		for pass := 1; pass < c.Loop.Passes; pass++ {
			for l := c.Loop.Start; l < c.Loop.End; l++ {
				plan = append(plan, inkVisit{layer: l, pass: pass})
			}
		}
	}
	for l := end; l < c.NumLayers; l++ {
		plan = append(plan, inkVisit{layer: l})
	}
	return plan
}

// LoadInkling2 loads the model in dir, a directory of one safetensors file a
// tensor, named as the publisher names them (model.llm.…), as kinfer makes of
// Ollama's copy.
func LoadInkling2(dir string) (m *Inkling2, err error) {
	defer mlx.Recover(&err)
	c, err := readInkling2Config(dir)
	if err != nil {
		return nil, err
	}
	files, _ := filepath.Glob(filepath.Join(dir, "model.llm.*.safetensors"))
	if len(files) == 0 {
		return nil, fmt.Errorf("no model.llm.*.safetensors in %s", dir)
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
	defer func() {
		for _, v := range raw {
			v.Free()
		}
	}()

	m = &Inkling2{Config: c, plan: c.visits(), positions: map[int]mlx.Array{},
		logitCap: c.LogitCap, routeScale: c.RouteScale, gateCap: c.GateCap, upCap: c.UpCap, normEps: c.NormEps,
		hidden: c.HiddenSize, experts: c.Experts, topK: c.TopK, width: c.ExpertWidth, taps: c.ConvTaps,
		blockSize: c.BlockSize, relative: c.RelativeDim, relativeRange: c.RelativeRange}
	const prefix = "model.llm."
	var tmp []mlx.Array
	keep := func(a mlx.Array) mlx.Array {
		m.all = append(m.all, a)
		return a
	}
	has := func(name string) bool { _, ok := raw[prefix+name]; return ok }
	take := func(name string) mlx.Array {
		a, ok := raw[prefix+name]
		if !ok {
			panic(&mlx.Error{Msg: "weight " + prefix + name + " missing"})
		}
		delete(raw, prefix+name)
		return keep(a)
	}
	// made is an array worked out from weights, kept; its sources are freed
	// once it is evaluated.
	made := func(a mlx.Array, from ...mlx.Array) mlx.Array {
		tmp = append(tmp, from...)
		return keep(a)
	}
	src := func(name string) mlx.Array {
		a, ok := raw[prefix+name]
		if !ok {
			panic(&mlx.Error{Msg: "weight " + prefix + name + " missing"})
		}
		delete(raw, prefix+name)
		return a
	}
	f32 := func(name string) mlx.Array {
		a := src(name)
		return made(mlx.Astype(a, mlx.Float32), a)
	}
	// query is residual queries, [hidden, n] in float32, scaled as mix wants.
	scale := float32(1 / math.Sqrt(float64(c.HiddenSize)))
	query := func(a mlx.Array, n int, from ...mlx.Array) mlx.Array {
		q := mlx.MulScalar(mlx.Reshape(mlx.Astype(a, mlx.Float32), n, c.HiddenSize), scale)
		return made(mlx.Contiguous(mlx.T(q)), from...)
	}
	conv := func(stem string, channels, heads int) inkConv {
		if has(stem + ".static_w") {
			w := src(stem + ".static_w") // [taps, channels], by lag
			cv := inkConv{heads: heads}
			wf := mlx.Astype(w, mlx.Float32)
			tmp = append(tmp, w)
			cv.taps = keep(wf)
			for t := 0; t < c.ConvTaps; t++ {
				row := mlx.Reshape(mlx.Slice(wf, []int{t, 0}, []int{t + 1, channels}), 1, 1, heads, channels/heads)
				cv.rows = append(cv.rows, keep(mlx.Contiguous(row)))
			}
			return cv
		}
		w := src(stem + ".weight") // [channels, 1, taps], oldest first
		wf := mlx.Astype(mlx.Reshape(w, channels, c.ConvTaps), mlx.Float32)
		tmp = append(tmp, w, wf)
		cv := inkConv{taps: keep(mlx.Contiguous(mlx.T(wf)))}
		for t := 0; t < c.ConvTaps; t++ {
			row := mlx.Reshape(mlx.Slice(wf, []int{0, t}, []int{channels, t + 1}), channels)
			cv.rows = append(cv.rows, keep(mlx.Contiguous(row)))
		}
		return cv
	}
	bank := func(name string) inkBank {
		w := take(name)
		if w.Dtype() != mlx.Uint32 {
			return inkBank{w: w}
		}
		b := inkBank{w: w, scales: take(name + "_scale")}
		if has(name + "_amax") {
			amax, den := src(name+"_amax"), src(name+"_outer_scale_denominator")
			b.mode, b.group, b.bits = "nvfp4", 16, 4
			b.outer = made(mlx.Divide(mlx.Astype(amax, mlx.Float32), mlx.Astype(den, mlx.Float32)), amax, den)
		} else {
			b.mode, b.group, b.bits = "mxfp8", 32, 8
		}
		return b
	}

	m.DenseBits = inkDenseBits()
	pack := func(w mlx.Array) Linear {
		q := mlx.Quantize(w, 64, m.DenseBits, "affine")
		return Linear{W: keep(q[0]), Scales: keep(q[1]), Biases: keep(q[2]), Group: 64, Bits: m.DenseBits, Mode: "affine"}
	}
	m.embed = take("embed.weight")
	ue := src("unembed.weight")
	if m.DenseBits > 0 {
		m.unembedQ = pack(mlx.Slice(ue, []int{0, 0}, []int{c.UnpaddedVocab, c.HiddenSize}))
		tmp = append(tmp, ue)
	} else {
		m.unembed = made(mlx.T(mlx.Contiguous(mlx.Slice(ue, []int{0, 0}, []int{c.UnpaddedVocab, c.HiddenSize}))), ue)
	}
	m.embedNorm, m.outNorm = f32("embed_norm.weight"), f32("norm.weight")
	oq := src("attn_res.output_query")
	m.outQuery = query(oq, 1, oq)
	var loopQ, loopKV mlx.Array
	if c.Loop != nil {
		loopQ, loopKV = src("middle_cycle_cross_pass_queries"), src("middle_cycle_cross_pass_kv_queries")
		tmp = append(tmp, loopQ, loopKV)
	}
	local := map[int]bool{}
	for _, l := range c.LocalLayers {
		local[l] = true
	}
	m.layers = make([]inkLayer, c.NumLayers)
	for i := range m.layers {
		p := fmt.Sprintf("layers.%d.", i)
		L := &m.layers[i]
		L.local = local[i]
		dim, heads, kv, extent := c.HeadDim, c.Heads, c.KVHeads, c.RelativeRange
		if L.local {
			dim, heads, kv, extent = c.LocalHeadDim, c.LocalHeads, c.LocalKVHeads, c.Window
		}
		a := &L.attn
		a.headDim, a.heads, a.kvHeads = dim, heads, kv
		if L.local {
			a.window = c.Window
		}
		wq, wr, ws, wg := src(p+"attn.wq_du.weight"), src(p+"attn.wr_du.weight"), src(p+"attn.attn_sink_proj.weight"), src(p+"attn.gate_attn.weight")
		wk, wv, wo := src(p+"attn.wk_dv.weight"), src(p+"attn.wv_dv.weight"), src(p+"attn.wo_ud.weight")
		forQ, forK, forV := []mlx.Array{wq, wr, ws, wg}, []mlx.Array{wk}, []mlx.Array{wv}
		if !L.local {
			forK, forV = append(forK, src(p+"attn.k_sconv.weight_proj")), append(forV, src(p+"attn.v_sconv.weight_proj"))
		}
		side := func(ws ...mlx.Array) mlx.Array {
			return made(mlx.T(mlx.Concatenate(0, ws...)))
		}
		forAll := append(append(append([]mlx.Array(nil), forQ...), forK...), forV...)
		if m.DenseBits > 0 {
			for j, ws := range [][]mlx.Array{forAll, forQ, forK, forV, {wo}} {
				a.packed[j] = pack(mlx.Concatenate(0, ws...))
			}
		} else {
			a.fromQ, a.fromK, a.fromV = side(forQ...), side(forK...), side(forV...)
			a.fromAll = side(forAll...)
			a.o = made(mlx.T(wo))
		}
		tmp = append(tmp, wo)
		tmp = append(append(append(tmp, forQ...), forK...), forV...)
		a.positions = f32(p + "attn.rel_logits_proj.proj")
		if s := a.positions.Shape(); len(s) != 2 || s[0] != c.RelativeDim || s[1] != extent {
			panic(&mlx.Error{Msg: fmt.Sprintf("layer %d: relative positions are %v, want [%d %d]", i, s, c.RelativeDim, extent)})
		}
		a.qScale, a.relScale = f32(p+"attn.q_l2_clip.scale"), f32(p+"attn.rel_l2_clip.scale")
		a.kConv, a.vConv = conv(p+"attn.k_sconv", kv*dim, kv), conv(p+"attn.v_sconv", kv*dim, kv)
		L.attnOut, L.mlpOut = conv(p+"attn_sconv", c.HiddenSize, 0), conv(p+"mlp_sconv", c.HiddenSize, 0)
		if L.local {
			n := f32(p + "attn_norm.weight")
			L.attnNorm = [3]mlx.Array{n, n, n}
		} else {
			L.attnNorm = [3]mlx.Array{f32(p + "q_input_norm.weight"), f32(p + "k_input_norm.weight"), f32(p + "v_input_norm.weight")}
		}
		if i > 0 {
			q := src(fmt.Sprintf("attn_res.attn_queries.%d", i-1))
			n := 1
			if len(q.Shape()) == 2 {
				n = q.Shape()[0]
			}
			L.attnQuery = query(q, n, q)
			L.attnNorms = made(mlx.Contiguous(mlx.Stack(0, L.attnNorm[:n]...)))
		}
		mq := src(fmt.Sprintf("attn_res.mlp_queries.%d", i))
		L.mlpQuery = query(mq, 1, mq)
		L.mlpNorm = f32(p + "mlp_norm.weight")
		rw := src(p + "mlp.gate.weight")
		L.router = made(mlx.Contiguous(mlx.T(mlx.Astype(rw, mlx.Float32))), rw)
		L.routerRows = made(mlx.Astype(rw, mlx.Float32))
		if c.GateBias {
			L.bias = f32(p + "mlp.gate.bias")
		}
		L.gateUp, L.down = bank(p+"mlp.experts.w13_weight"), bank(p+"mlp.experts.w2_weight")
		if c.Loop != nil && i >= c.Loop.Start && i < c.Loop.End {
			row := 2 * (i - c.Loop.Start)
			H := c.HiddenSize
			rows := mlx.Concatenate(0,
				mlx.Slice(loopQ, []int{row, 0}, []int{row + 1, H}), mlx.Slice(loopKV, []int{row, 0}, []int{row + 2, H}),
				mlx.Slice(loopQ, []int{row + 1, 0}, []int{row + 2, H}))
			L.messages = query(rows, 4, rows)
			for g, name := range []string{"attn", "k", "v", "mlp"} {
				stem := p + "controllers." + name + "."
				dw, gw, fw := src(stem+"down_weight"), src(stem+"gain_weight"), src(stem+"forget_weight")
				// Gain and forget are read from the same few numbers: one product.
				both := mlx.Contiguous(mlx.T(mlx.Astype(mlx.Concatenate(0, gw, fw), mlx.Float32)))
				L.gates[g] = inkGate{down: made(mlx.T(dw)), both: made(both, gw, fw)}
				tmp = append(tmp, dw)
			}
		}
	}
	mlx.Eval(m.all...)
	for _, a := range tmp {
		a.Free()
	}
	for _, a := range m.all {
		m.bytes += int64(a.Nbytes())
	}
	if m.eyes, err = loadInkEyes(dir, c.HiddenSize); err != nil {
		m.Free()
		return nil, err
	}
	if m.eyes != nil {
		mlx.Eval(m.eyes.all...)
		for _, a := range m.eyes.all {
			m.bytes += int64(a.Nbytes())
		}
	}
	return m, nil
}

// Visits is how many times a layer is applied to a token: every layer once,
// the looped ones once a pass.
func (m *Inkling2) Visits() int { return len(m.plan) }

// Bytes is the size of the weights in memory.
func (m *Inkling2) Bytes() int64 { return m.bytes }

// Free releases the weights.
func (m *Inkling2) Free() {
	for _, a := range m.all {
		a.Free()
	}
	for _, a := range m.positions {
		a.Free()
	}
	if m.eyes != nil {
		for _, a := range m.eyes.all {
			a.Free()
		}
	}
	m.all, m.positions, m.eyes = nil, nil, nil
}

// Inkling2Cache is one sequence's state: for every visit of a layer, its
// keys and values, in buffers that grow kvStep positions at a time, and the
// last inputs of its four convolutions.
type Inkling2Cache struct {
	keys, vals []mlx.Array    // per visit, [B, Hkv, capacity, D]
	conv       [][4]mlx.Array // per visit: keys, values, attention output, feed-forward output; [B, taps-1, C]
	Offset     int

	// pad is set for the rows of a batch (Inkling2Batch): row r's sequence
	// begins at column pad[r], the columns before it empty. A cache of its
	// own has none. rows is what a step works out from it once, for every
	// layer application to use.
	pad  []int
	rows *inkRows
}

// inkRows is, for a step of one token a row, where each row's keys begin for
// a layer that sees everything and for one that sees a window, and how much
// each row's seeing-everything layers sharpen.
type inkRows struct {
	all, window mlx.Array // [B] int32
	first       [2]int    // the least of each
	tau         mlx.Array // [B] float32
}

// stepRows works rows out for a step at c.Offset.
func (m *Inkling2) stepRows(c *Inkling2Cache, B int) *inkRows {
	cf := m.Config
	all, win, tau := make([]int32, B), make([]int32, B), make([]float32, B)
	r := &inkRows{first: [2]int{c.Offset, c.Offset}}
	for b := 0; b < B; b++ {
		pad := 0
		if b < len(c.pad) {
			pad = c.pad[b]
		}
		all[b] = int32(pad)
		win[b] = int32(max(pad, c.Offset-cf.Window+1))
		r.first[0], r.first[1] = min(r.first[0], int(all[b])), min(r.first[1], int(win[b]))
		tau[b] = 1
		// A layer that sees everything sharpens with the length of what it
		// sees, past the first few thousand tokens.
		if n := float32(c.Offset + 1 - pad); cf.LogFloor > 0 && cf.LogAlpha != 0 && n > cf.LogFloor {
			tau[b] = 1 + cf.LogAlpha*float32(math.Log(float64(n/cf.LogFloor)))
		}
	}
	r.all, r.window, r.tau = mlx.FromInt32(all, B), mlx.FromInt32(win, B), mlx.FromFloat32(tau, B)
	return r
}

// NewCache makes an empty cache for one sequence.
func (m *Inkling2) NewCache() *Inkling2Cache {
	n := len(m.plan)
	return &Inkling2Cache{keys: make([]mlx.Array, n), vals: make([]mlx.Array, n), conv: make([][4]mlx.Array, n)}
}

// Arrays are the cache's current arrays, to evaluate along with a step's
// logits.
func (c *Inkling2Cache) Arrays() []mlx.Array {
	var r []mlx.Array
	for i := range c.keys {
		for _, a := range []mlx.Array{c.keys[i], c.vals[i], c.conv[i][0], c.conv[i][1], c.conv[i][2], c.conv[i][3]} {
			if a != 0 {
				r = append(r, a)
			}
		}
	}
	return r
}

// Clone is a second cache at the same point, sharing c's arrays.
func (c *Inkling2Cache) Clone() *Inkling2Cache {
	d := &Inkling2Cache{keys: make([]mlx.Array, len(c.keys)), vals: make([]mlx.Array, len(c.keys)),
		conv: make([][4]mlx.Array, len(c.keys)), Offset: c.Offset}
	ref := func(a mlx.Array) mlx.Array {
		if a == 0 {
			return 0
		}
		return a.Ref()
	}
	for i := range c.keys {
		d.keys[i], d.vals[i] = ref(c.keys[i]), ref(c.vals[i])
		for j := range c.conv[i] {
			d.conv[i][j] = ref(c.conv[i][j])
		}
	}
	return d
}

// Free releases the cache's arrays.
func (c *Inkling2Cache) Free() {
	for _, a := range c.Arrays() {
		a.Free()
	}
	n := len(c.keys)
	*c = Inkling2Cache{keys: make([]mlx.Array, n), vals: make([]mlx.Array, n), conv: make([][4]mlx.Array, n)}
}

// Forward runs tokens [B, L] through the model, advancing cache, and returns
// the logits of the last position, [B, vocab], evaluated.
func (m *Inkling2) Forward(tokens [][]int32, cache *Inkling2Cache) (logits mlx.Array, err error) {
	defer mlx.Recover(&err)
	B, L := len(tokens), len(tokens[0])
	flat := make([]int32, 0, B*L)
	for _, t := range tokens {
		flat = append(flat, t...)
	}
	ids := mlx.FromInt32(flat, B, L)
	logits = m.Step(ids, cache)
	ids.Free()
	mlx.Eval(append([]mlx.Array{logits}, cache.Arrays()...)...)
	return logits, nil
}

// Step records one forward pass of tokens [B, L] int32 and returns the last
// position's logits, [B, vocab], without running anything, as Qwen35.Step
// does; the cache's arrays are replaced by lazy ones too.
func (m *Inkling2) Step(tokens mlx.Array, cache *Inkling2Cache) mlx.Array {
	s := tokens.Shape()
	B, L := s[0], s[1]
	scope := mlx.NewScope()
	defer scope.Close()
	old := cache.Arrays()

	if L == 1 && !inkPlain {
		cache.rows = m.stepRows(cache, B)
	}
	h := m.hiddenStates(tokens, cache)
	cache.rows = nil
	last := mlx.Slice(h, []int{0, L - 1, 0}, []int{B, L, m.hidden})
	logits := m.logits(mlx.Reshape(last, B, m.hidden))

	cache.Offset += L
	scope.Keep(logits)
	fresh := cache.Arrays()
	scope.Escape(fresh...)
	freeOld(old, fresh)
	return logits
}

// logits is the vocabulary's scores for hidden states [N, hidden], float32.
func (m *Inkling2) logits(h mlx.Array) mlx.Array {
	var lg mlx.Array
	if q := m.unembedQ; q.quantized() {
		lg = mlx.QuantizedMatmul(h, q.W, q.Scales, q.Biases, q.Group, q.Bits, q.Mode)
	} else {
		lg = mlx.Matmul(h, m.unembed)
	}
	return inkSoftcap(mlx.Astype(lg, mlx.Float32), m.logitCap)
}

// inkStream is the residual stream: the blocks closed so far and the one
// being written.
type inkStream struct {
	blocks  []mlx.Array
	partial mlx.Array
}

func (s inkStream) sources() []mlx.Array {
	return append(s.blocks[:len(s.blocks):len(s.blocks)], s.partial)
}

// hiddenStates is the model's normed output for tokens [B, L], [B, L, hidden].
func (m *Inkling2) hiddenStates(tokens mlx.Array, cache *Inkling2Cache) mlx.Array {
	c := m.Config
	state := inkStream{partial: m.placed(inkRMS(mlx.Take(m.embed, tokens, 0), m.embedNorm, m.normEps))}
	var prelude inkStream
	var previous []mlx.Array
	for i, v := range m.plan {
		if c.Loop != nil && v.layer == c.Loop.Start {
			if v.pass == 0 {
				prelude = state
			} else {
				// What the pass before wrote: its blocks after the prelude's
				// and the one it left open.
				previous = append(append([]mlx.Array(nil), state.blocks[len(prelude.blocks)+1:]...), state.partial)
				state = prelude
			}
		}
		var messages []mlx.Array
		l := &m.layers[v.layer]
		if v.pass > 0 {
			messages = m.messages(previous, l.messages)
		}
		state = m.layer(l, state, messages, cache, i, v.layer%m.blockSize == 0)
	}
	if !inkPlain && inkGrouped(m.hidden) {
		return inkMixFused(state.sources(), m.outQuery, m.outNorm, m.normEps)[0]
	}
	return inkRMS(inkPick(inkMix(state.sources(), m.outQuery), 0), m.outNorm, m.normEps)
}

// messages is what a layer's four inputs read of the pass before: its blocks
// mixed by the layer's four queries, each [B, L, hidden].
func (m *Inkling2) messages(previous []mlx.Array, queries mlx.Array) []mlx.Array {
	if !inkPlain && inkGrouped(m.hidden) {
		return inkMixFused(previous, queries, 0, 0)
	}
	mixed := inkMix(previous, queries)
	return []mlx.Array{inkPick(mixed, 0), inkPick(mixed, 1), inkPick(mixed, 2), inkPick(mixed, 3)}
}

func (m *Inkling2) layer(l *inkLayer, state inkStream, messages []mlx.Array, cache *Inkling2Cache, visit int, boundary bool) inkStream {
	grouped := !inkPlain && inkGrouped(m.hidden)
	in := [3]mlx.Array{state.partial, state.partial, state.partial}
	normed := false
	if len(state.blocks) != 0 && l.attnQuery != 0 {
		n := l.attnQuery.Shape()[1]
		var mixes []mlx.Array
		switch {
		case grouped && messages == nil:
			// Mixed, normed and scaled in one operation.
			mixes, normed = inkMixFused(state.sources(), l.attnQuery, l.attnNorms, m.normEps), true
		case grouped:
			mixes = inkMixFused(state.sources(), l.attnQuery, 0, 0)
		default:
			mixed := inkMix(state.sources(), l.attnQuery)
			for i := 0; i < n; i++ {
				mixes = append(mixes, inkPick(mixed, i))
			}
		}
		for i := range in {
			in[i] = mixes[min(i, n-1)]
		}
	}
	raw := in
	for i := range in {
		switch {
		case normed:
		case messages != nil && grouped:
			in[i] = l.gates[i].applyNormed(in[i], messages[i], l.attnNorm[i], m.normEps)
		case messages != nil:
			in[i] = inkRMS(l.gates[i].apply(in[i], messages[i]), l.attnNorm[i], m.normEps)
		case i > 0 && raw[i] == raw[0] && l.attnNorm[i] == l.attnNorm[0]:
			// One input, normed once: the attention then projects it once.
			in[i] = in[0]
		default:
			in[i] = inkRMS(in[i], l.attnNorm[i], m.normEps)
		}
	}
	att := m.attention(&l.attn, in[0], in[1], in[2], cache, visit)
	next := state
	switch {
	case boundary:
		next = inkStream{blocks: state.sources(), partial: l.attnOut.run(att, 0, &cache.conv[visit][2])}
	case inkPlain:
		att = l.attnOut.run(att, 0, &cache.conv[visit][2])
		next.partial = mlx.Add(mlx.Astype(state.partial, mlx.Float32), mlx.Astype(att, mlx.Float32))
	default:
		next.partial = l.attnOut.fusedAdd(att, 0, &cache.conv[visit][2], state.partial, mlx.Float32)
	}
	var ff mlx.Array
	switch {
	case grouped && messages == nil:
		ff = inkMixFused(next.sources(), l.mlpQuery, l.mlpNorm, m.normEps)[0]
	case grouped:
		ff = l.gates[3].applyNormed(inkMixFused(next.sources(), l.mlpQuery, 0, 0)[0], messages[3], l.mlpNorm, m.normEps)
	default:
		ff = inkPick(inkMix(next.sources(), l.mlpQuery), 0)
		if messages != nil {
			ff = l.gates[3].apply(ff, messages[3])
		}
		ff = inkRMS(ff, l.mlpNorm, m.normEps)
	}
	out := m.feedForward(l, ff)
	if inkPlain {
		out = l.mlpOut.run(out, 0, &cache.conv[visit][3])
		next.partial = mlx.Astype(mlx.Add(mlx.Astype(next.partial, mlx.Float32), mlx.Astype(out, mlx.Float32)), mlx.Bfloat16)
	} else {
		next.partial = l.mlpOut.fusedAdd(out, 0, &cache.conv[visit][3], next.partial, mlx.Bfloat16)
	}
	return next
}

// inkRMS is x normed in float32, as the model norms everything, in bfloat16.
func inkRMS(x, w mlx.Array, eps float32) mlx.Array {
	return mlx.Astype(mlx.RMSNorm(mlx.Astype(x, mlx.Float32), w, eps), mlx.Bfloat16)
}

// inkMix mixes the residual blocks, each normed, for every token on its own:
// queries [hidden, Q] attend over the blocks and each gets its mixture,
// [B, L, Q, hidden] float32.
func inkMix(sources []mlx.Array, queries mlx.Array) mlx.Array {
	f := make([]mlx.Array, len(sources))
	for i, s := range sources {
		f[i] = mlx.Astype(s, mlx.Float32)
	}
	keys := mlx.RMSNorm(mlx.Stack(2, f...), 0, inkResidualEps) // [B, L, S, hidden]
	scores := mlx.Swapaxes(mlx.Matmul(keys, queries), 2, 3)
	return mlx.Matmul(mlx.Softmax(scores, -1), keys)
}

// inkPick is mixture i of inkMix's, [B, L, hidden].
func inkPick(mixed mlx.Array, i int) mlx.Array {
	s := mixed.Shape()
	return mlx.Reshape(mlx.Slice(mixed, []int{0, 0, i, 0}, []int{s[0], s[1], i + 1, s[3]}), s[0], s[1], s[3])
}

func inkTanh(x mlx.Array) mlx.Array {
	return mlx.AddScalar(mlx.MulScalar(mlx.Sigmoid(mlx.MulScalar(x, 2)), 2), -1)
}

func inkSoftcap(x mlx.Array, limit float32) mlx.Array {
	if limit <= 0 {
		return x
	}
	return mlx.MulScalar(inkTanh(mlx.MulScalar(x, 1/limit)), limit)
}

// apply lets a message from the pass before into a layer's input: both
// normed, read through a narrow projection, which says how much of the input
// to forget and how much of the message to take.
func (g inkGate) apply(input, message mlx.Array) mlx.Array {
	norm := func(x mlx.Array) mlx.Array {
		return mlx.Astype(mlx.RMSNorm(mlx.Astype(x, mlx.Float32), 0, inkResidualEps), mlx.Bfloat16)
	}
	z := mlx.Astype(mlx.Matmul(mlx.Concatenate(-1, norm(input), norm(message)), g.down), mlx.Float32)
	s := input.Shape()
	B, L, H := s[0], s[1], s[2]
	if !inkPlain {
		return mixGate(input, message, mlx.Matmul(z, g.both))
	}
	gf := inkTanh(mlx.Matmul(z, g.both))
	gain, forget := mlx.Slice(gf, []int{0, 0, 0}, []int{B, L, H}), mlx.Slice(gf, []int{0, 0, H}, []int{B, L, 2 * H})
	keep := mlx.AddScalar(mlx.Negative(forget), 1)
	return mlx.Add(mlx.Multiply(mlx.Astype(input, mlx.Float32), keep), mlx.Multiply(mlx.Astype(message, mlx.Float32), gain))
}

// run is the convolution of x [B, L, C] over the last taps tokens, its
// history kept in *hist: fixed taps, or taps that shifts [B, L, heads*taps],
// a projection of the layer's input, adds to for every token and head; plus
// x itself, the sum scaled to keep its size.
func (cv inkConv) run(x, shifts mlx.Array, hist *mlx.Array) mlx.Array {
	if !inkPlain {
		return cv.fused(x, shifts, hist)
	}
	s := x.Shape()
	B, L, C := s[0], s[1], s[2]
	taps := len(cv.rows)
	tail := taps - 1
	h := *hist
	if h == 0 {
		h = mlx.Zeros(x.Dtype(), B, tail, C)
	}
	window := mlx.Concatenate(1, h, x)
	*hist = mlx.Contiguous(mlx.Slice(window, []int{0, L, 0}, []int{B, L + tail, C}))
	wf := mlx.Astype(window, mlx.Float32)
	terms := taps + 1
	var per mlx.Array
	if shifts != 0 {
		per = mlx.Reshape(mlx.Astype(shifts, mlx.Float32), B, L, cv.heads, taps)
		terms += taps
	}
	sum := mlx.Astype(x, mlx.Float32)
	for t := 0; t < taps; t++ {
		start, coef := t, cv.rows[t]
		if per != 0 {
			// These taps are indexed by how far back they reach.
			start = tail - t
			shift := mlx.Slice(per, []int{0, 0, 0, t}, []int{B, L, cv.heads, t + 1})
			coef = mlx.Reshape(mlx.Add(coef, shift), B, L, C)
		}
		sum = mlx.Add(sum, mlx.Multiply(mlx.Slice(wf, []int{0, start, 0}, []int{B, start + L, C}), coef))
	}
	return mlx.Astype(mlx.MulScalar(sum, float32(1/math.Sqrt(float64(terms)))), x.Dtype())
}

// inkClip scales x's last axis by weight, shortens it to at most limit long,
// and scales it by temperature; either may be 0.
func inkClip(x, weight, temperature mlx.Array, limit float32) mlx.Array {
	v := mlx.Astype(x, mlx.Float32)
	if weight != 0 {
		v = mlx.Multiply(v, weight)
	}
	norm := mlx.Sqrt(mlx.Sum(mlx.Square(v), -1, true))
	factor := mlx.Divide(mlx.Scalar(limit), mlx.AddScalar(norm, inkResidualEps))
	v = mlx.Multiply(v, mlx.Minimum(factor, mlx.Scalar(1)))
	if temperature != 0 {
		v = mlx.Multiply(v, temperature)
	}
	return mlx.Astype(v, x.Dtype())
}

// attention is a layer's attention of its three inputs, [B, L, hidden], up to
// and including its output projection.
func (m *Inkling2) attention(a *inkAttention, qIn, kIn, vIn mlx.Array, cache *Inkling2Cache, visit int) mlx.Array {
	c := m.Config
	s := qIn.Shape()
	B, L := s[0], s[1]
	H, Hkv, D := a.heads, a.kvHeads, a.headDim
	off := cache.Offset
	taps := 0
	if a.window == 0 {
		taps = Hkv * m.taps
	}
	// What each input is projected to, side by side: one product where the
	// three are one.
	wQ, wK := H*D+H*m.relative+H+H*D, Hkv*D+taps
	same := qIn == kIn && kIn == vIn
	if L == 1 && !inkPlain && D%32 == 0 && cache.rows != nil {
		var all mlx.Array
		if same {
			all = a.project(qIn, 0)
		} else {
			all = mlx.Concatenate(-1, a.project(qIn, 1), a.project(kIn, 2), a.project(vIn, 3))
		}
		rows := cache.rows
		start, first, tau := rows.all, rows.first[0], mlx.Array(0)
		if a.window > 0 {
			start, first = rows.window, rows.first[1]
			tau = inkOnes(B)
		} else {
			tau = rows.tau
		}
		q, r, k, v := m.prepFused(a, all, tau, &cache.conv[visit][0], &cache.conv[visit][1])
		end := off + 1
		for _, kv := range []struct {
			buf *mlx.Array
			new mlx.Array
		}{{&cache.keys[visit], k}, {&cache.vals[visit], v}} {
			if *kv.buf == 0 || (*kv.buf).Shape()[2] < end {
				grown := mlx.Zeros(kv.new.Dtype(), B, Hkv, roundUp(end), D)
				if *kv.buf != 0 && off > 0 {
					grown = mlx.SliceUpdate(grown, mlx.Slice(*kv.buf, []int{0, 0, 0, 0}, []int{B, Hkv, off, D}),
						[]int{0, 0, 0, 0}, []int{B, Hkv, off, D})
				}
				*kv.buf = grown
			}
			*kv.buf = mlx.SliceUpdate(*kv.buf, kv.new, []int{0, 0, off, 0}, []int{B, Hkv, end, D})
		}
		out := inkAttend(q, cache.keys[visit], cache.vals[visit], r, a.positions, all, start,
			H*D+H*m.relative, wQ-H*D, first, end, H)
		return a.project(out, 4)
	}
	var fromQ, fromK, fromV mlx.Array
	if same {
		all := a.project(qIn, 0)
		fromQ = mlx.Slice(all, []int{0, 0, 0}, []int{B, L, wQ})
		fromK = mlx.Slice(all, []int{0, 0, wQ}, []int{B, L, wQ + wK})
		fromV = mlx.Slice(all, []int{0, 0, wQ + wK}, []int{B, L, wQ + 2*wK})
	} else {
		fromQ, fromK, fromV = a.project(qIn, 1), a.project(kIn, 2), a.project(vIn, 3)
	}
	part := func(x mlx.Array, from, to int) mlx.Array {
		return mlx.Slice(x, []int{0, 0, from}, []int{B, L, to})
	}
	q := mlx.Reshape(part(fromQ, 0, H*D), B, L, H, D)
	r := mlx.Reshape(part(fromQ, H*D, H*D+H*m.relative), B, L, H, m.relative)
	sinks := part(fromQ, H*D+H*m.relative, H*D+H*m.relative+H) // [B, L, H]
	gates := part(fromQ, wQ-H*D, wQ)                           // [B, L, H*D]
	var kTaps, vTaps mlx.Array
	if taps > 0 {
		kTaps, vTaps = part(fromK, Hkv*D, wK), part(fromV, Hkv*D, wK)
	}
	k := mlx.Reshape(a.kConv.run(part(fromK, 0, Hkv*D), kTaps, &cache.conv[visit][0]), B, L, Hkv, D)
	v := mlx.Reshape(a.vConv.run(part(fromV, 0, Hkv*D), vTaps, &cache.conv[visit][1]), B, L, Hkv, D)

	// A layer that sees everything sharpens with the length of what it sees.
	var tau mlx.Array
	if a.window == 0 && c.LogFloor > 0 && c.LogAlpha != 0 && float32(off+L) > c.LogFloor {
		length := mlx.Arange(float64(off+1), float64(off+L+1), 1, mlx.Float32)
		ratio := mlx.Maximum(mlx.MulScalar(length, 1/c.LogFloor), mlx.Scalar(1))
		tau = mlx.AddScalar(mlx.MulScalar(mlx.Log(ratio), c.LogAlpha), 1)
	}
	clip := inkClipFused
	if inkPlain {
		clip = func(x, w, tau mlx.Array, limit float32) mlx.Array {
			if tau != 0 {
				tau = mlx.Reshape(tau, 1, L, 1, 1)
			}
			return inkClip(x, w, tau, limit)
		}
	}
	q = clip(q, a.qScale, tau, c.QKClip)                                                                // [B, L, H, D]
	k = mlx.Transpose(clip(k, 0, 0, c.QKClip), 0, 2, 1, 3)                                              // [B, Hkv, L, D]
	v = mlx.Transpose(mlx.RMSNorm(v, 0, m.normEps), 0, 2, 1, 3)                                         // [B, Hkv, L, D]
	table := mlx.Matmul(mlx.Astype(clip(r, a.relScale, tau, c.RelativeClip), mlx.Float32), a.positions) // [B, L, H, range]

	end := off + L
	for _, kv := range []struct {
		buf *mlx.Array
		new mlx.Array
	}{{&cache.keys[visit], k}, {&cache.vals[visit], v}} {
		if *kv.buf == 0 || (*kv.buf).Shape()[2] < end {
			grown := mlx.Zeros(kv.new.Dtype(), B, Hkv, roundUp(end), D)
			if *kv.buf != 0 && off > 0 {
				grown = mlx.SliceUpdate(grown, mlx.Slice(*kv.buf, []int{0, 0, 0, 0}, []int{B, Hkv, off, D}),
					[]int{0, 0, 0, 0}, []int{B, Hkv, off, D})
			}
			*kv.buf = grown
		}
		*kv.buf = mlx.SliceUpdate(*kv.buf, kv.new, []int{0, 0, off, 0}, []int{B, Hkv, end, D})
	}
	// The keys any of these positions may read: all of them, or the window
	// before the first.
	first := 0
	if a.window > 0 {
		first = max(0, off-a.window+1)
	}
	S := end - first
	keys := mlx.Astype(mlx.Slice(cache.keys[visit], []int{0, 0, first, 0}, []int{B, Hkv, end, D}), mlx.Float32)
	vals := mlx.Astype(mlx.Slice(cache.vals[visit], []int{0, 0, first, 0}, []int{B, Hkv, end, D}), mlx.Float32)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	table = mlx.Transpose(table, 0, 2, 1, 3) // [B, H, L, range]
	sink := mlx.Reshape(mlx.Transpose(mlx.Astype(sinks, mlx.Float32), 0, 2, 1), B, H, L, 1)
	gate := mlx.Sigmoid(mlx.Astype(gates, mlx.Float32))

	group := H / Hkv
	qg := mlx.Reshape(mlx.Astype(q, mlx.Float32), B, Hkv, group, L, D)
	scores := mlx.MulScalar(mlx.Matmul(qg, mlx.Swapaxes(mlx.Reshape(keys, B, Hkv, 1, S, D), 3, 4)), float32(1/math.Sqrt(float64(D))))
	scores = mlx.Reshape(scores, B, H, L, S)
	extent := table.Shape()[3]
	if L == 1 {
		// One position: every key in view is behind it, the last at no
		// distance. Its biases are its table backwards, for as many keys as
		// the table reaches.
		n := min(S, extent)
		bias := mlx.Take(table, m.backwards(n), 3)
		if S > n {
			bias = mlx.Concatenate(3, mlx.Zeros(mlx.Float32, B, H, 1, S-n), bias)
		}
		scores = mlx.Add(scores, bias)
	} else {
		// How far back each key is from each position: its bias comes from
		// the position's table at that distance, while the table reaches.
		dist := mlx.Subtract(mlx.Reshape(mlx.Arange(float64(off), float64(end), 1, mlx.Int32), L, 1),
			mlx.Reshape(mlx.Arange(float64(first), float64(end), 1, mlx.Int32), 1, S))
		idx := mlx.Maximum(mlx.Minimum(dist, mlx.ConstInt(extent-1)), mlx.ConstInt(0))
		bias := mlx.TakeAlongAxis(table, mlx.BroadcastTo(mlx.Reshape(idx, 1, 1, L, S), B, H, L, S), -1)
		bias = mlx.Where(mlx.Reshape(mlx.Equal(idx, dist), 1, 1, L, S), bias, mlx.Scalar(0))
		scores = mlx.Add(scores, bias)
		allowed := mlx.Greater(dist, mlx.ConstInt(-1))
		if a.window > 0 {
			allowed = mlx.Multiply(mlx.Astype(allowed, mlx.Int32), mlx.Astype(mlx.Less(dist, mlx.ConstInt(a.window)), mlx.Int32))
			allowed = mlx.Greater(allowed, mlx.ConstInt(0))
		}
		scores = mlx.Where(mlx.Reshape(allowed, 1, 1, L, S), scores, mlx.Scalar(float32(math.Inf(-1))))
	}
	// The sink takes its share of the attention and gives nothing.
	p := mlx.Softmax(mlx.Concatenate(-1, scores, sink), -1)
	p = mlx.Slice(p, []int{0, 0, 0, 0}, []int{B, H, L, S})
	out := mlx.Matmul(mlx.Reshape(p, B, Hkv, group, L, S), mlx.Reshape(vals, B, Hkv, 1, S, D))
	out = mlx.Reshape(mlx.Transpose(mlx.Reshape(out, B, H, L, D), 0, 2, 1, 3), B, L, H*D)
	return a.project(mlx.Astype(mlx.Multiply(out, gate), qIn.Dtype()), 4)
}

// project is x through one of the attention's five projections: 0 all of
// them of one input, 1 the query's input's, 2 the key's, 3 the value's, 4
// the output.
func (a *inkAttention) project(x mlx.Array, which int) mlx.Array {
	if q := a.packed[which]; q.quantized() {
		return mlx.QuantizedMatmul(x, q.W, q.Scales, q.Biases, q.Group, q.Bits, q.Mode)
	}
	return mlx.Matmul(x, [...]mlx.Array{a.fromAll, a.fromQ, a.fromK, a.fromV, a.o}[which])
}

// inkDenseBits is how many bits the attention's projections and the output
// layer, which the publisher keeps in bfloat16, are packed to when the model
// is loaded: 8, unless KINFER_INK_DENSE says 4, or 16 or off for as
// published. They are 2.6 GB and 0.9 GB of the 6.7 GB a token reads, and a
// step is as long as its weights take to read: on an M3 Ultra 47 tok/s as
// published, 52 at 8 bits, 55 at 4. Packed, the model no longer computes
// exactly what the published weights compute; at 8 bits its answers to the
// tests here read the same.
func inkDenseBits() int {
	switch os.Getenv("KINFER_INK_DENSE") {
	case "4":
		return 4
	case "16", "off":
		return 0
	}
	return 8
}

// backwards is the indices n-1 … 0, kept for each n it is asked for: a
// position's biases for the n keys before it are its table read backwards.
func (m *Inkling2) backwards(n int) mlx.Array {
	if a, ok := m.positions[n]; ok {
		return a
	}
	idx := make([]int32, n)
	for i := range idx {
		idx[i] = int32(n - 1 - i)
	}
	a := mlx.FromInt32(idx, n)
	mlx.Outlive(a)
	m.positions[n] = a
	return a
}

// feedForward routes each token of x [B, L, hidden] to topK of the layer's
// experts and sums what they make of it, each weighted by the router.
func (m *Inkling2) feedForward(l *inkLayer, x mlx.Array) mlx.Array {
	if !inkPlain {
		return m.feedForwardFused(l, x)
	}
	s := x.Shape()
	B, L, H := s[0], s[1], s[2]
	N, K, W := B*L, m.topK, m.width
	flat := mlx.Reshape(x, N, H)

	logits := mlx.Matmul(mlx.Astype(flat, mlx.Float32), l.router) // [N, experts]
	choose := mlx.Sigmoid(logits)
	if l.bias != 0 {
		choose = mlx.Add(choose, l.bias)
	}
	idx := mlx.Slice(mlx.Argpartition(mlx.Negative(choose), K-1, -1), []int{0, 0}, []int{N, K})
	chosen := mlx.TakeAlongAxis(logits, idx, -1)
	weights := mlx.MulScalar(mlx.Softmax(mlx.Negative(mlx.Softplus(mlx.Negative(chosen))), -1), m.routeScale)

	// The first projection's rows are gate and up by turns.
	pair := mlx.Reshape(l.gateUp.project(mlx.Reshape(flat, N, 1, 1, H), idx), N, K, W, 2)
	gate := mlx.Slice(pair, []int{0, 0, 0, 0}, []int{N, K, W, 1})
	up := mlx.Slice(pair, []int{0, 0, 0, 1}, []int{N, K, W, 2})
	act := mlx.Multiply(mlx.Multiply(inkSoftcap(gate, m.gateCap), mlx.Sigmoid(gate)), inkSoftcap(up, m.upCap))
	act = mlx.Multiply(mlx.Reshape(act, N, K, 1, W), mlx.Reshape(weights, N, K, 1, 1))
	down := l.down.project(mlx.Astype(act, x.Dtype()), idx) // [N, K, 1, hidden]
	return mlx.Astype(mlx.Reshape(mlx.Sum(down, 1, false), B, L, H), x.Dtype())
}

// project is x [N, …, in] times the expert idx [N, K] names for each row,
// [N, K, 1, out] float32.
func (b inkBank) project(x, idx mlx.Array) mlx.Array {
	if b.scales == 0 {
		return mlx.Astype(mlx.GatherMM(x, mlx.Swapaxes(b.w, 1, 2), idx, false), mlx.Float32)
	}
	y := mlx.Astype(mlx.GatherQMM(x, b.w, b.scales, 0, idx, b.group, b.bits, b.mode, false), mlx.Float32)
	if b.outer != 0 {
		s := idx.Shape()
		y = mlx.Multiply(y, mlx.Reshape(mlx.Take(b.outer, idx, 0), s[0], s[1], 1, 1))
	}
	return y
}
