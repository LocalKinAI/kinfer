package mlxembed

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"sync"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// Sound, as EmbeddingGemma 2 hears it: a WAV at any rate, made 16 kHz mono,
// cut into pieces of at most 30 s at its quietest moments, each piece's
// log-mel frames through Gemma 4's audio tower — a USM conformer, four times
// fewer frames after two strided convolutions — and projected into the text's
// space. Ported from Ollama (mlxrunner/model/gemma4 and model/audio), whose
// vectors the home's memory already holds: the same decoder, resampler, cuts
// and features, so a sound placed by either is placed alike.

const (
	audioRate       = 16000
	audioMelBins    = 128
	audioFrameLen   = 320 // 20 ms
	audioHopLen     = 160 // 10 ms
	audioFFTLen     = 512
	audioMelFloor   = 1e-3
	audioMaxFreq    = 8000.0
	audioPieceSecs  = 30
	audioMaxSeconds = 600
	audioMaxRate    = 384000
)

// audioConfig is config.json's audio_config.
type audioConfig struct {
	Hidden       int     `json:"hidden_size"`
	Layers       int     `json:"num_hidden_layers"`
	Heads        int     `json:"num_attention_heads"`
	ConvKernel   int     `json:"conv_kernel_size"`
	Residual     float32 `json:"residual_weight"`
	Chunk        int     `json:"attention_chunk_size"`
	ContextLeft  int     `json:"attention_context_left"`
	ContextRight int     `json:"attention_context_right"`
	LogitCap     float32 `json:"attention_logit_cap"`
	InvalidLogit float32 `json:"attention_invalid_logits_value"`
	Eps          float32 `json:"rms_norm_eps"`
	Clipping     float32 `json:"gradient_clipping"`
	ModelType    string  `json:"model_type"`
}

func (a *audioConfig) defaults() {
	set := func(v *int, d int) {
		if *v == 0 {
			*v = d
		}
	}
	setf := func(v *float32, d float32) {
		if *v == 0 {
			*v = d
		}
	}
	set(&a.Hidden, 1024)
	set(&a.Layers, 12)
	set(&a.Heads, 8)
	set(&a.ConvKernel, 5)
	set(&a.Chunk, 12)
	set(&a.ContextLeft, 13)
	setf(&a.Residual, 0.5)
	setf(&a.LogitCap, 50)
	setf(&a.InvalidLogit, -1e9)
	setf(&a.Eps, 1e-6)
	setf(&a.Clipping, 1e10)
}

type audioTower struct {
	cfg       audioConfig
	convs     [2]struct{ w, norm mlx.Array } // w [out, 3, 3, in]
	inputProj tlinear
	layers    []audioLayer
	output    tlinear
	proj      tlinear // into the text's space
	clipMin   mlx.Array
	clipMax   mlx.Array
	keep      []mlx.Array // what the loader made: freed with the model
}

type audioFF struct {
	pre, post mlx.Array
	up, down  tlinear
}

type audioLayer struct {
	ff1, ff2                 audioFF
	q, k, v, post            tlinear
	qScale, relK             mlx.Array // softplus(per_dim_scale) f32; [1, heads, headDim, positions] f32
	lcPre, lcNorm            mlx.Array
	lcStart, lcEnd           tlinear
	dw                       mlx.Array // [hidden, kernel, 1]
	preAttn, postAttn, outNm mlx.Array
}

// loadAudio reads the tower under audio_tower. and its projection.
func loadAudio(t map[string]mlx.Array, cfg audioConfig) (*audioTower, error) {
	cfg.defaults()
	w := &towerWeights{t: t}
	at := &audioTower{cfg: cfg}
	// What is derived from the weights here is the model's, and freed with
	// it; whatever it took to derive it goes with the scope.
	sc := mlx.NewScope()
	defer sc.Close()
	keep := func(a mlx.Array) mlx.Array { sc.Escape(a); at.keep = append(at.keep, a); return a }
	for i := range at.convs {
		p := fmt.Sprintf("audio_tower.subsample_conv_projection.layer%d.", i)
		if c := w.arr(p + "conv.weight"); c != 0 {
			at.convs[i].w = keep(mlx.Transpose(c, 0, 2, 3, 1)) // [out, in, kH, kW] → channels last
		}
		at.convs[i].norm = w.arr(p + "norm.weight")
	}
	at.inputProj = w.linear("audio_tower.subsample_conv_projection.input_proj_linear")

	table := relPosTable(cfg)
	headDim := cfg.Hidden / cfg.Heads
	positions := table.Shape()[0]
	tableBF := mlx.Astype(table, mlx.Bfloat16)
	for i := 0; i < cfg.Layers; i++ {
		p := fmt.Sprintf("audio_tower.layers.%d.", i)
		ff := func(name string) audioFF {
			return audioFF{pre: w.arr(p + name + ".pre_layer_norm.weight"), up: w.linear(p + name + ".ffw_layer_1"),
				down: w.linear(p + name + ".ffw_layer_2"), post: w.arr(p + name + ".post_layer_norm.weight")}
		}
		l := audioLayer{ff1: ff("feed_forward1"), ff2: ff("feed_forward2"),
			q: w.linear(p + "self_attn.q_proj"), k: w.linear(p + "self_attn.k_proj"),
			v: w.linear(p + "self_attn.v_proj"), post: w.linear(p + "self_attn.post"),
			lcPre: w.arr(p + "lconv1d.pre_layer_norm.weight"), lcStart: w.linear(p + "lconv1d.linear_start"),
			lcNorm: w.arr(p + "lconv1d.conv_norm.weight"), lcEnd: w.linear(p + "lconv1d.linear_end"),
			preAttn: w.arr(p + "norm_pre_attn.weight"), postAttn: w.arr(p + "norm_post_attn.weight"),
			outNm: w.arr(p + "norm_out.weight")}
		if s := w.arr(p + "self_attn.per_dim_scale"); s != 0 {
			// The reference takes the softplus, and runs the table through
			// relative_k_proj, in the checkpoint's dtype, then lifts to f32.
			l.qScale = keep(mlx.Astype(mlx.Softplus(s), mlx.Float32))
		}
		if rel := w.linear(p + "self_attn.relative_k_proj"); rel.w != 0 {
			r := mlx.Reshape(rel.apply(tableBF), positions, cfg.Heads, headDim)
			l.relK = keep(mlx.Astype(mlx.ExpandDims(mlx.Transpose(r, 1, 2, 0), 0), mlx.Float32))
		}
		if dw := w.arr(p + "lconv1d.depthwise_conv1d.weight"); dw != 0 {
			l.dw = keep(mlx.Transpose(dw, 0, 2, 1)) // [hidden, 1, kernel] → [hidden, kernel, 1]
		}
		at.layers = append(at.layers, l)
	}
	at.output = w.linear("audio_tower.output_proj")
	at.proj = w.linear("embed_audio.embedding_projection")
	if w.missing != "" {
		at.free()
		return nil, fmt.Errorf("no tensor %s", w.missing)
	}
	at.clipMin = keep(mlx.ScalarAs(-cfg.Clipping, mlx.Bfloat16))
	at.clipMax = keep(mlx.ScalarAs(cfg.Clipping, mlx.Bfloat16))
	// What was derived is computed now, once, rather than at every sound.
	mlx.Eval(at.keep...)
	return at, nil
}

func (at *audioTower) free() {
	for _, a := range at.keep {
		a.Free()
	}
	at.keep = nil
}

// relPosTable is the [positions, hidden] sinusoidal table over relative
// positions from half the context down to 0, each row [sin…, cos…].
func relPosTable(a audioConfig) mlx.Array {
	positions := (a.Chunk+a.ContextLeft-1+a.ContextRight)/2 + 1
	half := a.Hidden / 2
	inc := math.Log(10000) / float64(half-1)
	table := make([]float32, positions*a.Hidden)
	for i := 0; i < positions; i++ {
		p := float64(positions - 1 - i)
		for d := 0; d < half; d++ {
			angle := p * math.Exp(-float64(d)*inc)
			table[i*a.Hidden+d] = float32(math.Sin(angle))
			table[i*a.Hidden+half+d] = float32(math.Cos(angle))
		}
	}
	return mlx.FromFloat32(table, positions, a.Hidden)
}

func (at *audioTower) clip(x mlx.Array) mlx.Array { return clip(x, at.clipMin, at.clipMax) }

// encode runs the tower over one piece's [frames, mel] features: [tokens,
// text hidden].
func (at *audioTower) encode(mel []float32, frames int) mlx.Array {
	c := at.cfg
	h := mlx.Astype(mlx.FromFloat32(mel, 1, frames, audioMelBins, 1), mlx.Bfloat16)
	for _, cv := range at.convs {
		h = mlx.Conv2d(h, cv.w, 2, 1)
		h = mlx.LayerNorm(h, cv.norm, 0, c.Eps)
		h = mlx.Maximum(h, mlx.ScalarAs(0, h.Dtype()))
	}
	sh := h.Shape() // [1, n, w, channels]
	n := sh[1]
	h = at.inputProj.apply(mlx.Reshape(h, n, sh[2]*sh[3]))

	mask := audioMask(n, c)
	idx := audioIndices((n+c.Chunk-1)/c.Chunk, c)
	for i := range at.layers {
		h = at.layers[i].forward(at, h, mask, idx)
	}
	h = mlx.RMSNorm(at.output.apply(h), 0, c.Eps)
	return at.proj.apply(h)
}

func (f audioFF) forward(at *audioTower, h mlx.Array) mlx.Array {
	x := mlx.RMSNorm(at.clip(h), f.pre, 1e-6)
	x = f.down.apply(mlx.SiLU(f.up.apply(x)))
	x = mlx.RMSNorm(at.clip(x), f.post, 1e-6)
	return mlx.Add(h, mlx.MulScalar(x, at.cfg.Residual))
}

func (l *audioLayer) forward(at *audioTower, h, mask, idx mlx.Array) mlx.Array {
	h = l.ff1.forward(at, h)
	x := mlx.RMSNorm(at.clip(h), l.preAttn, 1e-6)
	x = l.attend(at, x, mask, idx)
	h = mlx.Add(h, mlx.RMSNorm(at.clip(x), l.postAttn, 1e-6))
	h = l.lightConv(at, h)
	h = l.ff2.forward(at, h)
	return mlx.RMSNorm(at.clip(h), l.outNm, 1e-6)
}

// attend is chunked local attention with relative positions: queries in
// blocks of chunk frames, each block's keys and values its window of context,
// the position scores shifted to offsets, then soft-capped and masked.
func (l *audioLayer) attend(at *audioTower, h, mask, idx mlx.Array) mlx.Array {
	c := at.cfg
	n := h.Shape()[0]
	heads, hd := c.Heads, c.Hidden/c.Heads
	chunk, left, right := c.Chunk, c.ContextLeft-1, c.ContextRight
	ctx := chunk + left + right
	blocks := (n + chunk - 1) / chunk

	proj := func(p tlinear) mlx.Array {
		return mlx.Reshape(mlx.Astype(p.apply(h), mlx.Float32), n, heads, hd)
	}
	q, k, v := proj(l.q), proj(l.k), proj(l.v)
	q = mlx.Multiply(mlx.MulScalar(q, float32(1/(math.Sqrt(float64(hd))*math.Ln2))), l.qScale)
	k = mlx.MulScalar(k, float32(math.Log(1+math.E)/math.Ln2))

	q = pad(q, 0, 0, blocks*chunk-n)
	qb := mlx.Transpose(mlx.Reshape(q, blocks, chunk, heads, hd), 0, 2, 1, 3)
	gather := func(x mlx.Array) mlx.Array {
		x = pad(x, 0, left, blocks*chunk-n+right)
		x = mlx.Reshape(mlx.Take(x, idx, 0), blocks, ctx, heads, hd)
		return mlx.Transpose(x, 0, 2, 1, 3)
	}
	kb, vb := gather(k), gather(v)

	ac := mlx.Matmul(qb, mlx.Transpose(kb, 0, 1, 3, 2))
	bd := mlx.Matmul(qb, l.relK) // [blocks, heads, chunk, positions]
	positions := bd.Shape()[3]
	bd = pad(bd, 3, 0, ctx+1-positions)
	bd = mlx.Reshape(bd, blocks, heads, chunk*(ctx+1))
	bd = mlx.Slice(bd, []int{0, 0, 0}, []int{blocks, heads, chunk * ctx})
	bd = mlx.Reshape(bd, blocks, heads, chunk, ctx)

	logits := mlx.Add(ac, bd)
	logits = mlx.MulScalar(mlx.Tanh(mlx.MulScalar(logits, 1/c.LogitCap)), c.LogitCap)
	logits = mlx.Where(mask, logits, mlx.Scalar(c.InvalidLogit))
	o := mlx.Matmul(mlx.Softmax(logits, -1), vb)
	o = mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), blocks*chunk, c.Hidden)
	o = mlx.Slice(o, []int{0, 0}, []int{n, c.Hidden})
	return l.post.apply(mlx.Astype(o, mlx.Bfloat16))
}

func (l *audioLayer) lightConv(at *audioTower, h mlx.Array) mlx.Array {
	c := at.cfg
	n, hid := h.Shape()[0], c.Hidden
	x := l.lcStart.apply(mlx.RMSNorm(h, l.lcPre, c.Eps))
	x = mlx.Multiply(mlx.Slice(x, []int{0, 0}, []int{n, hid}), mlx.Sigmoid(mlx.Slice(x, []int{0, hid}, []int{n, 2 * hid})))
	// Causal depthwise convolution: time padded on the left only.
	x = pad(mlx.Reshape(x, 1, n, hid), 1, c.ConvKernel-1, 0)
	x = mlx.Reshape(mlx.Conv1d(x, l.dw, 1, 0, 1, hid), n, hid)
	x = mlx.SiLU(mlx.RMSNorm(at.clip(x), l.lcNorm, c.Eps))
	return mlx.Add(h, l.lcEnd.apply(x))
}

// audioMask is [blocks, 1, chunk, context]: block b's context slot j is frame
// b*chunk + j - (left-1), attended when it exists and lies in the window.
func audioMask(n int, c audioConfig) mlx.Array {
	chunk, left, right := c.Chunk, c.ContextLeft-1, c.ContextRight
	ctx := chunk + left + right
	blocks := (n + chunk - 1) / chunk
	vals := make([]float32, blocks*chunk*ctx)
	i := 0
	for b := 0; b < blocks; b++ {
		for q := 0; q < chunk; q++ {
			gq := b*chunk + q
			for j := 0; j < ctx; j++ {
				gk := b*chunk + j - left
				d := gq - gk
				if gq < n && gk >= 0 && gk < n && ((d >= 0 && d < left) || (d < 0 && -d < right)) {
					vals[i] = 1
				}
				i++
			}
		}
	}
	return mlx.Greater(mlx.FromFloat32(vals, blocks, 1, chunk, ctx), mlx.Scalar(0))
}

// audioIndices maps block b's context slot j to b*chunk+j of the padded
// keys and values.
func audioIndices(blocks int, c audioConfig) mlx.Array {
	ctx := c.Chunk + c.ContextLeft - 1 + c.ContextRight
	idx := make([]int32, blocks*ctx)
	for b := 0; b < blocks; b++ {
		for j := 0; j < ctx; j++ {
			idx[b*ctx+j] = int32(b*c.Chunk + j)
		}
	}
	return mlx.FromInt32(idx, blocks*ctx)
}

// audioPiece is one piece of a sound, its features computed: [frames, mel].
type audioPiece struct {
	mel    []float32
	frames int
	tokens int
}

// prepareAudio decodes a WAV and makes its pieces.
func prepareAudio(data []byte) ([]audioPiece, error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		wav, err := toWAV(data) // mp3, m4a, aac, ogg
		if err != nil {
			return nil, fmt.Errorf("not a sound this reads: %w", err)
		}
		data = wav
	}
	samples, rate, err := decodeWAV(data)
	if err != nil {
		return nil, err
	}
	samples = resample(samples, rate, audioRate)
	if len(samples) < audioFrameLen+1-audioFrameLen/2 {
		return nil, errors.New("the sound is too short")
	}
	var out []audioPiece
	for _, s := range splitQuiet(samples, audioRate, audioPieceSecs) {
		mel, frames := melSpectrogram(s)
		if frames == 0 {
			continue
		}
		tokens := frames
		for range 2 {
			tokens = (tokens-1)/2 + 1
		}
		out = append(out, audioPiece{mel: mel, frames: frames, tokens: tokens})
	}
	if len(out) == 0 {
		return nil, errors.New("the sound is too short")
	}
	return out, nil
}

// decodeWAV reads a WAV's samples, its channels averaged, at its own rate:
// 8, 16, 24 or 32-bit PCM, or 32-bit float; at most audioMaxSeconds.
func decodeWAV(data []byte) ([]float32, int, error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, errors.New("not a WAV: sound is read as WAV, as Ollama reads it")
	}
	var format uint16
	var channels, rate, bits int
	var pcm []byte
	fmtSeen := false
	for off := 12; off+8 <= len(data); {
		id := string(data[off : off+4])
		size := int(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		body := data[off+8 : min(off+8+size, len(data))]
		switch id {
		case "fmt ":
			if len(body) < 16 {
				return nil, 0, errors.New("wav: fmt chunk too short")
			}
			format = binary.LittleEndian.Uint16(body[0:2])
			channels = int(binary.LittleEndian.Uint16(body[2:4]))
			rate = int(binary.LittleEndian.Uint32(body[4:8]))
			bits = int(binary.LittleEndian.Uint16(body[14:16]))
			if format == 0xfffe && len(body) >= 26 {
				format = binary.LittleEndian.Uint16(body[24:26])
			}
			fmtSeen = true
		case "data":
			pcm = body
		}
		off += 8 + size + size%2
	}
	switch {
	case !fmtSeen:
		return nil, 0, errors.New("wav: no fmt chunk")
	case pcm == nil:
		return nil, 0, errors.New("wav: no data chunk")
	case channels < 1 || rate < 1:
		return nil, 0, fmt.Errorf("wav: %d channels at %d Hz", channels, rate)
	case rate > audioMaxRate:
		return nil, 0, fmt.Errorf("wav: %d Hz, more than %d", rate, audioMaxRate)
	}
	var sample func([]byte) float64
	switch {
	case format == 1 && bits == 8:
		sample = func(b []byte) float64 { return (float64(b[0]) - 128) / 128 }
	case format == 1 && bits == 16:
		sample = func(b []byte) float64 { return float64(int16(binary.LittleEndian.Uint16(b))) / 32768 }
	case format == 1 && bits == 24:
		sample = func(b []byte) float64 {
			v := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16
			return float64(v<<8>>8) / 8388608
		}
	case format == 1 && bits == 32:
		sample = func(b []byte) float64 { return float64(int32(binary.LittleEndian.Uint32(b))) / 2147483648 }
	case format == 3 && bits == 32:
		sample = func(b []byte) float64 { return float64(math.Float32frombits(binary.LittleEndian.Uint32(b))) }
	default:
		return nil, 0, fmt.Errorf("wav: format %d with %d bits is not read", format, bits)
	}
	stride := bits / 8
	n := len(pcm) / (stride * channels)
	if n > audioMaxSeconds*rate {
		return nil, 0, fmt.Errorf("the sound is longer than %d seconds", audioMaxSeconds)
	}
	mono := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float64
		for c := 0; c < channels; c++ {
			sum += sample(pcm[(i*channels+c)*stride:])
		}
		mono[i] = float32(sum / float64(channels))
	}
	return mono, rate, nil
}

const (
	resampleLobes   = 16
	resampleRolloff = 0.9
	resampleBeta    = 9.0
)

var resampleKernel = sync.OnceValue(func() []float64 {
	const size = 8192
	i0 := besselI0(resampleBeta)
	k := make([]float64, size+1)
	for i := range k {
		u := float64(i) / size
		k[i] = sinc(resampleLobes*u) * besselI0(resampleBeta*math.Sqrt(1-u*u)) / i0
	}
	return k
})

// resample is a Kaiser-windowed sinc at each output sample's place in the
// source, normalized per sample.
func resample(s []float32, from, to int) []float32 {
	if from == to || len(s) < 2 {
		return s
	}
	ratio := float64(to) / float64(from)
	scale := resampleRolloff * min(1, ratio)
	half := resampleLobes / scale
	kernel := resampleKernel()
	n := int((int64(len(s))*int64(to) + int64(from) - 1) / int64(from))
	out := make([]float32, n)
	for i := range out {
		center := float64(i) / ratio
		lo := max(int(math.Ceil(center-half)), 0)
		hi := min(int(math.Floor(center+half)), len(s)-1)
		var acc, norm float64
		for j := lo; j <= hi; j++ {
			pos := math.Abs(float64(j)-center) / half * float64(len(kernel)-1)
			k := int(pos)
			if k >= len(kernel)-1 {
				continue
			}
			w := kernel[k] + (kernel[k+1]-kernel[k])*(pos-float64(k))
			acc += w * float64(s[j])
			norm += w
		}
		if norm != 0 {
			out[i] = float32(acc / norm)
		}
	}
	return out
}

// splitQuiet divides samples into the fewest pieces of at most maxSecs, each
// cut moved to the quietest 100 ms within 4 s of its even share.
func splitQuiet(s []float32, rate, maxSecs int) [][]float32 {
	maxPiece := rate * maxSecs
	span := rate * 4
	window := max(rate*100/1000, 1)
	var out [][]float32
	for k := (len(s) + maxPiece - 1) / maxPiece; k > 1; k-- {
		target := len(s) / k
		lo := max(target-span/2, len(s)-(k-1)*maxPiece)
		hi := min(target+span/2, maxPiece)
		cut := lo + quietest(s[lo:hi], window)
		out = append(out, s[:cut])
		s = s[cut:]
	}
	return append(out, s)
}

func quietest(x []float32, window int) int {
	if len(x) <= window {
		return len(x) / 2
	}
	var e float64
	for _, v := range x[:window] {
		e += float64(v) * float64(v)
	}
	best, bestE := 0, e
	for i := window; i < len(x); i++ {
		e += float64(x[i])*float64(x[i]) - float64(x[i-window])*float64(x[i-window])
		if e <= bestE {
			best, bestE = i-window+1, e
		}
	}
	return best + window/2
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}

func besselI0(x float64) float64 {
	sum, term := 1.0, 1.0
	for k := 1; k < 64; k++ {
		term *= x * x / (4 * float64(k) * float64(k))
		sum += term
		if term < 1e-16*sum {
			break
		}
	}
	return sum
}

var audioWindow = sync.OnceValue(func() []float32 {
	w := make([]float32, audioFrameLen)
	for i := range w {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/audioFrameLen))
	}
	return w
})

// melSpectrogram is [frames, mel] log-mel features with half a frame of
// zeros before the first sample, so the first frame is centred at 0.
func melSpectrogram(s []float32) ([]float32, int) {
	p := audioFrameLen / 2
	if len(s)+p < audioFrameLen+1 {
		return nil, 0
	}
	frames := (len(s)+p-(audioFrameLen+1))/audioHopLen + 1
	padded := make([]float32, p+len(s))
	copy(padded[p:], s)
	window, filters := audioWindow(), melFilters()
	bins := audioFFTLen/2 + 1
	out := make([]float32, frames*audioMelBins)
	buf := make([]complex128, audioFFTLen)
	mags := make([]float64, bins)
	for f := 0; f < frames; f++ {
		frame := padded[f*audioHopLen:]
		for i := 0; i < audioFrameLen; i++ {
			buf[i] = complex(float64(frame[i]*window[i]), 0)
		}
		for i := audioFrameLen; i < audioFFTLen; i++ {
			buf[i] = 0
		}
		fft(buf)
		for k := 0; k < bins; k++ {
			mags[k] = cmplx.Abs(buf[k])
		}
		for m := 0; m < audioMelBins; m++ {
			var mel float64
			for k := 0; k < bins; k++ {
				mel += mags[k] * filters[m*bins+k]
			}
			out[f*audioMelBins+m] = float32(math.Log(mel + audioMelFloor))
		}
	}
	return out, frames
}

// melFilters is the HTK-scale triangular filterbank, [mel, bin], unnormalized.
var melFilters = sync.OnceValue(func() []float64 {
	toMel := func(f float64) float64 { return 2595 * math.Log10(1+f/700) }
	toHz := func(m float64) float64 { return 700 * (math.Pow(10, m/2595) - 1) }
	top := toMel(audioMaxFreq)
	corners := make([]float64, audioMelBins+2)
	for i := range corners {
		corners[i] = toHz(float64(i) * top / float64(audioMelBins+1))
	}
	bins := audioFFTLen/2 + 1
	f := make([]float64, audioMelBins*bins)
	for m := 0; m < audioMelBins; m++ {
		l, c, r := corners[m], corners[m+1], corners[m+2]
		for k := 0; k < bins; k++ {
			hz := float64(k) * audioRate / audioFFTLen
			if v := min((hz-l)/(c-l), (r-hz)/(r-c)); v > 0 {
				f[m*bins+k] = v
			}
		}
	}
	return f
})

// fft is an in-place radix-2 Cooley–Tukey FFT.
func fft(x []complex128) {
	n := len(x)
	if n <= 1 {
		return
	}
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for j&bit != 0 {
			j ^= bit
			bit >>= 1
		}
		j ^= bit
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		half := size / 2
		w := cmplx.Exp(complex(0, -2*math.Pi/float64(size)))
		for start := 0; start < n; start += size {
			wn := complex(1, 0)
			for k := 0; k < half; k++ {
				t := wn * x[start+k+half]
				x[start+k+half] = x[start+k] - t
				x[start+k] = x[start+k] + t
				wn *= w
			}
		}
	}
}
