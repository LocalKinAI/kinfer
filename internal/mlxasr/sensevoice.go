package mlxasr

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlx"
)

// SenseVoice Small, FunAudioLLM's speech model, on MLX, following mlx-audio's
// sensevoice.py: Kaldi's log-mel filterbank, seven frames stacked every six
// (LFR) and normalized by the model's am.mvn (CMVN), four query vectors in
// front — the language, the event and the emotion, inverse text normalization
// — 70 SANM layers (self-attention beside a depthwise convolution, FSMN), and a
// CTC head: the first four positions name the language, the emotion and the
// sound heard, the rest are the transcript, greedily.
//
// Two things differ from mlx-audio on purpose. mlx-audio does not fetch
// am.mvn with the MLX conversion and so runs without CMVN; this applies it,
// as FunASR does. And FunASR dithers the samples by default (σ = 1 on the
// 16-bit scale, seeded), which this does not: what it computes is mlx-audio's
// arithmetic with the normalization the model was trained with.

type svConfig struct {
	ModelType string `json:"model_type"`
	Vocab     int    `json:"vocab_size"`
	Input     int    `json:"input_size"`
	Enc       struct {
		Out    int `json:"output_size"`
		Heads  int `json:"attention_heads"`
		Units  int `json:"linear_units"`
		Blocks int `json:"num_blocks"`
		TP     int `json:"tp_blocks"`
		Kernel int `json:"kernel_size"`
		Shift  int `json:"sanm_shift"`
	} `json:"encoder_conf"`
	Front struct {
		FS         int    `json:"fs"`
		Window     string `json:"window"`
		Mels       int    `json:"n_mels"`
		FrameLen   int    `json:"frame_length"`
		FrameShift int    `json:"frame_shift"`
		LfrM       int    `json:"lfr_m"`
		LfrN       int    `json:"lfr_n"`
	} `json:"frontend_conf"`
}

func readSVConfig(dir string) (*svConfig, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c svConfig
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("%s/config.json: %w", dir, err)
	}
	if c.ModelType != "sensevoice" {
		return nil, fmt.Errorf("%s is a %q model, not SenseVoice", dir, c.ModelType)
	}
	return &c, nil
}

type sanmLayer struct {
	ln1w, ln1b, ln2w, ln2b mlx.Array
	qkv, out, w1, w2       linear
	fsmn                   mlx.Array // [d, K, 1], depthwise
	in                     int
}

// SenseVoice is a loaded SenseVoice model.
type SenseVoice struct {
	cfg              *svConfig
	layers           []sanmLayer // encoders0, then encoders, then tp_encoders
	afterW, afterB   mlx.Array
	tpW, tpB         mlx.Array
	ctc              linear
	embed            mlx.Array // [16, input]
	means, istd      mlx.Array // CMVN, [input]
	pieces           []string
	bank             mlx.Array // [nfft/2+1, mels]
	window           []float32
	winLen, shift    int
	nfft             int
	all              []mlx.Array
	cmvn             bool
	langID           map[string]int32
	queryIDs         [2]int32 // the event and emotion queries
	itnID, woitnID   int32
	blank            int32
	maxChunkPosition int
}

const svLayerNormEps = 1e-5

// LoadSenseVoice reads a SenseVoice directory as MLX releases one, with the
// am.mvn of the original release beside it. mlx.Open must have been called.
func LoadSenseVoice(dir string) (s *SenseVoice, err error) {
	defer mlx.Recover(&err)
	if err := mlx.AudioOps(); err != nil {
		return nil, err
	}
	cfg, err := readSVConfig(dir)
	if err != nil {
		return nil, err
	}
	s = &SenseVoice{cfg: cfg, blank: 0, itnID: 14, woitnID: 15, queryIDs: [2]int32{1, 2},
		langID: map[string]int32{"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}}
	models, _ := filepath.Glob(filepath.Join(dir, "*.model"))
	if len(models) == 0 {
		return nil, fmt.Errorf("%s has no SentencePiece model (*.model) to decode with", dir)
	}
	if s.pieces, err = readPieces(models[0]); err != nil {
		return nil, err
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	w := &weights{t: map[string]mlx.Array{}, taken: map[string]bool{}}
	for _, f := range files {
		t, err := mlx.LoadSafetensors(f)
		if err != nil {
			return nil, err
		}
		for k, v := range t {
			w.t[k] = v
			s.all = append(s.all, v)
		}
	}
	fail := func(err error) (*SenseVoice, error) { s.Free(); return nil, err }
	get := func(name string) mlx.Array {
		a, gerr := w.get(name)
		if gerr != nil && err == nil {
			err = gerr
		}
		return a
	}
	lin := func(name string) linear {
		l, lerr := w.lin(name)
		if lerr != nil && err == nil {
			err = lerr
		}
		return l
	}
	layer := func(p string, in int) sanmLayer {
		l := sanmLayer{in: in,
			ln1w: get(p + "norm1.weight"), ln1b: get(p + "norm1.bias"), ln2w: get(p + "norm2.weight"), ln2b: get(p + "norm2.bias"),
			qkv: lin(p + "self_attn.linear_q_k_v"), out: lin(p + "self_attn.linear_out"),
			w1: lin(p + "feed_forward.w_1"), w2: lin(p + "feed_forward.w_2")}
		f := get(p + "self_attn.fsmn_block.weight")
		if f != 0 {
			if sh := f.Shape(); len(sh) == 3 && sh[1] == 1 { // PyTorch's [d, 1, K] → MLX's [d, K, 1]
				f = mlx.Contiguous(mlx.Transpose(f, 0, 2, 1))
				mlx.Eval(f)
				s.all = append(s.all, f)
			}
			l.fsmn = f
		}
		return l
	}
	E := cfg.Enc
	s.layers = append(s.layers, layer("encoder.encoders0.0.", cfg.Input))
	for i := 0; i < E.Blocks-1; i++ {
		s.layers = append(s.layers, layer(fmt.Sprintf("encoder.encoders.%d.", i), E.Out))
	}
	for i := 0; i < E.TP; i++ {
		s.layers = append(s.layers, layer(fmt.Sprintf("encoder.tp_encoders.%d.", i), E.Out))
	}
	s.afterW, s.afterB = get("encoder.after_norm.weight"), get("encoder.after_norm.bias")
	s.tpW, s.tpB = get("encoder.tp_norm.weight"), get("encoder.tp_norm.bias")
	s.embed = get("embed.weight")
	if _, ok := w.t["ctc.ctc_lo.weight"]; ok {
		s.ctc = lin("ctc.ctc_lo")
	} else {
		s.ctc = lin("ctc_lo")
	}
	if err != nil {
		return fail(err)
	}

	// CMVN, from the original release's am.mvn.
	if means, istd, cerr := readAmMvn(filepath.Join(dir, "am.mvn")); cerr == nil {
		if len(means) != cfg.Input || len(istd) != cfg.Input {
			return fail(fmt.Errorf("am.mvn has %d and %d values, the model reads %d", len(means), len(istd), cfg.Input))
		}
		s.means, s.istd, s.cmvn = mlx.FromFloat32(means, cfg.Input), mlx.FromFloat32(istd, cfg.Input), true
		s.all = append(s.all, s.means, s.istd)
	} else if !os.IsNotExist(cerr) {
		return fail(cerr)
	}

	// The frame's window, the FFT size and Kaldi's mel filters.
	F := cfg.Front
	s.winLen, s.shift = F.FS*F.FrameLen/1000, F.FS*F.FrameShift/1000
	s.nfft = 1
	for s.nfft < s.winLen {
		s.nfft <<= 1
	}
	s.window = make([]float32, s.winLen)
	for i := range s.window {
		x := 2 * math.Pi * float64(i) / float64(s.winLen-1)
		switch F.Window {
		case "hanning":
			s.window[i] = float32(0.5 - 0.5*math.Cos(x))
		case "povey":
			s.window[i] = float32(math.Pow(0.5-0.5*math.Cos(x), 0.85))
		case "rectangular":
			s.window[i] = 1
		default: // hamming
			s.window[i] = float32(0.54 - 0.46*math.Cos(x))
		}
	}
	s.bank = mlx.FromFloat32(kaldiMelBank(F.Mels, s.nfft, float64(F.FS), 20, 0), s.nfft/2+1, F.Mels)
	s.all = append(s.all, s.bank)
	// Two minutes a pass: attention over them is 64 MB a layer, and a recording
	// that short is read whole, as mlx-audio reads every one.
	s.maxChunkPosition = 120 * F.FS / s.shift / F.LfrN
	return s, nil
}

func readAmMvn(path string) (means, istd []float32, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	block := func(name string) ([]float32, error) {
		m := regexp.MustCompile(`(?s)<` + name + `>.*?<LearnRateCoef>\s+\S+\s+\[(.*?)\]`).FindSubmatch(b)
		if m == nil {
			return nil, fmt.Errorf("%s: no <%s>", path, name)
		}
		var out []float32
		for _, f := range strings.Fields(string(m[1])) {
			v, err := strconv.ParseFloat(f, 32)
			if err != nil {
				return nil, fmt.Errorf("%s: %q", path, f)
			}
			out = append(out, float32(v))
		}
		return out, nil
	}
	if means, err = block("AddShift"); err != nil {
		return nil, nil, err
	}
	istd, err = block("Rescale")
	return means, istd, err
}

// kaldiMelBank is Kaldi's mel filterbank, [nfft/2+1, mels], the last row
// (Nyquist) zero, as mlx-audio's get_mel_banks_kaldi padded.
func kaldiMelBank(mels, nfft int, fs, low, high float64) []float32 {
	mel := func(f float64) float64 { return 1127 * math.Log(1+f/700) }
	if high <= 0 {
		high += fs / 2
	}
	bins := nfft / 2
	width := fs / float64(nfft)
	lo, hi := mel(low), mel(high)
	delta := (hi - lo) / float64(mels+1)
	out := make([]float32, (bins+1)*mels)
	for b := 0; b < mels; b++ {
		left, center, right := lo+float64(b)*delta, lo+float64(b+1)*delta, lo+float64(b+2)*delta
		for i := 0; i < bins; i++ {
			m := mel(width * float64(i))
			up := (m - left) / (center - left)
			down := (right - m) / (right - center)
			out[i*mels+b] = float32(math.Max(0, math.Min(up, down)))
		}
	}
	return out
}

// Free releases the weights.
func (s *SenseVoice) Free() {
	for _, a := range s.all {
		a.Free()
	}
	s.all = nil
}

// features is the model's input for pcm, 16 kHz mono: [positions, input].
func (s *SenseVoice) features(pcm []float32) (mlx.Array, error) {
	F := s.cfg.Front
	n := len(pcm)
	if n < s.winLen {
		return 0, fmt.Errorf("the recording is too short: %d samples", n)
	}
	T := 1 + (n-s.winLen)/s.shift
	frames := make([]float32, T*s.nfft)
	x := make([]float64, s.winLen)
	for t := 0; t < T; t++ {
		var mean float64
		for i := range x {
			x[i] = float64(pcm[t*s.shift+i]) * 32768
			mean += x[i]
		}
		mean /= float64(len(x))
		row := frames[t*s.nfft : t*s.nfft+s.winLen]
		prev := 0.0
		for i := range x {
			v := x[i] - mean
			p := v
			if i > 0 {
				p = v - 0.97*prev
			}
			prev = v
			row[i] = float32(p) * s.window[i]
		}
	}
	spec := mlx.Power2(mlx.RFFT(mlx.FromFloat32(frames, T, s.nfft), s.nfft, -1)) // [T, nfft/2+1]
	fb := mlx.Log(mlx.Maximum(mlx.Matmul(spec, s.bank), mlx.Scalar(1e-8)))       // [T, mels]

	// LFR: m frames stacked every n, the first frame repeated (m-1)/2 times in
	// front and the last behind.
	m, step := F.LfrM, F.LfrN
	L := (T + step - 1) / step
	idx := make([]int32, L*m)
	for i := 0; i < L; i++ {
		for j := 0; j < m; j++ {
			src := i*step + j - (m-1)/2
			idx[i*m+j] = int32(min(max(src, 0), T-1))
		}
	}
	feats := mlx.Reshape(mlx.Take(fb, mlx.FromInt32(idx, L*m), 0), L, m*F.Mels)
	if s.cmvn {
		feats = mlx.Multiply(mlx.Add(feats, s.means), s.istd)
	}
	return feats, nil
}

// logProbs is the CTC head's log-probabilities for feats [positions, input],
// the four query positions first: [positions + 4, vocab].
func (s *SenseVoice) logProbs(feats mlx.Array, lang int32, itn bool) mlx.Array {
	E := s.cfg.Enc
	tn := s.woitnID
	if itn {
		tn = s.itnID
	}
	q := mlx.Take(s.embed, mlx.FromInt32([]int32{lang, s.queryIDs[0], s.queryIDs[1], tn}, 4), 0)
	x := mlx.Concatenate(0, q, feats)
	L, D := x.Shape()[0], x.Shape()[1]
	x = mlx.MulScalar(mlx.Reshape(x, 1, L, D), float32(math.Sqrt(float64(E.Out))))
	half := D / 2
	inc := math.Log(10000) / float64(half-1)
	pos := make([]float32, L*D)
	for t := 0; t < L; t++ {
		for i := 0; i < half; i++ {
			a := float64(t+1) * math.Exp(-inc*float64(i))
			pos[t*D+i] = float32(math.Sin(a))
			pos[t*D+half+i] = float32(math.Cos(a))
		}
	}
	x = mlx.Add(x, mlx.FromFloat32(pos, 1, L, D))

	owned := false
	for i := range s.layers {
		sc := mlx.NewScope()
		next := s.layer(&s.layers[i], x)
		if i == E.Blocks-1 {
			next = mlx.LayerNorm(next, s.afterW, s.afterB, svLayerNormEps)
		}
		sc.Escape(next)
		sc.Close()
		if owned {
			x.Free()
		}
		x, owned = next, true
	}
	h := mlx.LayerNorm(x, s.tpW, s.tpB, svLayerNormEps)
	if owned {
		x.Free()
	}
	logits := mlx.Reshape(s.ctc.apply(h), L, s.cfg.Vocab)
	return mlx.Subtract(logits, mlx.Logsumexp(logits, -1, true))
}

func (s *SenseVoice) layer(l *sanmLayer, x mlx.Array) mlx.Array {
	E := s.cfg.Enc
	L, d := x.Shape()[1], E.Out
	y := mlx.LayerNorm(x, l.ln1w, l.ln1b, svLayerNormEps)
	qkv := mlx.Split(l.qkv.apply(y), -1, d, 2*d)
	q, k, v := qkv[0], qkv[1], qkv[2]

	// FSMN: the values through a depthwise convolution, padded to keep the length.
	left := (E.Kernel-1)/2 + E.Shift
	right := E.Kernel - 1 - left
	vp := v
	if left > 0 || right > 0 {
		vp = mlx.Concatenate(1, mlx.Zeros(v.Dtype(), 1, left, d), v, mlx.Zeros(v.Dtype(), 1, right, d))
	}
	mem := mlx.Add(mlx.Conv1d(vp, l.fsmn, 1, 0, 1, d), v)

	H, dk := E.Heads, d/E.Heads
	heads := func(a mlx.Array) mlx.Array { return mlx.Transpose(mlx.Reshape(a, 1, L, H, dk), 0, 2, 1, 3) }
	o := mlx.SDPA(heads(q), heads(k), heads(v), float32(1/math.Sqrt(float64(dk))), "", 0)
	att := mlx.Add(l.out.apply(mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), 1, L, d)), mem)
	if l.in == d {
		x = mlx.Add(x, att)
	} else {
		x = att
	}
	y = mlx.LayerNorm(x, l.ln2w, l.ln2b, svLayerNormEps)
	return mlx.Add(x, l.w2.apply(mlx.Maximum(l.w1.apply(y), mlx.Scalar(0))))
}

// Heard is what SenseVoice heard in a recording.
type Heard struct {
	Text     string
	Language string   // zh, en, yue, ja, ko, nospeech
	Emotion  string   // happy, sad, angry, neutral, fearful, disgusted, surprised, other; "" for unknown
	Events   []string // sounds besides speech: BGM, Laughter, Applause, Cry, Sneeze, Breath, Cough
	Tokens   int
	Took     time.Duration
}

func tagName(p string) string { return strings.TrimSuffix(strings.TrimPrefix(p, "<|"), "|>") }

// Hear runs SenseVoice over pcm, 16 kHz mono, two minutes a pass:
// the transcripts joined, the emotion the one heard most (ties to the first),
// and every sound besides speech that was heard. language is "auto" or one of
// zh, en, yue, ja, ko.
func (s *SenseVoice) Hear(ctx context.Context, pcm []float32, language string, itn bool) (h Heard, err error) {
	defer mlx.Recover(&err)
	t0 := time.Now()
	lang, ok := s.langID[language]
	if !ok {
		lang = 0
	}
	chunk := s.maxChunkPosition * s.cfg.Front.LfrN * s.shift // samples a pass
	counts := map[string]int{}
	var order []string
	seen := map[string]bool{}
	var texts []string
	for start := 0; start < len(pcm); start += chunk {
		if ctx.Err() != nil {
			return h, ctx.Err()
		}
		part := pcm[start:min(start+chunk, len(pcm))]
		if len(part) < s.winLen {
			break
		}
		var ids []int32
		func() {
			sc := mlx.NewScope()
			defer sc.Close()
			feats, ferr := s.features(part)
			if ferr != nil {
				err = ferr
				return
			}
			best := mlx.Astype(mlx.Argmax(s.logProbs(feats, lang, itn), -1, false), mlx.Int32)
			ids = best.Int32s()
		}()
		if err != nil {
			return h, err
		}
		if len(ids) < 4 {
			continue
		}
		piece := func(id int32) string {
			if id >= 0 && int(id) < len(s.pieces) {
				return s.pieces[id]
			}
			return ""
		}
		if h.Language == "" {
			h.Language = tagName(piece(ids[0]))
		}
		if e := strings.ToLower(tagName(piece(ids[1]))); e != "" && e != "emo_unknown" && e != "unk" {
			if counts[e] == 0 {
				order = append(order, e)
			}
			counts[e]++
		}
		if ev := tagName(piece(ids[2])); ev != "" && ev != "Speech" && !seen[ev] {
			seen[ev] = true
			h.Events = append(h.Events, ev)
		}
		var text []int32
		prev := int32(-1)
		for _, id := range ids[4:] {
			if id != prev && id != s.blank {
				text = append(text, id)
			}
			prev = id
		}
		h.Tokens += len(text)
		if t := decodePieces(s.pieces, text); t != "" {
			texts = append(texts, t)
		}
	}
	for _, e := range order {
		if counts[e] > counts[h.Emotion] {
			h.Emotion = e
		}
	}
	h.Text = strings.Join(texts, " ")
	h.Took = time.Since(t0)
	mlx.ClearCache()
	return h, nil
}
