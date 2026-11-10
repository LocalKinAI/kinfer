// Package mlxasr runs Qwen3-ASR on MLX: a speech recognition model that is a
// Qwen3 language model with an audio encoder in front, the encoder's output
// taking the place of the prompt's <|audio_pad|> tokens. It follows
// mlx-audio's implementation (mlx_audio/stt/models/qwen3_asr), on the same
// MLX library through mlx-c, so that kinfer's MLX path computes what the
// Python service does: Whisper's log-mel features, a convolutional front end
// and 24 transformer layers attending within windows of eight seconds, then
// the decoder reading the prompt with the sound in it and writing the
// transcript greedily.
//
// The llama.cpp path (libmtmd, a GGUF and its mmproj) stays the one for Linux
// and Windows; this one is Apple silicon only, as MLX is.
package mlxasr

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/tokenizer"
)

type audioConfig struct {
	DModel       int `json:"d_model"`
	Heads        int `json:"encoder_attention_heads"`
	FFN          int `json:"encoder_ffn_dim"`
	Layers       int `json:"encoder_layers"`
	Mels         int `json:"num_mel_bins"`
	MaxPositions int `json:"max_source_positions"`
	NWindow      int `json:"n_window"`
	NWindowInfer int `json:"n_window_infer"`
	OutputDim    int `json:"output_dim"`
	Downsample   int `json:"downsample_hidden_size"`
}

type textConfig struct {
	Hidden       int     `json:"hidden_size"`
	Layers       int     `json:"num_hidden_layers"`
	Heads        int     `json:"num_attention_heads"`
	KVHeads      int     `json:"num_key_value_heads"`
	HeadDim      int     `json:"head_dim"`
	FFN          int     `json:"intermediate_size"`
	Eps          float32 `json:"rms_norm_eps"`
	RopeTheta    float32 `json:"rope_theta"`
	Vocab        int     `json:"vocab_size"`
	Tied         bool    `json:"tie_word_embeddings"`
	MaxPositions int     `json:"max_position_embeddings"`
}

type quantization struct {
	Group int    `json:"group_size"`
	Bits  int    `json:"bits"`
	Mode  string `json:"mode"`
}

type config struct {
	ModelType string `json:"model_type"`
	Thinker   struct {
		Audio      audioConfig `json:"audio_config"`
		Text       textConfig  `json:"text_config"`
		AudioToken int32       `json:"audio_token_id"`
		AudioStart int32       `json:"audio_start_token_id"`
		AudioEnd   int32       `json:"audio_end_token_id"`
	} `json:"thinker_config"`
	Quantization *quantization `json:"quantization"`
	Languages    []string      `json:"support_languages"`
}

func readConfig(dir string) (*config, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("%s/config.json: %w", dir, err)
	}
	if c.ModelType != "qwen3_asr" {
		return nil, fmt.Errorf("%s is a %q model, not Qwen3-ASR", dir, c.ModelType)
	}
	if q := c.Quantization; q != nil && q.Mode == "" {
		q.Mode = "affine"
	}
	return &c, nil
}

// Serves says whether a config.json's model_type is one this package serves
// on MLX: Qwen3-ASR's or SenseVoice's.
func Serves(modelType string) bool { return modelType == "qwen3_asr" || modelType == "sensevoice" }

// IsModelDir says whether path is a Qwen3-ASR or SenseVoice model as MLX
// releases one: a directory with its config.json and safetensors.
func IsModelDir(path string) bool {
	if st, err := os.Stat(path); err != nil || !st.IsDir() {
		return false
	}
	_, qerr := readConfig(path)
	_, serr := readSVConfig(path)
	if qerr != nil && serr != nil {
		return false
	}
	m, _ := filepath.Glob(filepath.Join(path, "*.safetensors"))
	return len(m) > 0
}

// linear is a weight [out, in], plain or quantized as MLX's QuantizedLinear
// stores it, and a bias or 0.
type linear struct {
	W, Scales, Biases, Bias mlx.Array
	Group, Bits             int
	Mode                    string
}

func (l linear) apply(x mlx.Array) mlx.Array {
	var y mlx.Array
	if l.Scales != 0 {
		y = mlx.QuantizedMatmul(x, l.W, l.Scales, l.Biases, l.Group, l.Bits, l.Mode)
	} else {
		y = mlx.Matmul(x, mlx.T(l.W))
	}
	if l.Bias != 0 {
		y = mlx.Add(y, l.Bias)
	}
	return y
}

// rows looks rows of l up, as an embedding: ids [B, L] → [B, L, in].
func (l linear) rows(ids mlx.Array) mlx.Array {
	if l.Scales == 0 {
		return mlx.Take(l.W, ids, 0)
	}
	return mlx.Dequantize(mlx.Take(l.W, ids, 0), mlx.Take(l.Scales, ids, 0), mlx.Take(l.Biases, ids, 0), l.Group, l.Bits, l.Mode)
}

type weights struct {
	t     map[string]mlx.Array
	q     *quantization
	taken map[string]bool
}

func (w *weights) get(name string) (mlx.Array, error) {
	a, ok := w.t[name]
	if !ok {
		return 0, fmt.Errorf("the weights have no %s", name)
	}
	w.taken[name] = true
	return a, nil
}

func (w *weights) lin(prefix string) (linear, error) {
	var l linear
	var err error
	if l.W, err = w.get(prefix + ".weight"); err != nil {
		return l, err
	}
	if s, ok := w.t[prefix+".scales"]; ok {
		if w.q == nil {
			return l, fmt.Errorf("%s is quantized but config.json says nothing of how", prefix)
		}
		l.Scales, l.Biases = s, w.t[prefix+".biases"]
		w.taken[prefix+".scales"], w.taken[prefix+".biases"] = true, true
		l.Group, l.Bits, l.Mode = w.q.Group, w.q.Bits, w.q.Mode
	}
	if b, ok := w.t[prefix+".bias"]; ok {
		l.Bias = b
		w.taken[prefix+".bias"] = true
	}
	return l, nil
}

// Model is a loaded Qwen3-ASR.
type Model struct {
	cfg       *config
	enc       *encoder
	dec       *decoder
	tok       *tokenizer.Tokenizer
	eos       map[int32]bool
	languages map[string]string // lower-case name → the model's spelling
	all       []mlx.Array
}

// Load reads a Qwen3-ASR directory. mlx.Open must have been called.
func Load(dir string) (m *Model, err error) {
	defer mlx.Recover(&err)
	if err := mlx.AudioOps(); err != nil {
		return nil, err
	}
	cfg, err := readConfig(dir)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadDir(dir)
	if err != nil {
		return nil, err
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	w := &weights{t: map[string]mlx.Array{}, q: cfg.Quantization, taken: map[string]bool{}}
	for _, f := range files {
		t, err := mlx.LoadSafetensors(f)
		if err != nil {
			return nil, err
		}
		for k, v := range t {
			w.t[k] = v
		}
	}
	m = &Model{cfg: cfg, tok: tok, eos: map[int32]bool{}, languages: map[string]string{}}
	for _, a := range w.t {
		m.all = append(m.all, a)
	}
	fail := func(err error) (*Model, error) { m.Free(); return nil, err }
	if m.enc, err = loadEncoder(w, cfg.Thinker.Audio); err != nil {
		return fail(err)
	}
	if m.dec, err = loadDecoder(w, cfg.Thinker.Text); err != nil {
		return fail(err)
	}
	m.all = append(m.all, m.enc.pos)
	for _, s := range []string{"<|im_end|>", "<|endoftext|>"} {
		if id, ok := tok.ID(s); ok {
			m.eos[id] = true
		}
	}
	if len(m.eos) == 0 {
		m.eos[151645], m.eos[151643] = true, true
	}
	for _, l := range cfg.Languages {
		m.languages[strings.ToLower(l)] = l
	}
	return m, nil
}

// Bytes is what the weights take.
func (m *Model) Bytes() int64 {
	var n int64
	for _, a := range m.all {
		n += int64(a.Nbytes())
	}
	return n
}

// Free releases the weights.
func (m *Model) Free() {
	for _, a := range m.all {
		a.Free()
	}
	m.all = nil
}
