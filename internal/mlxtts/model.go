// Package mlxtts runs Qwen3-TTS on MLX: text to speech as mlx-audio does it
// (mlx_audio/tts/models/qwen3_tts), on the same MLX library through mlx-c.
//
// Speech is made twelve and a half frames a second. Each frame is sixteen
// codes: the talker — a Qwen3 language model reading the text, the voice and
// the frames so far — writes the first; the code predictor, five layers more,
// writes the other fifteen from the talker's last hidden state. The speech
// tokenizer's decoder then turns the codes into 24 kHz audio: the codebooks
// looked up, eight transformer layers, and convolutions upsampling each frame
// to 1,920 samples.
//
// The CustomVoice models are served: a voice by name (vivian, serena, ryan,
// ...), an instruction for the tone if wanted. Apple silicon only, as MLX is.
package mlxtts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/tokenizer"
)

type quantization struct {
	Group int    `json:"group_size"`
	Bits  int    `json:"bits"`
	Mode  string `json:"mode"`
}

type stackConfig struct {
	Hidden    int     `json:"hidden_size"`
	Layers    int     `json:"num_hidden_layers"`
	Heads     int     `json:"num_attention_heads"`
	KVHeads   int     `json:"num_key_value_heads"`
	HeadDim   int     `json:"head_dim"`
	FFN       int     `json:"intermediate_size"`
	Eps       float32 `json:"rms_norm_eps"`
	RopeTheta float32 `json:"rope_theta"`
	Vocab     int     `json:"vocab_size"`
}

type talkerConfig struct {
	stackConfig
	CodePredictor struct {
		stackConfig
		Groups int `json:"num_code_groups"`
	} `json:"code_predictor_config"`
	Groups      int              `json:"num_code_groups"`
	TextHidden  int              `json:"text_hidden_size"`
	TextVocab   int              `json:"text_vocab_size"`
	CodecBOS    int32            `json:"codec_bos_id"`
	CodecEOS    int32            `json:"codec_eos_token_id"`
	CodecPad    int32            `json:"codec_pad_id"`
	Think       int32            `json:"codec_think_id"`
	NoThink     int32            `json:"codec_nothink_id"`
	ThinkBOS    int32            `json:"codec_think_bos_id"`
	ThinkEOS    int32            `json:"codec_think_eos_id"`
	Languages   map[string]int32 `json:"codec_language_id"`
	Speakers    map[string]int32 `json:"spk_id"`
	Dialects    map[string]any   `json:"spk_is_dialect"` // false, or the dialect's language
	MaxPosition int              `json:"max_position_embeddings"`
	RopeScaling map[string]any   `json:"rope_scaling"`
}

type config struct {
	ModelType    string        `json:"model_type"`
	TTSType      string        `json:"tts_model_type"`
	Size         string        `json:"tts_model_size"`
	Talker       talkerConfig  `json:"talker_config"`
	TTSBOS       int32         `json:"tts_bos_token_id"`
	TTSEOS       int32         `json:"tts_eos_token_id"`
	TTSPad       int32         `json:"tts_pad_token_id"`
	Quantization *quantization `json:"quantization"`
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
	if c.ModelType != "qwen3_tts" {
		return nil, fmt.Errorf("%s is a %q model, not Qwen3-TTS", dir, c.ModelType)
	}
	if q := c.Quantization; q != nil && q.Mode == "" {
		q.Mode = "affine"
	}
	return &c, nil
}

// Serves says whether a config.json's model_type is one this package serves.
func Serves(modelType string) bool { return modelType == "qwen3_tts" }

// IsModelDir says whether path is a Qwen3-TTS model as MLX releases one: a
// directory with its config.json, safetensors and speech_tokenizer.
func IsModelDir(path string) bool {
	if st, err := os.Stat(path); err != nil || !st.IsDir() {
		return false
	}
	if _, err := readConfig(path); err != nil {
		return false
	}
	m, _ := filepath.Glob(filepath.Join(path, "*.safetensors"))
	s, _ := filepath.Glob(filepath.Join(path, "speech_tokenizer", "*.safetensors"))
	return len(m) > 0 && len(s) > 0
}

// linear is a weight [out, in], plain or quantized as MLX's QuantizedLinear
// stores it, and a bias or 0.
type linear struct {
	W, Scales, Biases, Bias mlx.Array
	Group, Bits             int
	Mode                    string
}

// apply is MLX's: a quantized layer adds its bias after the product, a plain
// one in the same kernel (addmm), as nn.QuantizedLinear and nn.Linear do.
func (l linear) apply(x mlx.Array) mlx.Array {
	if l.Scales != 0 {
		y := mlx.QuantizedMatmul(x, l.W, l.Scales, l.Biases, l.Group, l.Bits, l.Mode)
		if l.Bias != 0 {
			y = mlx.Add(y, l.Bias)
		}
		return y
	}
	if l.Bias != 0 {
		return mlx.Addmm(l.Bias, x, mlx.T(l.W))
	}
	return mlx.Matmul(x, mlx.T(l.W))
}

// rows looks rows of an embedding table up: ids [B, L] → [B, L, dim].
func rows(table, ids mlx.Array) mlx.Array { return mlx.Take(table, ids, 0) }

type weights struct {
	t     map[string]mlx.Array
	q     *quantization
	taken map[string]bool
}

func loadWeights(dir string, q *quantization) (*weights, error) {
	files, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if len(files) == 0 {
		return nil, fmt.Errorf("%s has no safetensors", dir)
	}
	w := &weights{t: map[string]mlx.Array{}, q: q, taken: map[string]bool{}}
	for _, f := range files {
		t, err := mlx.LoadSafetensors(f)
		if err != nil {
			w.free()
			return nil, err
		}
		for k, v := range t {
			w.t[k] = v
		}
	}
	return w, nil
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

// free releases the arrays nothing took.
func (w *weights) free() {
	for k, a := range w.t {
		if !w.taken[k] {
			a.Free()
		}
	}
	w.t = nil
}

// Model is a loaded Qwen3-TTS CustomVoice: the talker, its code predictor,
// the speech tokenizer's decoder and the text tokenizer.
type Model struct {
	cfg    *config
	talker *talker
	codec  *codec
	tok    *tokenizer.Tokenizer
	all    []mlx.Array
}

// Load reads a Qwen3-TTS directory. mlx.Open must have been called.
func Load(dir string) (m *Model, err error) {
	defer mlx.Recover(&err)
	if err := mlx.AudioOps(); err != nil {
		return nil, err
	}
	cfg, err := readConfig(dir)
	if err != nil {
		return nil, err
	}
	if cfg.TTSType != "custom_voice" {
		return nil, fmt.Errorf("%s is Qwen3-TTS %s; kinfer serves the CustomVoice models (voices by name)", dir, cfg.TTSType)
	}
	tok, err := tokenizer.LoadDir(dir)
	if err != nil {
		return nil, err
	}
	m = &Model{cfg: cfg, tok: tok}
	fail := func(err error) (*Model, error) { m.Free(); return nil, err }

	w, err := loadWeights(dir, cfg.Quantization)
	if err != nil {
		return nil, err
	}
	m.talker, err = loadTalker(w, cfg, dir)
	m.keep(w)
	if err != nil {
		return fail(err)
	}
	cw, err := loadWeights(filepath.Join(dir, "speech_tokenizer"), nil)
	if err != nil {
		return fail(err)
	}
	m.codec, err = loadCodec(cw, filepath.Join(dir, "speech_tokenizer"))
	m.keep(cw)
	if err != nil {
		return fail(err)
	}
	m.all = append(m.all, m.talker.made...)
	m.all = append(m.all, m.codec.made...)
	return m, nil
}

// keep holds what w's loaders took and frees the rest (the speech
// tokenizer's encoder, which a CustomVoice model never uses).
func (m *Model) keep(w *weights) {
	for k, a := range w.t {
		if w.taken[k] {
			m.all = append(m.all, a)
		}
	}
	w.free()
}

// Voices are the voices the model speaks in, sorted.
func (m *Model) Voices() []string {
	var v []string
	for name := range m.cfg.Talker.Speakers {
		v = append(v, name)
	}
	sort.Strings(v)
	return v
}

// Languages are the languages it can be told to speak, sorted; "auto" lets
// it tell from the text.
func (m *Model) Languages() []string {
	var v []string
	for name := range m.cfg.Talker.Languages {
		if !strings.Contains(name, "dialect") {
			v = append(v, name)
		}
	}
	sort.Strings(v)
	return v
}

// SampleRate is the audio's, 24 kHz.
func (m *Model) SampleRate() int { return sampleRate }

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
	if m.talker != nil {
		m.talker.textEmbed.close()
	}
}
