// Package mlximage makes pictures from words on Apple's MLX: Z-Image-Turbo
// (Tongyi-MAI, Apache-2.0), a port of mflux's — the text encoder (Qwen3-4B,
// the hidden state of its second-to-last layer), the 6B single-stream
// diffusion transformer, nine flow-matching steps, and Flux's VAE decoder —
// reading mflux's own saved weights (mflux-community/z-image-turbo-mflux-q8),
// so that what mflux computes is the reference, op for op.
package mlximage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/tokenizer"
)

// The parts of a model directory, as mflux saves one: each a folder of
// safetensors with an index whose metadata says how they are quantized.
var parts = []string{"transformer", "text_encoder", "vae"}

// IsModelDir says whether path is a Z-Image model as mflux saves one: its
// transformer, text encoder, VAE and tokenizer, the transformer's tensors
// Z-Image's (all_x_embedder, cap_embedder).
func IsModelDir(path string) bool {
	for _, p := range parts {
		if m, _ := filepath.Glob(filepath.Join(path, p, "*.safetensors")); len(m) == 0 {
			return false
		}
	}
	if _, err := os.Stat(filepath.Join(path, "tokenizer", "tokenizer.json")); err != nil {
		return false
	}
	idx, err := readIndex(filepath.Join(path, "transformer"))
	if err != nil {
		return false
	}
	x, c := false, false
	for k := range idx.WeightMap {
		x = x || strings.HasPrefix(k, "all_x_embedder.")
		c = c || strings.HasPrefix(k, "cap_embedder.")
	}
	return x && c
}

type index struct {
	Metadata  map[string]string `json:"metadata"`
	WeightMap map[string]string `json:"weight_map"`
}

func readIndex(dir string) (*index, error) {
	b, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		return nil, err
	}
	var i index
	if err := json.Unmarshal(b, &i); err != nil {
		return nil, fmt.Errorf("%s: %w", dir, err)
	}
	return &i, nil
}

// linear is a weight [out, in], plain or quantized as MLX's QuantizedLinear
// stores it, and a bias or 0.
type linear struct {
	W, Scales, Biases, Bias mlx.Array
	Group, Bits             int
}

// apply is MLX's: a quantized layer adds its bias after the product, a plain
// one in the same kernel (addmm), as nn.QuantizedLinear and nn.Linear do.
func (l linear) apply(x mlx.Array) mlx.Array {
	if l.Scales != 0 {
		y := mlx.QuantizedMatmul(x, l.W, l.Scales, l.Biases, l.Group, l.Bits, "affine")
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

// embedding is a table of rows, quantized as nn.QuantizedEmbedding stores it.
type embedding struct{ linear }

// rows are the rows of ids, dequantized as nn.QuantizedEmbedding does.
func (e embedding) rows(ids mlx.Array) mlx.Array {
	if e.Scales == 0 {
		return mlx.Take(e.W, ids, 0)
	}
	return mlx.Dequantize(mlx.Take(e.W, ids, 0), mlx.Take(e.Scales, ids, 0), mlx.Take(e.Biases, ids, 0), e.Group, e.Bits, "affine")
}

// weights are one part's tensors and how they are quantized.
type weights struct {
	part  string
	t     map[string]mlx.Array
	bits  int
	taken map[string]bool
}

func loadWeights(dir, part string) (*weights, error) {
	d := filepath.Join(dir, part)
	files, _ := filepath.Glob(filepath.Join(d, "*.safetensors"))
	if len(files) == 0 {
		return nil, fmt.Errorf("%s has no safetensors", d)
	}
	w := &weights{part: part, t: map[string]mlx.Array{}, taken: map[string]bool{}}
	if idx, err := readIndex(d); err == nil {
		if q := idx.Metadata["quantization_level"]; q != "" && q != "None" {
			if w.bits, err = strconv.Atoi(q); err != nil {
				return nil, fmt.Errorf("%s: quantization_level %q", d, q)
			}
		}
	}
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

func (w *weights) get(name string) mlx.Array {
	a, ok := w.t[name]
	if !ok {
		panic(fmt.Errorf("the %s has no %s", w.part, name))
	}
	w.taken[name] = true
	return a
}

func (w *weights) lin(prefix string) linear {
	l := linear{W: w.get(prefix + ".weight")}
	if s, ok := w.t[prefix+".scales"]; ok {
		if w.bits == 0 {
			panic(fmt.Errorf("the %s's %s is quantized and its index does not say how", w.part, prefix))
		}
		l.Scales, l.Biases = s, w.get(prefix+".biases")
		w.taken[prefix+".scales"] = true
		l.Bits = w.bits
		in := l.W.Shape()[1] * 32 / l.Bits
		l.Group = in / s.Shape()[1]
	}
	if b, ok := w.t[prefix+".bias"]; ok {
		l.Bias = b
		w.taken[prefix+".bias"] = true
	}
	return l
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

// Model is a loaded Z-Image: its text encoder, transformer, VAE decoder and
// tokenizer.
type Model struct {
	tok   *tokenizer.Tokenizer
	te    *textEncoder
	dit   *dit
	vae   *decoder
	all   []mlx.Array
	bytes int64
}

// Load reads a Z-Image directory as mflux saves one. mlx.Open must have been
// called.
func Load(dir string) (m *Model, err error) {
	defer mlx.Recover(&err)
	if err := mlx.AudioOps(); err != nil { // conv2d and layer norm
		return nil, err
	}
	if err := mlx.ImageOps(); err != nil {
		return nil, err
	}
	if !IsModelDir(dir) {
		return nil, fmt.Errorf("%s is not a Z-Image model as mflux saves one (transformer/, text_encoder/, vae/, tokenizer/)", dir)
	}
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer", "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	m = &Model{tok: tok}
	ws := map[string]*weights{}
	defer func() {
		for _, w := range ws {
			w.free()
		}
		if err != nil {
			m.Free()
		}
	}()
	for _, p := range parts {
		w, err := loadWeights(dir, p)
		if err != nil {
			return nil, err
		}
		ws[p] = w
	}
	m.te = loadTextEncoder(ws["text_encoder"])
	m.dit = loadDiT(ws["transformer"])
	m.vae = loadDecoder(ws["vae"])
	for _, w := range ws {
		for k := range w.taken {
			a := w.t[k]
			m.all = append(m.all, a)
			m.bytes += int64(a.Nbytes())
		}
	}
	return m, nil
}

// Bytes is what the weights take.
func (m *Model) Bytes() int64 { return m.bytes }

// Free releases the weights.
func (m *Model) Free() {
	for _, a := range m.all {
		a.Free()
	}
	m.all = nil
	if m.dit != nil {
		m.dit.free()
	}
}
