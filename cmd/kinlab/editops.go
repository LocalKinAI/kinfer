package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/mlxmodel"
	"github.com/LocalKinAI/kinfer/internal/safetensors"
)

// listFlag is a flag that may be given more than once.
type listFlag []string

func (l *listFlag) String() string     { return strings.Join(*l, " ") }
func (l *listFlag) Set(v string) error { *l = append(*l, v); return nil }

// editOps changes a base's tensors in memory, one operation after another,
// and gives back the ones that end up different.
type editOps struct {
	base *weights
	cfg  *mlxmodel.Qwen35Config
	work map[string][]byte // tensors an operation touched, as they are now
}

func newEditOps(base *weights, cfg *mlxmodel.Qwen35Config) *editOps {
	return &editOps{base: base, cfg: cfg, work: map[string][]byte{}}
}

// tensor is the current bytes of a tensor, read from the base the first time.
func (e *editOps) tensor(name string) ([]byte, safetensors.Info, error) {
	info, ok := e.base.info(name)
	if !ok {
		return nil, info, fmt.Errorf("the base has no tensor %s", name)
	}
	if b, ok := e.work[name]; ok {
		return b, info, nil
	}
	b, err := e.base.read(name)
	if err != nil {
		return nil, info, err
	}
	e.work[name] = b
	return b, info, nil
}

// changed is every tensor whose bytes now differ from the base's.
func (e *editOps) changed() ([]safetensors.Tensor, error) {
	var out []safetensors.Tensor
	for _, n := range e.base.names() {
		b, ok := e.work[n]
		if !ok {
			continue
		}
		orig, err := e.base.read(n)
		if err != nil {
			return nil, err
		}
		if !bytes.Equal(b, orig) {
			info, _ := e.base.info(n)
			out = append(out, safetensors.Tensor{Name: n, Dtype: info.Dtype, Shape: info.Shape, Data: b})
		}
	}
	return out, nil
}

// layerTensor is the name of layer l's tensor that ends in suffix: the
// language model's, not the MTP head's or the vision tower's, which number
// their layers too.
func (e *editOps) layerTensor(l int, suffix string) (string, error) {
	if l < 0 || l >= e.cfg.NumHiddenLayers {
		return "", fmt.Errorf("layer %d: the model has layers 0 to %d", l, e.cfg.NumHiddenLayers-1)
	}
	end := fmt.Sprintf("layers.%d.%s", l, suffix)
	var found []string
	for _, n := range e.base.names() {
		if (n == end || strings.HasSuffix(n, "."+end)) && !strings.HasPrefix(n, "mtp.") && !strings.Contains(n, "visual") {
			found = append(found, n)
		}
	}
	if len(found) != 1 {
		return "", fmt.Errorf("layer %d has %d tensors named …%s: %v", l, len(found), end, found)
	}
	return found[0], nil
}

// scale multiplies every tensor whose name matches a pattern: "REGEX=FACTOR".
func (e *editOps) scale(spec string) error {
	pat, f, ok := strings.Cut(spec, "=")
	factor, err := strconv.ParseFloat(f, 64)
	if !ok || err != nil {
		return fmt.Errorf("-scale %q: want PATTERN=FACTOR, as in 'layers\\.15\\.mlp\\.down_proj=0.5'", spec)
	}
	re, err := regexp.Compile(pat)
	if err != nil {
		return fmt.Errorf("-scale %q: %w", spec, err)
	}
	n := 0
	for _, name := range e.base.names() {
		if !re.MatchString(name) || strings.HasPrefix(name, "mtp.") || strings.Contains(name, "visual") {
			continue
		}
		b, info, err := e.tensor(name)
		if err != nil {
			return err
		}
		v, err := floats(b, info.Dtype)
		if err != nil {
			return fmt.Errorf("-scale %s: %w", name, err)
		}
		for i := range v {
			v[i] = float32(float64(v[i]) * factor)
		}
		e.work[name] = fromFloats(v, info.Dtype)
		n++
	}
	if n == 0 {
		return fmt.Errorf("-scale %q matches no tensor of the language model", spec)
	}
	return nil
}

// zeroNeurons zeroes neurons of a layer's MLP, "LAYER:I,J,K-M": their
// columns of down_proj, so they add nothing to the residual stream.
func (e *editOps) zeroNeurons(spec string) error {
	l, idx, err := layerAndList(spec)
	if err != nil {
		return fmt.Errorf("-zero-neuron %q: %w", spec, err)
	}
	name, err := e.layerTensor(l, "mlp.down_proj.weight")
	if err != nil {
		return err
	}
	return e.zeroColumns(name, idx, 1, e.cfg.IntermediateSize)
}

// zeroHead zeroes heads of a layer's attention, "LAYER:H,H": their columns of
// the output projection — o_proj's in a full-attention layer, of head_dim
// each, and out_proj's in a DeltaNet layer, of linear_value_head_dim.
func (e *editOps) zeroHead(spec string) error {
	l, heads, err := layerAndList(spec)
	if err != nil {
		return fmt.Errorf("-zero-head %q: %w", spec, err)
	}
	suffix, width, count := "linear_attn.out_proj.weight", e.cfg.LinearValueHeadDim, e.cfg.LinearNumValueHeads
	if (l+1)%e.cfg.FullAttentionInterval == 0 {
		suffix, width, count = "self_attn.o_proj.weight", e.cfg.HeadDim, e.cfg.NumAttentionHeads
	}
	name, err := e.layerTensor(l, suffix)
	if err != nil {
		return err
	}
	return e.zeroColumns(name, heads, width, count)
}

// zeroColumns zeroes groups of width columns of a [rows, count*width] weight.
func (e *editOps) zeroColumns(name string, groups []int, width, count int) error {
	b, info, err := e.tensor(name)
	if err != nil {
		return err
	}
	size, err := elemSize(info.Dtype)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	if len(info.Shape) != 2 || info.Shape[1] != width*count {
		return fmt.Errorf("%s is %v, not [rows, %d×%d]", name, info.Shape, count, width)
	}
	cols := info.Shape[1]
	for _, g := range groups {
		if g < 0 || g >= count {
			return fmt.Errorf("%s has %d of them; there is no %d", name, count, g)
		}
		for r := 0; r < info.Shape[0]; r++ {
			at := (r*cols + g*width) * size
			clear(b[at : at+width*size])
		}
	}
	return nil
}

// mix moves the base toward another copy of its architecture: base +
// alpha·(other − base), tensor by tensor where the two differ. 1 is the
// other's tensors exactly, 0.5 halfway, and past 1 further along the same
// way.
func (e *editOps) mix(other *weights, alpha float64, take func(string) bool) error {
	for _, n := range other.names() {
		if !take(n) {
			continue
		}
		bi, ok := e.base.info(n)
		oi := other.where[n].Tensors[n]
		if !ok || bi.Dtype != oi.Dtype || !slices.Equal(bi.Shape, oi.Shape) {
			return fmt.Errorf("-mix: %s is not the same tensor in both", n)
		}
		b, info, err := e.tensor(n)
		if err != nil {
			return err
		}
		o, err := other.read(n)
		if err != nil {
			return err
		}
		if bytes.Equal(b, o) {
			continue
		}
		if alpha == 1 {
			e.work[n] = o
			continue
		}
		x, err := floats(b, info.Dtype)
		if err != nil {
			return fmt.Errorf("-mix %s: %w", n, err)
		}
		y, _ := floats(o, info.Dtype)
		for i := range x {
			x[i] = float32(float64(x[i]) + alpha*(float64(y[i])-float64(x[i])))
		}
		e.work[n] = fromFloats(x, info.Dtype)
	}
	return nil
}

// layerAndList reads "LAYER:A,B,C-D".
func layerAndList(spec string) (int, []int, error) {
	ls, list, ok := strings.Cut(spec, ":")
	l, err := strconv.Atoi(ls)
	if !ok || err != nil || list == "" {
		return 0, nil, fmt.Errorf("want LAYER:INDEX[,INDEX|,FROM-TO], as in 15:0,7,100-163")
	}
	var out []int
	for _, part := range strings.Split(list, ",") {
		a, b, isRange := strings.Cut(part, "-")
		lo, err1 := strconv.Atoi(a)
		hi, err2 := lo, error(nil)
		if isRange {
			hi, err2 = strconv.Atoi(b)
		}
		if err1 != nil || err2 != nil || hi < lo {
			return 0, nil, fmt.Errorf("%q is no index or range", part)
		}
		for i := lo; i <= hi; i++ {
			out = append(out, i)
		}
	}
	return l, out, nil
}

func elemSize(dtype string) (int, error) {
	switch dtype {
	case "BF16", "F16":
		return 2, nil
	case "F32":
		return 4, nil
	}
	return 0, fmt.Errorf("a %s tensor: kinlab edit changes BF16, F16 and F32 weights, not quantized ones; edit the full-precision model", dtype)
}

// floats decodes a BF16 or F32 tensor.
func floats(b []byte, dtype string) ([]float32, error) {
	switch dtype {
	case "BF16":
		v := make([]float32, len(b)/2)
		for i := range v {
			v[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
		}
		return v, nil
	case "F32":
		v := make([]float32, len(b)/4)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return v, nil
	}
	return nil, fmt.Errorf("arithmetic on %s weights: kinlab edit computes on BF16 and F32", dtype)
}

// fromFloats encodes v as dtype, rounding to the nearest BF16, ties to even.
func fromFloats(v []float32, dtype string) []byte {
	if dtype == "F32" {
		b := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
		}
		return b
	}
	b := make([]byte, 2*len(v))
	for i, x := range v {
		u := math.Float32bits(x)
		var h uint16
		if u&0x7fffffff > 0x7f800000 {
			h = uint16(u>>16) | 0x40 // a NaN stays one
		} else {
			h = uint16((u + 0x7fff + (u>>16)&1) >> 16)
		}
		binary.LittleEndian.PutUint16(b[2*i:], h)
	}
	return b
}
