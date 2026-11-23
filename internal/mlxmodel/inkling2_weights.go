package mlxmodel

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"github.com/LocalKinAI/kinfer/internal/mlx"
	"github.com/LocalKinAI/kinfer/internal/safetensors"
)

// inkWeights is where an Inkling 2's tensors are read from. The model comes
// in two layouts holding the same bytes:
//
//   - Ollama's copy, a safetensors file a tensor (model.llm.….safetensors),
//     which MLX maps and reads when a tensor is first used;
//   - the publisher's own, as Hugging Face has it and kinfer pull fetches it:
//     every tensor in one file, model-00001-of-00001.safetensors. MLX does not
//     open that file — the experts' block scales are labelled F8_E4M3, a
//     number format it has no name for — so its tensors are read here, each
//     when it is asked for, the experts' bytes given the labels Ollama's copy
//     gives them: four of a weight's bytes a uint32, a scale a uint8.
type inkWeights struct {
	mapped map[string]mlx.Array // Ollama's copy: every tensor, not yet read
	files  []*safetensors.File  // the publisher's
	where  map[string]int       // tensor → which of files
	buf    []byte
}

// InklingDraftFile is the publisher's file of the model's draft head, which
// kinfer does not use: no tensor is looked for in it, and a pull leaves it.
const InklingDraftFile = "mtp.safetensors"

// openInkWeights finds the tensors of the model in dir.
func openInkWeights(dir string) (*inkWeights, error) {
	w := &inkWeights{}
	if each, _ := filepath.Glob(filepath.Join(dir, "model.llm.*.safetensors")); len(each) > 0 {
		visual, _ := filepath.Glob(filepath.Join(dir, "model.visual.*.safetensors"))
		w.mapped = map[string]mlx.Array{}
		for _, f := range append(each, visual...) {
			t, err := mlx.LoadSafetensors(f)
			if err != nil {
				w.close()
				return nil, err
			}
			for k, v := range t {
				w.mapped[k] = v
			}
		}
		return w, nil
	}
	all, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	sort.Strings(all)
	w.where = map[string]int{}
	for _, path := range all {
		if filepath.Base(path) == InklingDraftFile {
			continue
		}
		f, err := safetensors.Open(path)
		if err != nil {
			w.close()
			return nil, err
		}
		w.files = append(w.files, f)
		for name := range f.Tensors {
			w.where[name] = len(w.files) - 1
		}
	}
	if len(w.where) == 0 {
		return nil, fmt.Errorf("no weights of an Inkling 2 in %s", dir)
	}
	return w, nil
}

func (w *inkWeights) has(name string) bool {
	if w.mapped != nil {
		_, ok := w.mapped[name]
		return ok
	}
	_, ok := w.where[name]
	return ok
}

// take hands a tensor over, the caller's to free; it panics with an
// *mlx.Error, as the loaders do, when the model has none of that name.
func (w *inkWeights) take(name string) mlx.Array {
	if w.mapped != nil {
		a, ok := w.mapped[name]
		if !ok {
			panic(&mlx.Error{Msg: "weight " + name + " missing"})
		}
		delete(w.mapped, name)
		return a
	}
	i, ok := w.where[name]
	if !ok {
		panic(&mlx.Error{Msg: "weight " + name + " missing"})
	}
	delete(w.where, name)
	f := w.files[i]
	info := f.Tensors[name]
	dt, shape, err := inkLabel(name, info)
	if err != nil {
		panic(&mlx.Error{Msg: err.Error()})
	}
	n := int(info.Offsets[1] - info.Offsets[0])
	if cap(w.buf) < n {
		w.buf = make([]byte, n)
	}
	if err := f.ReadInto(name, w.buf[:n]); err != nil {
		panic(&mlx.Error{Msg: err.Error()})
	}
	return mlx.FromBytes(w.buf[:n], dt, shape...)
}

// inkLabel is the dtype and shape a tensor of the publisher's file is read
// as.
func inkLabel(name string, info safetensors.Info) (mlx.Dtype, []int, error) {
	shape := append([]int(nil), info.Shape...)
	size, n := 0, 1
	for _, d := range shape {
		n *= d
	}
	var dt mlx.Dtype
	bank := strings.HasSuffix(name, ".experts.w13_weight") || strings.HasSuffix(name, ".experts.w2_weight")
	switch info.Dtype {
	case "BF16":
		dt, size = mlx.Bfloat16, 2
	case "F16":
		dt, size = mlx.Float16, 2
	case "F32":
		dt, size = mlx.Float32, 4
	case "I32":
		dt, size = mlx.Int32, 4
	case "U32":
		dt, size = mlx.Uint32, 4
	case "U8", "F8_E4M3":
		dt, size = mlx.Uint8, 1
		if bank {
			// The experts' packed weights: NVFP4's two a byte, MXFP8's one,
			// four bytes to the word the kernels read.
			last := len(shape) - 1
			if last < 0 || shape[last]%4 != 0 {
				return 0, nil, fmt.Errorf("weight %s of shape %v is not whole words", name, info.Shape)
			}
			shape[last] /= 4
			dt, size, n = mlx.Uint32, 4, n/4
		}
	default:
		return 0, nil, fmt.Errorf("weight %s is %s, which is not read", name, info.Dtype)
	}
	if int64(n)*int64(size) != info.Offsets[1]-info.Offsets[0] {
		return 0, nil, fmt.Errorf("weight %s: %d bytes are not %v of %s", name, info.Offsets[1]-info.Offsets[0], info.Shape, info.Dtype)
	}
	return dt, shape, nil
}

// close frees what was not taken and closes the files.
func (w *inkWeights) close() {
	for _, a := range w.mapped {
		a.Free()
	}
	for _, f := range w.files {
		f.Close()
	}
	w.mapped, w.files, w.where, w.buf = nil, nil, nil, nil
}
