package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlxengine"
	"github.com/LocalKinAI/kinfer/internal/mlxmodel"
	"github.com/LocalKinAI/kinfer/internal/safetensors"
	"github.com/LocalKinAI/kinfer/internal/store"
)

// editRecord is edit.json, beside an edited model's edit.safetensors: what
// it was made from, and how.
type editRecord struct {
	Base    string    `json:"base"`     // the base model's name
	BaseDir string    `json:"base_dir"` // and where it is
	Made    time.Time `json:"made"`
	From    string    `json:"from,omitempty"`    // the model whose differing tensors were taken
	Overlay string    `json:"overlay,omitempty"` // the file the tensors came from
	Note    string    `json:"note,omitempty"`
	Tensors []string  `json:"tensors"`
}

// cmdEdit makes an edited model: a directory of the store whose files are
// links to a base model's, and whose edit.safetensors holds the tensors that
// differ. The base stays as it was, and the edit costs only what it changes.
func cmdEdit(args []string) error {
	fs := flag.NewFlagSet("edit", flag.ExitOnError)
	from := fs.String("from", "", "take the tensors that differ from this model (another copy of the base's architecture)")
	overlay := fs.String("overlay", "", "take the tensors in this .safetensors file, written by anything (a Python script, say)")
	note := fs.String("note", "", "a line saying what the edit is, kept in its edit.json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 || (*from == "") == (*overlay == "") {
		return fmt.Errorf("usage: kinfer edit (-from <model> | -overlay <file.safetensors>) [-note text] <base> <new-name>")
	}
	st, err := store.Open()
	if err != nil {
		return err
	}
	baseDir, err := st.Resolve(fs.Arg(0))
	if err != nil {
		return err
	}
	if !mlxengine.IsModelDir(baseDir) {
		return fmt.Errorf("%s is no directory of safetensors; kinfer edits the models it serves on MLX", fs.Arg(0))
	}
	if _, err := os.Stat(filepath.Join(baseDir, mlxmodel.EditFile)); err == nil {
		return fmt.Errorf("%s is itself an edit; edit the model it was made from", fs.Arg(0))
	}
	name := fs.Arg(1)
	if strings.ContainsRune(name, os.PathSeparator) || name == "" || name[0] == '.' {
		return fmt.Errorf("%q is no name for a model", name)
	}
	dir := filepath.Join(st.Root(), name)
	if _, err := os.Lstat(dir); err == nil {
		return fmt.Errorf("%s is there already", dir)
	}

	base, err := openWeights(baseDir)
	if err != nil {
		return err
	}
	defer base.close()

	rec := editRecord{Base: filepath.Base(baseDir), BaseDir: baseDir, Made: time.Now().UTC(), Note: *note}
	var changed []safetensors.Tensor
	var size int64
	switch {
	case *from != "":
		otherDir, err := st.Resolve(*from)
		if err != nil {
			return err
		}
		other, err := openWeights(otherDir)
		if err != nil {
			return err
		}
		defer other.close()
		rec.From = otherDir
		for _, n := range other.names() {
			if _, ok := base.info(n); !ok {
				return fmt.Errorf("%s has a tensor %s that %s does not; an edit can only change the base's", *from, n, fs.Arg(0))
			}
		}
		for _, n := range base.names() {
			oi, ok := other.info(n)
			if !ok {
				continue // a tensor the other copy leaves out, as the base has it
			}
			bi, _ := base.info(n)
			if oi.Dtype != bi.Dtype || !slices.Equal(oi.Shape, bi.Shape) {
				return fmt.Errorf("%s is %s %v in %s and %s %v in %s", n, oi.Dtype, oi.Shape, *from, bi.Dtype, bi.Shape, fs.Arg(0))
			}
			a, err := base.read(n)
			if err != nil {
				return err
			}
			b, err := other.read(n)
			if err != nil {
				return err
			}
			if !bytes.Equal(a, b) {
				changed = append(changed, safetensors.Tensor{Name: n, Dtype: oi.Dtype, Shape: oi.Shape, Data: b})
				size += int64(len(b))
			}
		}
		if len(changed) == 0 {
			return fmt.Errorf("%s and %s have the same weights", *from, fs.Arg(0))
		}
	default:
		ov, err := safetensors.Open(*overlay)
		if err != nil {
			return err
		}
		defer ov.Close()
		rec.Overlay, _ = filepath.Abs(*overlay)
		for _, n := range ov.Names() {
			oi := ov.Tensors[n]
			bi, ok := base.info(n)
			switch {
			case !ok:
				return fmt.Errorf("%s: %s has no tensor %s", *overlay, fs.Arg(0), n)
			case oi.Dtype != bi.Dtype || !slices.Equal(oi.Shape, bi.Shape):
				return fmt.Errorf("%s: %s is %s %v, the base's is %s %v", *overlay, n, oi.Dtype, oi.Shape, bi.Dtype, bi.Shape)
			}
			b, err := ov.Read(n)
			if err != nil {
				return err
			}
			changed = append(changed, safetensors.Tensor{Name: n, Dtype: oi.Dtype, Shape: oi.Shape, Data: b})
			size += int64(len(b))
		}
	}
	for _, t := range changed {
		rec.Tensors = append(rec.Tensors, t.Name)
	}

	// Made beside the store and renamed into it, so the store never lists an
	// edit missing its edit.
	tmp := filepath.Join(filepath.Dir(st.Root()), "edit-"+name+".part")
	os.RemoveAll(tmp)
	if err := os.Mkdir(tmp, 0o755); err != nil {
		return err
	}
	ok := false
	defer func() {
		if !ok {
			os.RemoveAll(tmp)
		}
	}()
	entries, err := os.ReadDir(baseDir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		if err := os.Symlink(filepath.Join(baseDir, e.Name()), filepath.Join(tmp, e.Name())); err != nil {
			return err
		}
	}
	if err := safetensors.Write(filepath.Join(tmp, mlxmodel.EditFile), changed, map[string]string{"format": "pt"}); err != nil {
		return err
	}
	b, _ := json.MarshalIndent(rec, "", "  ")
	if err := os.WriteFile(filepath.Join(tmp, "edit.json"), append(b, '\n'), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, dir); err != nil {
		return err
	}
	ok = true
	fmt.Printf("✅ %s: %d of %d tensors of %s changed (%s), the rest linked\n",
		name, len(changed), len(base.names()), rec.Base, store.HumanSize(size))
	return nil
}

// weights are a model directory's tensors across its safetensors files.
type weights struct {
	files []*safetensors.File
	where map[string]*safetensors.File
}

func openWeights(dir string) (*weights, error) {
	paths, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	w := &weights{where: map[string]*safetensors.File{}}
	for _, p := range paths {
		if filepath.Base(p) == mlxmodel.EditFile {
			continue
		}
		f, err := safetensors.Open(p)
		if err != nil {
			w.close()
			return nil, err
		}
		w.files = append(w.files, f)
		for n := range f.Tensors {
			w.where[n] = f
		}
	}
	if len(w.files) == 0 {
		return nil, fmt.Errorf("no .safetensors in %s", dir)
	}
	return w, nil
}

func (w *weights) names() []string {
	out := make([]string, 0, len(w.where))
	for n := range w.where {
		out = append(out, n)
	}
	slices.Sort(out)
	return out
}

func (w *weights) info(name string) (safetensors.Info, bool) {
	f, ok := w.where[name]
	if !ok {
		return safetensors.Info{}, false
	}
	return f.Tensors[name], true
}

func (w *weights) read(name string) ([]byte, error) { return w.where[name].Read(name) }

func (w *weights) close() {
	for _, f := range w.files {
		f.Close()
	}
}
