// Package safetensors reads and writes the safetensors format in plain Go: an
// 8-byte little-endian header length, a JSON header naming each tensor's
// dtype, shape and byte range, then the bytes. It is for kinlab edit, which
// compares and rewrites weights without a GPU; models are loaded by MLX.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
)

// Info is one tensor's entry in a header.
type Info struct {
	Dtype   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

// File is an open safetensors file.
type File struct {
	f       *os.File
	start   int64 // where the bytes begin
	Tensors map[string]Info
	Meta    map[string]string
}

// Open reads the header of the file at path.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var n uint64
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if n > 100<<20 {
		f.Close()
		return nil, fmt.Errorf("%s: a %d-byte header is not a safetensors file's", path, n)
	}
	head := make([]byte, n)
	if _, err := io.ReadFull(f, head); err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(head, &raw); err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: header: %w", path, err)
	}
	sf := &File{f: f, start: 8 + int64(n), Tensors: map[string]Info{}}
	for k, v := range raw {
		if k == "__metadata__" {
			if err := json.Unmarshal(v, &sf.Meta); err != nil {
				f.Close()
				return nil, fmt.Errorf("%s: metadata: %w", path, err)
			}
			continue
		}
		var info Info
		if err := json.Unmarshal(v, &info); err != nil {
			f.Close()
			return nil, fmt.Errorf("%s: %s: %w", path, k, err)
		}
		sf.Tensors[k] = info
	}
	return sf, nil
}

// Names are the file's tensors in name order.
func (f *File) Names() []string {
	out := make([]string, 0, len(f.Tensors))
	for k := range f.Tensors {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Read returns a tensor's bytes.
func (f *File) Read(name string) ([]byte, error) {
	info, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("no tensor %s", name)
	}
	b := make([]byte, info.Offsets[1]-info.Offsets[0])
	if _, err := f.f.ReadAt(b, f.start+info.Offsets[0]); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return b, nil
}

// Close closes the file.
func (f *File) Close() error { return f.f.Close() }

// Tensor is one tensor to write.
type Tensor struct {
	Name  string
	Dtype string
	Shape []int
	Data  []byte
}

// Write writes tensors to path, in name order, with meta as the header's
// __metadata__. The header is padded with spaces to a multiple of 8 bytes, as
// the format allows, so every tensor starts aligned.
func Write(path string, tensors []Tensor, meta map[string]string) error {
	sort.Slice(tensors, func(i, j int) bool { return tensors[i].Name < tensors[j].Name })
	head := map[string]any{}
	if len(meta) > 0 {
		head["__metadata__"] = meta
	}
	var off int64
	for _, t := range tensors {
		head[t.Name] = Info{Dtype: t.Dtype, Shape: t.Shape, Offsets: [2]int64{off, off + int64(len(t.Data))}}
		off += int64(len(t.Data))
	}
	h, err := json.Marshal(head)
	if err != nil {
		return err
	}
	for len(h)%8 != 0 {
		h = append(h, ' ')
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	err = binary.Write(f, binary.LittleEndian, uint64(len(h)))
	if err == nil {
		_, err = f.Write(h)
	}
	for _, t := range tensors {
		if err == nil {
			_, err = f.Write(t.Data)
		}
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}
