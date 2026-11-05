package safetensors

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestWriteThenRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.safetensors")
	in := []Tensor{
		{Name: "b.weight", Dtype: "BF16", Shape: []int{2, 3}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
		{Name: "a.bias", Dtype: "F32", Shape: []int{1}, Data: []byte{0, 0, 128, 63}},
	}
	if err := Write(path, in, map[string]string{"format": "mlx"}); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if got := f.Names(); len(got) != 2 || got[0] != "a.bias" || got[1] != "b.weight" {
		t.Errorf("names %v", got)
	}
	if f.Meta["format"] != "mlx" {
		t.Errorf("metadata %v", f.Meta)
	}
	if f.start%8 != 0 {
		t.Errorf("the bytes start at %d, not on an 8-byte boundary", f.start)
	}
	for _, want := range in {
		got, err := f.Read(want.Name)
		if err != nil {
			t.Fatal(err)
		}
		info := f.Tensors[want.Name]
		if !bytes.Equal(got, want.Data) || info.Dtype != want.Dtype || len(info.Shape) != len(want.Shape) {
			t.Errorf("%s came back as %v %v %v", want.Name, info.Dtype, info.Shape, got)
		}
	}
	if _, err := f.Read("missing"); err == nil {
		t.Error("read a tensor that is not there")
	}
}
