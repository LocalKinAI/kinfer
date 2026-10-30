package hub

import (
	"bytes"
	"context"
	"encoding/binary"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// gguf writes a header with string and uint16 keys and one-dimensional
// tensors at the given offsets, padded with filler so a ranged reader has more
// than a header to (not) read.
func gguf(kvs map[string]any, tensors map[string]uint64, filler int) []byte {
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(len(tensors)))
	w(uint64(len(kvs)))
	for k, v := range kvs {
		str(k)
		switch v := v.(type) {
		case string:
			w(uint32(8))
			str(v)
		case uint16:
			w(uint32(2))
			w(v)
		}
	}
	for name, off := range tensors {
		str(name)
		w(uint32(1))
		w(uint64(1))
		w(uint32(0))
		w(off)
	}
	b.Write(make([]byte, filler))
	return b.Bytes()
}

// A V4.1-shaped repo: a set whose last shard holds the Engram table, and ISTA's
// shared Engram file, published on its own.
func TestQuantLazyReadsHeadersOnly(t *testing.T) {
	files := map[string][]byte{
		"M-IQ2-00001-of-00002.gguf": gguf(map[string]any{"general.architecture": "deepseek41", "split.count": uint16(2)}, nil, 0),
		"M-IQ2-00002-of-00002.gguf": gguf(map[string]any{"split.count": uint16(2), "split.no": uint16(1)},
			map[string]uint64{"blk.0.ffn_up_exps.weight": 0, "per_layer_token_embd.weight": 1 << 20, "blk.1.engram_kv.weight": 1<<20 + 100<<30}, 4<<20),
		"Engram-IQ4_NL.gguf": gguf(map[string]any{"split.count": uint16(6), "split.no": uint16(5)},
			map[string]uint64{"per_layer_token_embd.weight": 0, "blk.1.engram_kv.weight": 103 << 30}, 0),
	}
	var served atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := r.URL.Path[strings.LastIndex(r.URL.Path, "/")+1:]
		data, ok := files[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		cw := &countingResponse{ResponseWriter: w, n: &served}
		http.ServeContent(cw, r, name, time.Time{}, bytes.NewReader(data))
	}))
	defer srv.Close()
	t.Setenv("KINFER_HF_ENDPOINT", srv.URL)

	set := Quant{Repo: "x/y", Files: []string{"M-IQ2-00001-of-00002.gguf", "M-IQ2-00002-of-00002.gguf"}, sizes: map[string]int64{}}
	for _, f := range set.Files {
		set.sizes[f] = int64(len(files[f]))
	}
	lazy, standalone, err := set.Lazy(context.Background(), "deepseek41")
	if err != nil || !standalone || lazy != 100<<30 {
		t.Errorf("set: lazy %d, standalone %v, err %v; want %d, true", lazy, standalone, err, int64(100<<30))
	}
	if n := served.Load(); n >= 4<<20 {
		t.Errorf("read %d bytes: the filler behind the header was fetched", n)
	}

	table := Quant{Repo: "x/y", Files: []string{"Engram-IQ4_NL.gguf"}, sizes: map[string]int64{"Engram-IQ4_NL.gguf": int64(len(files["Engram-IQ4_NL.gguf"]))}}
	if _, standalone, err := table.Lazy(context.Background(), "deepseek41"); err != nil || standalone {
		t.Errorf("shared table: standalone %v, err %v; want false", standalone, err)
	}

	// An architecture with nothing lazy costs nothing but the headers.
	if lazy, _, err := set.Lazy(context.Background(), "llama"); err != nil || lazy != 0 {
		t.Errorf("llama: lazy %d, %v", lazy, err)
	}
}

type countingResponse struct {
	http.ResponseWriter
	n *atomic.Int64
}

func (c *countingResponse) Write(p []byte) (int, error) {
	c.n.Add(int64(len(p)))
	return c.ResponseWriter.Write(p)
}
