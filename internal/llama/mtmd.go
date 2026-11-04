package llama

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ─── pictures ────────────────────────────────────────────────────────────────
//
// A vision model reads a picture through its projector — the "mmproj" file
// beside it, a layer of its own in Ollama's store — which encodes the picture
// into embeddings the model reads in place of tokens. llama.cpp's libmtmd does
// that: it finds the places for pictures in a prompt's text, cuts the prompt
// into text and picture chunks, and reads them into a context, picture
// positions and all (Qwen-VL lays a picture out in two dimensions).

// Mtmd is a projector loaded over a model.
type Mtmd uintptr

// MediaMarker is where a picture goes in a prompt's text: libmtmd's default
// marker, one for each picture, in order.
const MediaMarker = "<__media__>"

// mtmdContextParams mirrors struct mtmd_context_params (96 bytes at b11998).
type mtmdContextParams struct {
	UseGPU                   uint8
	Device                   uintptr
	PrintTimings             uint8
	NThreads                 int32
	ImageMarker              uintptr
	MediaMarker              uintptr
	FlashAttnType            int32
	Warmup                   uint8
	ImageMinTokens           int32
	ImageMaxTokens           int32
	CbEval                   uintptr
	CbEvalUserData           uintptr
	BatchMaxTokens           int32
	ProgressCallback         uintptr
	ProgressCallbackUserData uintptr
}

// mtmdInputText mirrors struct mtmd_input_text (24 bytes).
type mtmdInputText struct {
	text         uintptr
	textLen      uint64
	addSpecial   uint8
	parseSpecial uint8
}

// mtmdHelperInitOpt mirrors struct mtmd_helper_init_opt (32 bytes): video
// options only, which kinfer leaves at their defaults.
type mtmdHelperInitOpt struct {
	fpsTarget           float32
	ffmpegBinDir        uintptr
	timestampIntervalMs int64
	noSoundtrack        uint8
}

// mtmdBitmapWrapper mirrors struct mtmd_helper_bitmap_wrapper (16 bytes).
type mtmdBitmapWrapper struct {
	bitmap uintptr
	video  uintptr
}

var (
	mtmdContextParamsDefault func() mtmdContextParams
	mtmdInitFromFile         func(fname *byte, model Model, p mtmdContextParams) Mtmd
	mtmdFree                 func(m Mtmd)
	mtmdSupportVision        func(m Mtmd) bool
	mtmdInputChunksInit      func() uintptr
	mtmdInputChunksFree      func(chunks uintptr)
	mtmdTokenize             func(m Mtmd, out uintptr, text *mtmdInputText, bitmaps *uintptr, n uint64) int32
	mtmdHelperInitOptDefault func() mtmdHelperInitOpt
	mtmdHelperBitmapFromBuf  func(m Mtmd, buf *byte, n uint64, placeholder bool, opt mtmdHelperInitOpt) mtmdBitmapWrapper
	mtmdBitmapFree           func(b uintptr)
	mtmdHelperNTokens        func(chunks uintptr) uint64
	mtmdHelperNPos           func(chunks uintptr) int32
	mtmdHelperEvalChunks     func(m Mtmd, lctx Context, chunks uintptr, nPast, seq, nBatch int32, logitsLast bool, newNPast *int32) int32
)

// bindMtmd loads libmtmd beside libllama, best-effort as libggml-rpc is: a
// build without it serves text and refuses pictures.
func bindMtmd(dir string) {
	lib, err := purego.Dlopen(siblingLib(dir, "mtmd"), purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&mtmdContextParamsDefault, "mtmd_context_params_default"},
		{&mtmdInitFromFile, "mtmd_init_from_file"},
		{&mtmdFree, "mtmd_free"},
		{&mtmdSupportVision, "mtmd_support_vision"},
		{&mtmdInputChunksInit, "mtmd_input_chunks_init"},
		{&mtmdInputChunksFree, "mtmd_input_chunks_free"},
		{&mtmdTokenize, "mtmd_tokenize"},
		{&mtmdHelperInitOptDefault, "mtmd_helper_init_opt_default"},
		{&mtmdHelperBitmapFromBuf, "mtmd_helper_bitmap_init_from_buf"},
		{&mtmdBitmapFree, "mtmd_bitmap_free"},
		{&mtmdHelperNTokens, "mtmd_helper_get_n_tokens"},
		{&mtmdHelperNPos, "mtmd_helper_get_n_pos"},
		{&mtmdHelperEvalChunks, "mtmd_helper_eval_chunks"},
	} {
		if err := register(b.fn, lib, b.name); err != nil {
			mtmdInitFromFile = nil // all or nothing
			return
		}
	}
}

// SeesPictures says whether this build has libmtmd to read pictures with.
func SeesPictures() bool { return mtmdInitFromFile != nil }

// LoadProjector loads a vision model's projector over model, its compute
// buffers warmed up so that they are counted with the model's memory. Zero
// when it cannot be loaded, or encodes no pictures; llama.cpp says why on
// stderr.
func LoadProjector(path string, model Model, threads int) Mtmd {
	if !SeesPictures() {
		return 0
	}
	p := mtmdContextParamsDefault()
	p.UseGPU = 1
	p.PrintTimings = 0
	p.Warmup = 1
	if threads > 0 {
		p.NThreads = int32(threads)
	}
	name := append([]byte(path), 0)
	m := mtmdInitFromFile(&name[0], model, p)
	runtime.KeepAlive(name)
	if m != 0 && !mtmdSupportVision(m) {
		mtmdFree(m)
		return 0
	}
	return m
}

// FreeProjector releases a projector.
func FreeProjector(m Mtmd) {
	if m != 0 {
		mtmdFree(m)
	}
}

// MediaPrompt is a prompt with pictures, cut into text and picture chunks.
// Tokens is what it takes in the KV cache, and Positions how far it moves the
// sequence: fewer than Tokens where a model lays a picture out in two
// dimensions. Free releases it.
type MediaPrompt struct {
	chunks    uintptr
	Tokens    int
	Positions int32
}

// TokenizeMedia cuts text into chunks, a picture — any image llama.cpp reads:
// jpg, png, bmp, gif — at each MediaMarker in it, in order.
func TokenizeMedia(m Mtmd, text string, pictures [][]byte) (*MediaPrompt, error) {
	bitmaps := make([]uintptr, 0, len(pictures))
	defer func() {
		for _, b := range bitmaps {
			mtmdBitmapFree(b)
		}
	}()
	opt := mtmdHelperInitOptDefault()
	for i, pic := range pictures {
		if len(pic) == 0 {
			return nil, fmt.Errorf("picture %d is empty", i)
		}
		w := mtmdHelperBitmapFromBuf(m, &pic[0], uint64(len(pic)), false, opt)
		runtime.KeepAlive(pic)
		if w.bitmap == 0 {
			return nil, fmt.Errorf("picture %d is not an image llama.cpp reads (jpg, png, bmp, gif)", i)
		}
		bitmaps = append(bitmaps, w.bitmap)
	}

	chunks := mtmdInputChunksInit()
	ctext := append([]byte(text), 0)
	in := mtmdInputText{text: uintptr(unsafe.Pointer(&ctext[0])), textLen: uint64(len(text)), addSpecial: 1, parseSpecial: 1}
	var first *uintptr
	if len(bitmaps) > 0 {
		first = &bitmaps[0]
	}
	rc := mtmdTokenize(m, chunks, &in, first, uint64(len(bitmaps)))
	runtime.KeepAlive(ctext)
	runtime.KeepAlive(bitmaps)
	switch rc {
	case 0:
	case 1:
		mtmdInputChunksFree(chunks)
		return nil, fmt.Errorf("%d pictures for the places the prompt has for them", len(pictures))
	default:
		mtmdInputChunksFree(chunks)
		return nil, fmt.Errorf("a picture could not be prepared for the model (code %d)", rc)
	}
	return &MediaPrompt{chunks: chunks, Tokens: int(mtmdHelperNTokens(chunks)), Positions: mtmdHelperNPos(chunks)}, nil
}

// Free releases the chunks.
func (p *MediaPrompt) Free() {
	if p != nil && p.chunks != 0 {
		mtmdInputChunksFree(p.chunks)
		p.chunks = 0
	}
}

// EvalMedia reads p into lctx as sequence seq from its start, text in passes
// of up to nBatch tokens and each picture encoded and then read, and leaves
// the logits of its last token at index -1. It returns the position after it.
// Not safe to call while anything else decodes on lctx.
func EvalMedia(m Mtmd, lctx Context, p *MediaPrompt, seq int32, nBatch int) (int32, error) {
	var next int32
	if rc := mtmdHelperEvalChunks(m, lctx, p.chunks, 0, seq, int32(nBatch), true, &next); rc != 0 {
		return 0, &DecodeError{Code: rc}
	}
	return next, nil
}
