package llama

import (
	"fmt"
	"runtime"
	"unsafe"
)

// ─── drafting with a model's MTP head ───────────────────────────────────────
//
// A model trained with multi-token prediction carries a head of its own — its
// "nextn" layers — that guesses the token after next from the hidden state the
// model left at the token before. A context of type MTP over the same model
// runs that head, with a KV cache of its own; the main context hands it, for
// every token it reads, the hidden state it computed there (the "nextn
// embedding"). Guesses are drafts: the main model then reads them all in one
// pass, and keeps the ones it would have written itself.
//
// llama.cpp's own server does this (--spec-type draft-mtp, common/speculative.cpp).
// Part of what it calls is staged in src/llama-ext.h, exported with C++ names,
// so those are bound by their mangled names. All of it is optional: a build
// without it serves without drafts.

// ExtBatch is a llama_batch_ext: a batch whose tokens can carry a hidden
// state, as an MTP context's must.
type ExtBatch uintptr

// ContextTypeMTP is LLAMA_CONTEXT_TYPE_MTP, for ContextParams.CtxType.
const ContextTypeMTP int32 = 1

// processDecode is LLAMA_PROCESS_TYPE_DECODE.
const processDecode int32 = 1

// llamaEmbd mirrors struct llama_embd (24 bytes).
type llamaEmbd struct {
	data  uintptr
	nRows uint64
	nEmbd uint64
}

var (
	modelNLayerNextn        func(m Model) int32
	batchExtInit            func(c Context) ExtBatch
	batchExtFree            func(b ExtBatch)
	batchExtClear           func(b ExtBatch)
	batchExtAddToken        func(b ExtBatch, seq int32, tok Token) int32
	batchExtSetEmbdState    func(b ExtBatch, idx int32, e llamaEmbd) bool
	batchExtSetOutputLogits func(b ExtBatch, idx int32, v bool) bool
	batchExtSetPos          func(b ExtBatch, idx int32, pos *int32) bool
	processExt              func(c Context, typ int32, b ExtBatch) int32
	setEmbeddingsNextn      func(c Context, on, masked bool)
	getEmbeddingsNextn      func(c Context) *float32
	getEmbeddingsNextnIth   func(c Context, i int32) *float32
)

// bindMTP resolves the drafting entry points in libllama, all or nothing.
func bindMTP(lib uintptr) {
	for _, b := range []struct {
		fn   any
		name string
	}{
		{&modelNLayerNextn, "llama_model_n_layer_nextn"},
		{&batchExtInit, "llama_batch_ext_init"},
		{&batchExtFree, "llama_batch_ext_free"},
		{&batchExtClear, "llama_batch_ext_clear"},
		{&batchExtAddToken, "llama_batch_ext_add_token"},
		{&batchExtSetEmbdState, "llama_batch_ext_set_embd_state"},
		{&batchExtSetOutputLogits, "llama_batch_ext_set_output_logits"},
		{&batchExtSetPos, "llama_batch_ext_set_pos"},
		{&processExt, "llama_process"},
		// src/llama-ext.h, C++ linkage: (llama_context *, bool, bool),
		// (llama_context *) → float * and (llama_context *, int) → float *.
		{&setEmbeddingsNextn, "_Z26llama_set_embeddings_nextnP13llama_contextbb"},
		{&getEmbeddingsNextn, "_Z26llama_get_embeddings_nextnP13llama_context"},
		{&getEmbeddingsNextnIth, "_Z30llama_get_embeddings_nextn_ithP13llama_contexti"},
	} {
		if err := register(b.fn, lib, b.name); err != nil {
			processExt = nil
			return
		}
	}
}

// CanDraft says whether this build drafts with an MTP head.
func CanDraft() bool { return processExt != nil }

// NextnLayers is how many MTP layers the model's header declares (0 for none).
// It is the header's word, not the weights': a file may declare them and carry
// none, which llama.cpp loads without complaint (see store.Shape.MTP), and a
// model loaded without ModelParams.LoadMtp has them on disk only.
func NextnLayers(m Model) int {
	if modelNLayerNextn == nil {
		return 0
	}
	return int(modelNLayerNextn(m))
}

// SetNextn makes c compute the hidden state its MTP head reads: for every
// token of a batch (masked false), as the main context must, or only for the
// tokens whose logits are asked for (masked true), as an MTP context does.
func SetNextn(c Context, on, masked bool) { setEmbeddingsNextn(c, on, masked) }

// NextnRow is the hidden state at index i of the last batch c read, dims
// values; it aliases llama.cpp's buffer, valid until c reads another batch.
func NextnRow(c Context, i int32, dims int) []float32 {
	p := getEmbeddingsNextnIth(c, i)
	if p == nil {
		return nil
	}
	return unsafe.Slice(p, dims)
}

// NextnRows is the hidden state at every one of the n tokens of the last batch
// c read, in batch order, dims values each — what a main context computes with
// SetNextn(c, true, false). It aliases llama.cpp's buffer like NextnRow, and
// waits for the batch to finish as reading logits does.
func NextnRows(c Context, n, dims int) []float32 {
	p := getEmbeddingsNextn(c)
	if p == nil || n <= 0 {
		return nil
	}
	return unsafe.Slice(p, n*dims)
}

// NewExtBatch makes a batch for c, freed with Free.
func NewExtBatch(c Context) ExtBatch { return batchExtInit(c) }

// Free releases the batch.
func (b ExtBatch) Free() {
	if b != 0 {
		batchExtFree(b)
	}
}

// Clear empties the batch.
func (b ExtBatch) Clear() { batchExtClear(b) }

// Add puts a token of sequence seq at pos into the batch, with the hidden state
// it is read with — which llama.cpp copies — and returns its index.
func (b ExtBatch) Add(tok Token, pos, seq int32, logits bool, state []float32) (int32, error) {
	idx := batchExtAddToken(b, seq, tok)
	if idx < 0 {
		return idx, fmt.Errorf("draft batch: cannot add a token (code %d)", idx)
	}
	p := pos
	if !batchExtSetPos(b, idx, &p) {
		return idx, fmt.Errorf("draft batch: cannot set position %d", pos)
	}
	if logits && !batchExtSetOutputLogits(b, idx, true) {
		return idx, fmt.Errorf("draft batch: cannot ask for logits")
	}
	if len(state) > 0 {
		e := llamaEmbd{data: uintptr(unsafe.Pointer(&state[0])), nRows: 1, nEmbd: uint64(len(state))}
		ok := batchExtSetEmbdState(b, idx, e)
		runtime.KeepAlive(state)
		if !ok {
			return idx, fmt.Errorf("draft batch: the context takes no state of %d values", len(state))
		}
	}
	return idx, nil
}

// Process reads b into c.
func Process(c Context, b ExtBatch) error {
	var rc int32
	onComputeThread(func() { rc = processExt(c, processDecode, b) })
	if rc != 0 {
		return &DecodeError{Code: rc}
	}
	return nil
}
