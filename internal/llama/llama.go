// Package llama binds llama.cpp directly, through purego.
//
// It replaces github.com/dianlight/gollama.cpp, which was wrong in ways that do
// not announce themselves. Its Go mirror of llama_context_params still declared
// a `seed` field that llama.cpp removed, and was missing `flash_attn_type`
// entirely — so every field after them landed in the wrong slot. A size
// assigned to NCtx arrived as n_batch, and nothing errored; the context simply
// stayed at llama.cpp's 512-token default no matter what was asked for. It also
// hardcoded the vocabulary size to 32 (Qwen has 151,936), called
// llama_vocab_get_text where it meant llama_token_to_piece, and accepted a
// library path it then discarded.
//
// The whole surface kinfer needs is about thirty entry points, so binding them
// here costs less than working around a mirror that drifts. Every struct below
// is transcribed from llama.h at the build embedded in internal/nativelib
// (b10901), and Open verifies the layout at runtime rather than trusting this
// comment. Both param structs changed shape between b6862 and b10901 — three
// new fields in one, two removed and two added in the other — and the size
// assertions in bind caught it before a single call was made. That is the whole
// point of having them.
//
// Still no CGO: purego resolves symbols at runtime, so cross-compilation from a
// Mac keeps working. Struct arguments and returns go through purego directly —
// no libffi, which is what gollama needed them for.
//
// # Layout rules
//
// Fields are transcribed in declaration order with C's padding. purego rejects
// Go's bool inside a struct, so C bools are uint8 (identical layout); the
// boolean helpers below convert. A mismatch here is silent corruption, which is
// why sizes are asserted at init and n_ctx is verified after every context is
// created.
package llama

import (
	"fmt"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Token is a vocabulary index (llama_token).
type Token = int32

// Model, Context, Vocab and Memory are opaque llama.cpp handles.
type (
	Model   uintptr
	Context uintptr
	Vocab   uintptr
	Memory  uintptr
	Sampler uintptr
)

// DefaultSeed asks llama.cpp for a random seed (LLAMA_DEFAULT_SEED).
const DefaultSeed uint32 = 0xFFFFFFFF

// SamplerChainParams mirrors struct llama_sampler_chain_params (1 byte).
type SamplerChainParams struct {
	NoPerf uint8
}

// ModelParams mirrors struct llama_model_params (80 bytes).
type ModelParams struct {
	Devices              uintptr
	TensorBuftOverrides  uintptr
	NGpuLayers           int32
	SplitMode            int32
	LoadMode             int32
	LazyMode             int32
	MainGpu              int32
	_                    int32 // padding before the next pointer
	TensorSplit          uintptr
	ProgressCallback     uintptr
	ProgressCallbackData uintptr
	KvOverrides          uintptr
	VocabOnly            uint8
	CheckTensors         uint8
	UseExtraBufts        uint8
	NoHost               uint8
	NoAlloc              uint8
	LoadMtp              uint8
}

// ContextParams mirrors struct llama_context_params (160 bytes).
//
// This is the struct gollama got wrong. The first field is n_ctx — there is no
// seed here; sampling owns the seed now.
type ContextParams struct {
	NCtx              uint32
	NBatch            uint32
	NUbatch           uint32
	NSeqMax           uint32
	NRsSeq            uint32
	NOutputsMax       uint32
	NOutputsMaxPerSeq uint32
	NThreads          int32
	NThreadsBatch     int32

	CtxType         int32
	RopeScalingType int32
	PoolingType     int32
	AttentionType   int32
	FlashAttnType   int32

	RopeFreqBase   float32
	RopeFreqScale  float32
	YarnExtFactor  float32
	YarnAttnFactor float32
	YarnBetaFast   float32
	YarnBetaSlow   float32
	YarnOrigCtx    uint32
	DefragThold    float32

	CbEval         uintptr
	CbEvalUserData uintptr

	TypeK int32
	TypeV int32

	AbortCallback     uintptr
	AbortCallbackData uintptr

	Embeddings uint8
	OffloadKQV uint8
	NoPerf     uint8
	OpOffload  uint8
	SwaFull    uint8
	KVUnified  uint8
	_          [2]uint8 // padding before the next pointer

	Samplers  uintptr
	NSamplers uint64
	CtxOther  uintptr
}

// Batch mirrors struct llama_batch (56 bytes): one unit of work for a decode.
//
// The pointer fields are left nil by BatchGetOne, which is the single-sequence
// shortcut. Filling them in by hand is how several sequences share one decode —
// the thing that makes concurrency worth having.
type Batch struct {
	NTokens int32
	_       int32
	Token   uintptr
	Embd    uintptr
	Pos     uintptr
	NSeqID  uintptr
	SeqID   uintptr
	Logits  uintptr
}

var (
	bindOnce sync.Once
	bindErr  error

	backendInit        func()
	backendFree        func()
	modelDefaultParams func() ModelParams
	ctxDefaultParams   func() ContextParams
	modelLoadFromFile  func(path string, p ModelParams) Model
	modelFree          func(m Model)
	initFromModel      func(m Model, p ContextParams) Context
	freeContext        func(c Context)
	decode             func(c Context, b Batch) int32
	synchronize        func(c Context)
	batchGetOne        func(tokens *Token, n int32) Batch
	getLogitsIth       func(c Context, i int32) *float32
	getMemory          func(c Context) Memory
	memoryClear        func(mem Memory, data bool)
	memorySeqRm        func(mem Memory, seq, p0, p1 int32) bool
	memorySeqCp        func(mem Memory, src, dst, p0, p1 int32)
	tokenize           func(v Vocab, text string, textLen int32, out *Token, max int32, addSpecial, parseSpecial bool) int32
	modelGetVocab      func(m Model) Vocab
	vocabNTokens       func(v Vocab) int32
	vocabIsEOG         func(v Vocab, t Token) bool
	vocabEOS           func(v Vocab) Token
	tokenToPiece       func(v Vocab, t Token, buf *byte, length, lstrip int32, special bool) int32
	nCtx               func(c Context) uint32

	samplerChainDefault func() SamplerChainParams
	samplerChainInit    func(p SamplerChainParams) Sampler
	samplerChainAdd     func(chain, s Sampler)
	samplerInitTopK     func(k int32) Sampler
	samplerInitTopP     func(p float32, minKeep uint64) Sampler
	samplerInitTemp     func(t float32) Sampler
	samplerInitPenalty  func(lastN int32, repeat, freq, present float32) Sampler
	samplerInitDist     func(seed uint32) Sampler
	samplerInitGreedy   func() Sampler
	samplerSample       func(s Sampler, c Context, idx int32) Token
	samplerReset        func(s Sampler)
	samplerFree         func(s Sampler)

	modelChatTemplate    func(m Model, name *byte) *byte
	chatApplyTemplate    func(tmpl *byte, msgs *chatMessage, n uint64, addAss bool, buf *byte, length int32) int32
	chatBuiltinTemplates func(out **byte, n uint64) int32
)

// chatMessage mirrors struct llama_chat_message: two C strings.
type chatMessage struct {
	role    uintptr
	content uintptr
}

// Bind makes llama.cpp usable: it loads libllama from dir, resolves every entry
// point, and initialises the backend. Safe to call repeatedly; the work happens
// once. Pass the directory nativelib.Prepare unpacked.
//
// Initialising the backend here rather than exposing it separately is
// deliberate. Every other call in this package is a raw function pointer that
// is nil until binding completes, so a caller who initialised first would get a
// nil-pointer panic with nothing to point at. There is no order to get wrong if
// there is only one call.
func Bind(dir string) error {
	bindOnce.Do(func() {
		if bindErr = bind(dir); bindErr == nil {
			backendInit()
		}
	})
	return bindErr
}

func bind(dir string) error {
	// A struct whose Go size differs from C's would corrupt every call that
	// passes it. Catch that here rather than in a debugger.
	for _, c := range []struct {
		name string
		got  uintptr
		want uintptr
	}{
		{"llama_model_params", unsafe.Sizeof(ModelParams{}), 80},
		{"llama_context_params", unsafe.Sizeof(ContextParams{}), 160},
		{"llama_batch", unsafe.Sizeof(Batch{}), 56},
		{"llama_sampler_chain_params", unsafe.Sizeof(SamplerChainParams{}), 1},
	} {
		if c.got != c.want {
			return fmt.Errorf("%s is %d bytes in Go but %d in llama.h — "+
				"the struct definition in internal/llama is out of date", c.name, c.got, c.want)
		}
	}

	path := filepath.Join(dir, libName())

	// RTLD_GLOBAL so symbols resolve against the copy the backend already uses;
	// two independently loaded copies of libllama would each carry their own
	// state and hand out handles the other cannot interpret.
	lib, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}

	for _, b := range []struct {
		fn   any
		name string
	}{
		{&backendInit, "llama_backend_init"},
		{&backendFree, "llama_backend_free"},
		{&modelDefaultParams, "llama_model_default_params"},
		{&ctxDefaultParams, "llama_context_default_params"},
		{&modelLoadFromFile, "llama_model_load_from_file"},
		{&modelFree, "llama_model_free"},
		{&initFromModel, "llama_init_from_model"},
		{&freeContext, "llama_free"},
		{&decode, "llama_decode"},
		{&synchronize, "llama_synchronize"},
		{&batchGetOne, "llama_batch_get_one"},
		{&getLogitsIth, "llama_get_logits_ith"},
		{&getMemory, "llama_get_memory"},
		{&memoryClear, "llama_memory_clear"},
		{&memorySeqRm, "llama_memory_seq_rm"},
		{&memorySeqCp, "llama_memory_seq_cp"},
		{&tokenize, "llama_tokenize"},
		{&modelGetVocab, "llama_model_get_vocab"},
		{&vocabNTokens, "llama_vocab_n_tokens"},
		{&vocabIsEOG, "llama_vocab_is_eog"},
		{&vocabEOS, "llama_vocab_eos"},
		{&tokenToPiece, "llama_token_to_piece"},
		{&nCtx, "llama_n_ctx"},
		{&samplerChainDefault, "llama_sampler_chain_default_params"},
		{&samplerChainInit, "llama_sampler_chain_init"},
		{&samplerChainAdd, "llama_sampler_chain_add"},
		{&samplerInitTopK, "llama_sampler_init_top_k"},
		{&samplerInitTopP, "llama_sampler_init_top_p"},
		{&samplerInitTemp, "llama_sampler_init_temp"},
		{&samplerInitPenalty, "llama_sampler_init_penalties"},
		{&samplerInitDist, "llama_sampler_init_dist"},
		{&samplerInitGreedy, "llama_sampler_init_greedy"},
		{&samplerSample, "llama_sampler_sample"},
		{&samplerReset, "llama_sampler_reset"},
		{&samplerFree, "llama_sampler_free"},
		{&modelChatTemplate, "llama_model_chat_template"},
		{&chatApplyTemplate, "llama_chat_apply_template"},
		{&chatBuiltinTemplates, "llama_chat_builtin_templates"},
	} {
		// RegisterLibFunc panics on a missing symbol, which would abort the
		// program on an unexpected llama.cpp build. Convert it to an error so
		// callers can report something actionable.
		if err := register(b.fn, lib, b.name); err != nil {
			return err
		}
	}

	// The device registry lives in libggml. It is how kinfer learns the GPU's
	// working-set budget, and it is best-effort: a build without it still
	// serves, it just cannot warn about a configuration that will not fit.
	bindGGML(dir)
	return nil
}

func register(fptr any, lib uintptr, name string) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("symbol %s not found in libllama (incompatible build?): %v", name, r)
		}
	}()
	purego.RegisterLibFunc(fptr, lib, name)
	return nil
}

// ─── backend ─────────────────────────────────────────────────────────────────

// BackendFree tears down llama.cpp's backends. Only a program that is about to
// exit needs it, and only after every model and context is freed.
func BackendFree() {
	if backendFree != nil {
		backendFree()
	}
}

// ─── model and context ───────────────────────────────────────────────────────

// DefaultModelParams returns llama.cpp's own defaults, asked of the library
// rather than copied into Go where they would go stale.
func DefaultModelParams() ModelParams { return modelDefaultParams() }

// DefaultContextParams likewise.
func DefaultContextParams() ContextParams { return ctxDefaultParams() }

// LoadModel reads a GGUF file. A zero Model means failure; llama.cpp has
// already written the reason to stderr.
func LoadModel(path string, p ModelParams) Model { return modelLoadFromFile(path, p) }

// FreeModel releases a model.
func FreeModel(m Model) { modelFree(m) }

// NewContext creates an inference context over a loaded model.
func NewContext(m Model, p ContextParams) Context { return initFromModel(m, p) }

// FreeContext releases a context.
func FreeContext(c Context) { freeContext(c) }

// NCtx reports the context size llama.cpp actually allocated — the ground truth
// for verifying that a requested size took effect.
func NCtx(c Context) int { return int(nCtx(c)) }

// ─── vocabulary ──────────────────────────────────────────────────────────────

// GetVocab returns the model's vocabulary handle.
func GetVocab(m Model) Vocab { return modelGetVocab(m) }

// NVocab is the true vocabulary size — the number gollama hardcoded to 32,
// which truncated every candidate list to the first 32 of Qwen's 151,936 and
// made correct sampling impossible.
func NVocab(v Vocab) int32 { return vocabNTokens(v) }

// IsEOG reports whether a token ends generation (end-of-sequence, end-of-turn,
// or any other terminator this model defines).
func IsEOG(v Vocab, t Token) bool { return vocabIsEOG(v, t) }

// EOS returns the primary end-of-sequence token id.
func EOS(v Vocab) Token { return vocabEOS(v) }

// TokenToPiece converts one token to its decoded text — the real
// llama_token_to_piece, so the result needs no byte-level BPE post-processing.
// gollama's Token_to_piece called llama_vocab_get_text instead and returned
// "ĠMerkle" where the text is " Merkle".
//
// special controls whether special tokens render as text ("<|im_end|>") or as
// nothing. Generation wants false; inspecting a prompt wants true.
func TokenToPiece(v Vocab, t Token, special bool) string {
	// Ask with a small buffer first. The C function returns the needed length
	// as a negative number when it does not fit, so one retry always suffices.
	buf := make([]byte, 64)
	n := tokenToPiece(v, t, &buf[0], int32(len(buf)), 0, special)
	if n < 0 {
		buf = make([]byte, -n)
		n = tokenToPiece(v, t, &buf[0], int32(len(buf)), 0, special)
		if n < 0 {
			return ""
		}
	}
	if n == 0 {
		return ""
	}
	return string(buf[:n])
}

// Tokenize converts text to tokens.
//
// addSpecial prepends the model's BOS when it wants one; parseSpecial makes
// markers like "<|im_start|>" tokenize as themselves rather than as literal
// text, which is what a rendered chat template needs.
func Tokenize(v Vocab, text string, addSpecial, parseSpecial bool) ([]Token, error) {
	if text == "" {
		return nil, nil
	}
	// llama_tokenize reports the required length as a negative count when the
	// buffer is too small. Start from an estimate that is almost always enough
	// — a token averages well over one byte — and retry exactly once.
	out := make([]Token, len(text)+16)
	n := tokenize(v, text, int32(len(text)), &out[0], int32(len(out)), addSpecial, parseSpecial)
	if n < 0 {
		out = make([]Token, -n)
		n = tokenize(v, text, int32(len(text)), &out[0], int32(len(out)), addSpecial, parseSpecial)
		if n < 0 {
			return nil, fmt.Errorf("tokenize: needed %d slots twice", -n)
		}
	}
	return out[:n], nil
}

// ─── generation ──────────────────────────────────────────────────────────────

// BatchGetOne wraps tokens as a single-sequence batch.
//
// The returned Batch borrows the caller's slice — llama.cpp does not copy it —
// so tokens must stay alive and unmodified until Decode returns.
func BatchGetOne(tokens []Token) Batch {
	if len(tokens) == 0 {
		return Batch{}
	}
	return batchGetOne(&tokens[0], int32(len(tokens)))
}

// DecodeTokens decodes one sequence in a single pass.
//
// It exists so the token slice cannot be collected while llama.cpp is reading
// it: Batch holds a raw uintptr, which the garbage collector does not see as a
// reference. Building the batch and decoding in one place keeps that window
// closed.
func DecodeTokens(c Context, tokens []Token) error {
	if len(tokens) == 0 {
		return nil
	}
	err := Decode(c, batchGetOne(&tokens[0], int32(len(tokens))))
	runtime.KeepAlive(tokens)
	return err
}

// DecodeError is a failed forward pass, carrying llama.cpp's own code so a
// caller can tell a bad request from a broken context.
type DecodeError struct{ Code int32 }

func (e *DecodeError) Error() string {
	switch {
	case e.Code == 1:
		return "no free KV slot for this batch (the context is full)"
	case e.Code == 2:
		return "generation was aborted"
	case e.Code == -1:
		return "invalid batch"
	default:
		return fmt.Sprintf("llama.cpp backend failed (code %d)", e.Code)
	}
}

// Fatal reports whether the context is unusable from here on.
//
// llama.h draws the line at -1: below it the failure is fatal and the
// half-processed batch stays in the context's memory. In practice this is a
// Metal command buffer that ran out of GPU memory, after which every later
// decode returns the same error — "backend is in error state from a previous
// command buffer failure - recreate the backend to recover". Nothing short of a
// new context recovers, so a caller that keeps using this one serves errors
// forever.
func (e *DecodeError) Fatal() bool { return e.Code < -1 }

// Decode runs one forward pass.
func Decode(c Context, b Batch) error {
	if rc := decode(c, b); rc != 0 {
		return &DecodeError{Code: rc}
	}
	return nil
}

// Synchronize waits for a queued forward pass to actually finish.
//
// Decode does not wait. It builds the graph, hands it to the backend and
// returns, and on Metal the GPU is still working when it does; llama.cpp settles
// up at the first read of the results, inside llama_get_logits_ith. That is
// invisible to a caller who only wants the tokens — the logits are correct
// either way — and wrong for one holding a stopwatch, because the wait is
// charged to whatever happens to read the logits rather than to the decode that
// caused it.
func Synchronize(c Context) { synchronize(c) }

// Logits returns the logit row for position i (-1 means the last token). The
// slice aliases llama.cpp's own buffer and is only valid until the next Decode.
func Logits(c Context, i int32, nVocab int32) []float32 {
	p := getLogitsIth(c, i)
	if p == nil {
		return nil
	}
	return unsafe.Slice(p, nVocab)
}

// ClearMemory wipes the KV cache. Without it, a second conversation on the same
// context silently inherits the first one's state.
func ClearMemory(c Context) { memoryClear(getMemory(c), true) }

// ForgetSequence drops one sequence's KV cells, leaving every other sequence
// sharing the context untouched. This is what frees a slot for reuse: clearing
// the whole cache would take the other slots' conversations with it.
//
// p0 and p1 bound the positions removed; -1 for either means "no bound".
func ForgetSequence(c Context, seq int32, p0, p1 int32) {
	memorySeqRm(getMemory(c), seq, p0, p1)
}

// CopySequence makes dst share src's KV cells over [p0, p1).
//
// It does not copy anything: llama.cpp tags the existing cells with a second
// sequence id. That is what will make a shared prompt prefix — one agent's
// system prompt, prefilled once and reused on every turn it takes — nearly
// free.
func CopySequence(c Context, src, dst int32, p0, p1 int32) {
	memorySeqCp(getMemory(c), src, dst, p0, p1)
}

// ─── multi-sequence batches ──────────────────────────────────────────────────

// BatchBuilder assembles one llama_batch holding tokens from several sequences.
//
// This is the unit of work continuous batching runs on: one forward pass that
// advances every active conversation by a token, plus whatever prompt chunks
// are being prefilled. The weights are read once and serve the whole batch,
// which is why aggregate throughput climbs while per-stream speed barely moves.
//
// It owns Go memory that llama.cpp reads through raw pointers, so a builder must
// outlive the Decode call that reads it. Keep one per scheduler rather than
// allocating per step.
type BatchBuilder struct {
	tokens  []Token
	pos     []int32
	nSeqID  []int32
	seqIDs  []int32   // one sequence per token — kinfer never needs more
	seqPtrs []uintptr // seqPtrs[i] = &seqIDs[i], because C wants llama_seq_id**
	logits  []int8
	n       int
}

// NewBatchBuilder makes a builder holding up to n tokens per batch.
func NewBatchBuilder(n int) *BatchBuilder {
	b := &BatchBuilder{
		tokens:  make([]Token, n),
		pos:     make([]int32, n),
		nSeqID:  make([]int32, n),
		seqIDs:  make([]int32, n),
		seqPtrs: make([]uintptr, n),
		logits:  make([]int8, n),
	}
	for i := range b.seqIDs {
		b.seqPtrs[i] = uintptr(unsafe.Pointer(&b.seqIDs[i]))
		b.nSeqID[i] = 1
	}
	return b
}

// Cap is the largest batch this builder can hold.
func (b *BatchBuilder) Cap() int { return len(b.tokens) }

// Len is how many tokens are queued.
func (b *BatchBuilder) Len() int { return b.n }

// Reset empties the batch.
func (b *BatchBuilder) Reset() { b.n = 0 }

// Add appends one token of sequence seq at position pos, returning the index to
// read its logits at. Returns -1 if the batch is full.
//
// wantLogits marks the token whose logits will be read afterwards — the last
// token of each sequence in the batch, and nothing else. llama.cpp computes
// only the output rows that are asked for, so marking every token would do many
// times the work for results nobody reads.
func (b *BatchBuilder) Add(tok Token, pos int32, seq int32, wantLogits bool) int32 {
	if b.n >= len(b.tokens) {
		return -1
	}
	i := b.n
	b.tokens[i] = tok
	b.pos[i] = pos
	b.seqIDs[i] = seq
	b.logits[i] = 0
	if wantLogits {
		b.logits[i] = 1
	}
	b.n++
	return int32(i)
}

// Decode runs the forward pass for everything added since Reset.
func (b *BatchBuilder) Decode(c Context) error {
	if b.n == 0 {
		return nil
	}
	batch := Batch{
		NTokens: int32(b.n),
		Token:   uintptr(unsafe.Pointer(&b.tokens[0])),
		Pos:     uintptr(unsafe.Pointer(&b.pos[0])),
		NSeqID:  uintptr(unsafe.Pointer(&b.nSeqID[0])),
		SeqID:   uintptr(unsafe.Pointer(&b.seqPtrs[0])),
		Logits:  uintptr(unsafe.Pointer(&b.logits[0])),
	}
	err := Decode(c, batch)
	// Every field above crossed as a raw pointer, which the collector cannot
	// see. Hold the slices until llama.cpp has finished reading them.
	runtime.KeepAlive(b)
	return err
}

// ─── sampling ────────────────────────────────────────────────────────────────

// NewSamplerChain creates an empty chain. Add samplers in pipeline order and
// end with a selector (Dist or Greedy), which is what actually picks a token.
//
// The chain owns everything added to it: SamplerFree on the chain frees the
// children too, so an added sampler must not be freed separately.
func NewSamplerChain() Sampler {
	return samplerChainInit(samplerChainDefault())
}

// SamplerChainAdd appends s to chain, transferring ownership.
func SamplerChainAdd(chain, s Sampler) { samplerChainAdd(chain, s) }

// SamplerTopK keeps only the k most likely tokens.
//
// This is where the throughput went: llama.cpp does a partial selection over
// the candidates, while doing it in Go meant sorting all 151,936 of them once
// per token.
func SamplerTopK(k int32) Sampler { return samplerInitTopK(k) }

// SamplerTopP keeps the smallest set of tokens whose probabilities reach p.
// minKeep floors how many survive, so an extremely peaked distribution still
// leaves something to choose from.
func SamplerTopP(p float32, minKeep uint64) Sampler { return samplerInitTopP(p, minKeep) }

// SamplerTemp flattens (>1) or sharpens (<1) the distribution.
func SamplerTemp(t float32) Sampler { return samplerInitTemp(t) }

// SamplerPenalties pushes down tokens seen in the last lastN positions.
// repeat 1.0, freq 0.0 and present 0.0 each disable their term.
func SamplerPenalties(lastN int32, repeat, freq, present float32) Sampler {
	return samplerInitPenalty(lastN, repeat, freq, present)
}

// SamplerDist draws from the distribution the chain produced. Pass DefaultSeed
// for a random draw.
func SamplerDist(seed uint32) Sampler { return samplerInitDist(seed) }

// SamplerGreedy always takes the most likely token.
func SamplerGreedy() Sampler { return samplerInitGreedy() }

// SamplerSample reads the logits for position idx (-1 means the last token),
// runs the chain, and returns the chosen token.
//
// It reads llama.cpp's own logit buffer, so nothing is copied into Go.
//
// It also accepts the token internally — llama_sampler_sample calls
// llama_sampler_accept before returning. The usage example in llama.h still
// shows an explicit accept after sampling; following it would count every token
// twice in the penalty ring buffer, weakening the repetition penalty for the
// exact tokens it is meant to suppress.
func SamplerSample(s Sampler, c Context, idx int32) Token { return samplerSample(s, c, idx) }

// SamplerReset clears a chain's accumulated state, such as the penalty history.
func SamplerReset(s Sampler) { samplerReset(s) }

// SamplerFree releases a sampler and, for a chain, everything in it.
func SamplerFree(s Sampler) { samplerFree(s) }

// libName is the file to dlopen. It is the versioned soname, not the plain
// one: llama.cpp's own macOS builds reference @rpath/libggml.0.dylib, so the
// unpacked directory has to carry the names its DT_NEEDED entries actually ask
// for. The unversioned libllama.dylib in the release tarball is a symlink, and
// //go:embed cannot carry symlinks — every file here is a real copy.
// ─── chat templates ──────────────────────────────────────────────────────────

// ChatTemplate returns the chat template stored in the model's GGUF metadata,
// or "" when the file carries none.
//
// It is a Jinja string, but it is not rendered as one — see ApplyChatTemplate.
func ChatTemplate(m Model) string {
	p := modelChatTemplate(m, nil)
	if p == nil {
		return ""
	}
	return goString(p)
}

// ApplyChatTemplate renders a conversation the way the model was fine-tuned to
// see it. roles and contents must be the same length.
//
// llama.cpp does NOT run Jinja here. It matches the template string against a
// list of families it implements in C++ and applies that. The consequence is
// worth knowing in both directions: kinfer gets every family llama.cpp knows
// without shipping a template engine, and a model whose template matches none
// of them fails rather than rendering something subtly wrong.
//
// addAssistant appends the tokens that open an assistant turn, which is what
// makes the model answer rather than continue the conversation.
func ApplyChatTemplate(tmpl string, roles, contents []string, addAssistant bool) (string, error) {
	if len(roles) != len(contents) {
		return "", fmt.Errorf("chat template: %d roles but %d contents", len(roles), len(contents))
	}
	if len(roles) == 0 {
		return "", nil
	}

	// C wants null-terminated strings and reads them during the call, so the
	// backing arrays have to stay alive until it returns.
	cstrs := make([][]byte, 0, len(roles)*2)
	cstr := func(s string) uintptr {
		b := append([]byte(s), 0)
		cstrs = append(cstrs, b)
		return uintptr(unsafe.Pointer(&b[0]))
	}

	msgs := make([]chatMessage, len(roles))
	total := 0
	for i := range roles {
		msgs[i] = chatMessage{role: cstr(roles[i]), content: cstr(contents[i])}
		total += len(roles[i]) + len(contents[i])
	}
	tmplC := append([]byte(tmpl), 0)

	// llama.h recommends twice the characters of all messages; the call reports
	// the size it needs when that is not enough, so one retry always suffices.
	buf := make([]byte, 2*total+1024)
	n := chatApplyTemplate(&tmplC[0], &msgs[0], uint64(len(msgs)), addAssistant, &buf[0], int32(len(buf)))
	if int(n) > len(buf) {
		buf = make([]byte, n)
		n = chatApplyTemplate(&tmplC[0], &msgs[0], uint64(len(msgs)), addAssistant, &buf[0], int32(len(buf)))
	}
	runtime.KeepAlive(cstrs)
	runtime.KeepAlive(tmplC)
	runtime.KeepAlive(msgs)

	if n < 0 {
		return "", fmt.Errorf("llama.cpp does not implement this model's chat template")
	}
	return string(buf[:n]), nil
}

// BuiltinTemplates lists the chat families llama.cpp implements.
func BuiltinTemplates() []string {
	n := chatBuiltinTemplates(nil, 0)
	if n <= 0 {
		return nil
	}
	// []*byte rather than []uintptr: the strings are static C data, and letting
	// Go hold them as pointers keeps this free of a uintptr conversion that vet
	// rightly distrusts.
	ptrs := make([]*byte, n)
	if got := chatBuiltinTemplates(&ptrs[0], uint64(n)); got < 0 {
		return nil
	}
	out := make([]string, 0, n)
	for _, p := range ptrs {
		if p != nil {
			out = append(out, goString(p))
		}
	}
	return out
}

// goString copies a null-terminated C string.
func goString(p *byte) string {
	if p == nil {
		return ""
	}
	var n int
	for ptr := unsafe.Pointer(p); *(*byte)(ptr) != 0; ptr = unsafe.Add(ptr, 1) {
		n++
		if n > 1<<20 {
			break // not a sane C string; refuse to walk off the end
		}
	}
	return string(unsafe.Slice(p, n))
}

func libName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libllama.0.dylib"
	case "windows":
		return "llama.dll"
	default:
		return "libllama.so.0"
	}
}
