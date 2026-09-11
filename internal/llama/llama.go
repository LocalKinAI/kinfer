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
// The whole surface kinfer needs is about twenty entry points, so binding them
// here costs less than working around a mirror that drifts. Every struct below
// is transcribed from llama.h at the build embedded in internal/nativelib, and
// Open verifies the layout at runtime rather than trusting this comment.
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

// ModelParams mirrors struct llama_model_params (72 bytes).
type ModelParams struct {
	Devices              uintptr
	TensorBuftOverrides  uintptr
	NGpuLayers           int32
	SplitMode            int32
	MainGpu              int32
	_                    int32 // padding before the next pointer
	TensorSplit          uintptr
	ProgressCallback     uintptr
	ProgressCallbackData uintptr
	KvOverrides          uintptr
	VocabOnly            uint8
	UseMmap              uint8
	UseMlock             uint8
	CheckTensors         uint8
	UseExtraBufts        uint8
	NoHost               uint8
}

// ContextParams mirrors struct llama_context_params (120 bytes).
//
// This is the struct gollama got wrong. The first field is n_ctx — there is no
// seed here; sampling owns the seed now.
type ContextParams struct {
	NCtx          uint32
	NBatch        uint32
	NUbatch       uint32
	NSeqMax       uint32
	NThreads      int32
	NThreadsBatch int32

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
	batchGetOne        func(tokens *Token, n int32) Batch
	getLogitsIth       func(c Context, i int32) *float32
	getMemory          func(c Context) Memory
	memoryClear        func(mem Memory, data bool)
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
)

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
		{"llama_model_params", unsafe.Sizeof(ModelParams{}), 72},
		{"llama_context_params", unsafe.Sizeof(ContextParams{}), 120},
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
		{&batchGetOne, "llama_batch_get_one"},
		{&getLogitsIth, "llama_get_logits_ith"},
		{&getMemory, "llama_get_memory"},
		{&memoryClear, "llama_memory_clear"},
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
	} {
		// RegisterLibFunc panics on a missing symbol, which would abort the
		// program on an unexpected llama.cpp build. Convert it to an error so
		// callers can report something actionable.
		if err := register(b.fn, lib, b.name); err != nil {
			return err
		}
	}
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

// Decode runs one forward pass.
func Decode(c Context, b Batch) error {
	if rc := decode(c, b); rc != 0 {
		// 1 means "no KV slot"; llama.cpp prints the detail to stderr.
		return fmt.Errorf("llama_decode failed with code %d", rc)
	}
	return nil
}

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

func libName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libllama.dylib"
	case "windows":
		return "llama.dll"
	default:
		return "libllama.so"
	}
}
