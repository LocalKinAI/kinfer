// Package llama binds the llama.cpp entry points that gollama.cpp either does
// not expose or gets wrong.
//
// Three of them matter enough to justify their own bindings:
//
//   - llama_vocab_n_tokens — gollama hardcodes the vocabulary size to 32
//     ("to avoid corruption issues"), which caps every candidate list at 32 of
//     Qwen's 151,936 tokens. Sampling is impossible without the real number.
//   - llama_token_to_piece — gollama's Token_to_piece actually calls
//     llama_vocab_get_text, returning the raw byte-level-BPE entry ("ĠMerkle")
//     instead of decoded text (" Merkle").
//   - llama_vocab_is_eog — there is no other way to know the model has stopped;
//     matching stop strings in the decoded text is a guess by comparison.
//
// Binding goes through purego like the rest of the stack, so this adds no CGO
// and does not disturb cross-compilation.
package llama

import (
	"fmt"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

var (
	bindOnce sync.Once
	bindErr  error

	modelGetVocab func(model uintptr) uintptr
	vocabNTokens  func(vocab uintptr) int32
	vocabIsEOG    func(vocab uintptr, token int32) bool
	vocabEOS      func(vocab uintptr) int32
	tokenToPiece  func(vocab uintptr, token int32, buf *byte, length int32, lstrip int32, special bool) int32
	nCtx          func(ctx uintptr) uint32
)

// Bind loads libllama from dir and resolves the extra symbols. It is safe to
// call repeatedly; binding happens once.
//
// Call it after nativelib.Load — that unpacks the library and initialises the
// backend. dlopen here returns a handle to the same already-loaded image rather
// than mapping a second copy.
func Bind(dir string) error {
	bindOnce.Do(func() { bindErr = bind(dir) })
	return bindErr
}

func bind(dir string) error {
	path := filepath.Join(dir, libName())

	// RTLD_GLOBAL so symbols resolve against the copy the backend already uses;
	// two independently loaded copies of libllama would each carry their own
	// state and hand out handles the other cannot interpret.
	lib, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}

	type binding struct {
		fn   any
		name string
	}
	for _, b := range []binding{
		{&modelGetVocab, "llama_model_get_vocab"},
		{&vocabNTokens, "llama_vocab_n_tokens"},
		{&vocabIsEOG, "llama_vocab_is_eog"},
		{&vocabEOS, "llama_vocab_eos"},
		{&tokenToPiece, "llama_token_to_piece"},
		{&nCtx, "llama_n_ctx"},
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

// Vocab returns the model's vocabulary handle. Every other call here needs it.
func Vocab(model uintptr) uintptr { return modelGetVocab(model) }

// NVocab returns the true vocabulary size — the number gollama hardcodes to 32.
func NVocab(vocab uintptr) int32 { return vocabNTokens(vocab) }

// IsEOG reports whether a token ends generation (end-of-sequence,
// end-of-turn, or any other terminator this model defines).
func IsEOG(vocab uintptr, token int32) bool { return vocabIsEOG(vocab, token) }

// EOS returns the primary end-of-sequence token id.
func EOS(vocab uintptr) int32 { return vocabEOS(vocab) }

// TokenToPiece converts one token to its decoded text — the real
// llama_token_to_piece, so the result needs no byte-level BPE post-processing.
//
// special controls whether special tokens render as text ("<|im_end|>") or as
// nothing. Generation wants false; debugging a prompt wants true.
func TokenToPiece(vocab uintptr, token int32, special bool) string {
	// Ask with a small buffer first. The C function returns the needed length as
	// a negative number when it does not fit, so one retry always suffices.
	buf := make([]byte, 64)
	n := tokenToPiece(vocab, token, &buf[0], int32(len(buf)), 0, special)
	if n < 0 {
		buf = make([]byte, -n)
		n = tokenToPiece(vocab, token, &buf[0], int32(len(buf)), 0, special)
		if n < 0 {
			return ""
		}
	}
	if n == 0 {
		return ""
	}
	return unsafe.String(&buf[0], int(n))
}

// NCtx reports the context size llama.cpp actually allocated — the ground truth
// for verifying that a requested size took effect.
func NCtx(ctx uintptr) int { return int(nCtx(ctx)) }

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
