// Command probe is the Phase 0 feasibility check for kinfer.
//
// It answers three questions before any product code gets written:
//  1. Can pure-Go (no CGO) load a GGUF and generate tokens at all?
//  2. Does Metal offload actually engage on Apple Silicon?
//  3. Is throughput in the same ballpark as Ollama?
//
// Run it twice to compare backends:
//
//	go run ./cmd/probe -ngl 0    # CPU only
//	go run ./cmd/probe -ngl 99   # offload everything to Metal
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/LocalKinAI/kinfer/internal/chat"
	"github.com/LocalKinAI/kinfer/internal/llama"
	"github.com/LocalKinAI/kinfer/internal/nativelib"
	"github.com/LocalKinAI/kinfer/internal/sampling"
)

func main() {
	var (
		modelPath = flag.String("model", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "path to a GGUF file")
		prompt    = flag.String("prompt", "Explain what a Merkle tree is, in two sentences.", "prompt to run")
		nGpuLayer = flag.Int("ngl", 99, "layers to offload to GPU (0 = pure CPU, 99 = all)")
		maxTokens = flag.Int("n", 64, "how many tokens to generate")
		nCtx      = flag.Int("ctx", 2048, "context size")
		system    = flag.String("system", "", "system prompt — a LocalKin soul goes here")
		raw       = flag.Bool("raw", false, "skip the chat template and continue the text directly")
		temp      = flag.Float64("temp", 0.7, "sampling temperature (0 = greedy)")
		topP      = flag.Float64("top-p", 0.95, "nucleus sampling threshold")
		topK      = flag.Int("top-k", 40, "keep only the K most likely tokens")
		repeatPen = flag.Float64("repeat-penalty", 1.1, "penalty on recently used tokens")
		seed      = flag.Int64("seed", 0, "sampling seed (0 = random)")
	)
	flag.Parse()

	if _, err := os.Stat(*modelPath); err != nil {
		fatal("model not found at %s — download a GGUF first", *modelPath)
	}

	fmt.Printf("── kinfer probe ──────────────────────────────\n")
	fmt.Printf("  binding : internal/llama (purego, no CGO)\n")
	fmt.Printf("  model   : %s\n", *modelPath)
	fmt.Printf("  ngl     : %d %s\n", *nGpuLayer, backendLabel(*nGpuLayer))
	fmt.Println()

	// Unpack the libraries this binary carries, and point gollama at them. This
	// is what makes kinfer standalone: no gollama-download, no ~/.cache/gollama,
	// no separately installed llama.cpp.
	tLib := time.Now()
	libDir, err := nativelib.Prepare()
	if err != nil {
		fatal("could not prepare embedded libraries: %v", err)
	}
	fmt.Printf("  ✅ libs unpacked          %6.2fs  %s\n", time.Since(tLib).Seconds(), libDir)

	tInit := time.Now()
	if err := llama.Bind(libDir); err != nil {
		fatal("could not bind llama.cpp symbols: %v", err)
	}
	defer llama.BackendFree()
	fmt.Printf("  ✅ native lib loaded      %6.2fs\n", time.Since(tInit).Seconds())

	// ---- load model ----
	mp := llama.DefaultModelParams()
	mp.NGpuLayers = int32(*nGpuLayer)

	tLoad := time.Now()
	model := llama.LoadModel(*modelPath, mp)
	if model == 0 {
		fatal("llama_model_load_from_file failed for %s", *modelPath)
	}
	defer llama.FreeModel(model)
	loadSec := time.Since(tLoad).Seconds()
	fmt.Printf("  ✅ model loaded           %6.2fs\n", loadSec)

	vocab := llama.GetVocab(model)
	nVocab := llama.NVocab(vocab)
	fmt.Printf("  ✅ vocabulary            %6d tokens\n", nVocab)

	// ---- context ----
	cp := llama.DefaultContextParams()
	cp.NCtx = uint32(*nCtx)

	lctx := llama.NewContext(model, cp)
	if lctx == 0 {
		fatal("llama_init_from_model failed")
	}
	if got := llama.NCtx(lctx); got != *nCtx {
		fatal("asked for n_ctx=%d but llama.cpp allocated %d", *nCtx, got)
	}
	defer llama.FreeContext(lctx)

	// ---- build the prompt ----
	// Without the template this is text continuation, not conversation: the
	// system prompt would be ignored and the soul would never take effect.
	tpl := chat.Detect(*modelPath)
	promptText := *prompt
	if !*raw {
		var msgs []chat.Message
		if *system != "" {
			msgs = append(msgs, chat.Message{Role: "system", Content: *system})
		}
		msgs = append(msgs, chat.Message{Role: "user", Content: *prompt})
		promptText = tpl.Render(msgs)
		fmt.Printf("  ✅ chat template          %s\n", tpl.Name)
	}

	// ---- tokenize ----
	tokens, err := llama.Tokenize(vocab, promptText, true, true)
	if err != nil {
		fatal("Tokenize failed: %v", err)
	}
	fmt.Printf("  ✅ prompt tokenized       %6d tokens\n", len(tokens))
	fmt.Println()

	// ---- generate ----
	sampler := sampling.New(sampling.Params{
		Temperature:   float32(*temp),
		TopK:          *topK,
		TopP:          float32(*topP),
		RepeatPenalty: float32(*repeatPen),
		RepeatLastN:   64,
		Seed:          *seed,
	})
	cands := make([]sampling.Candidate, nVocab)

	fmt.Printf("── output ────────────────────────────────────\n  ")

	cur := tokens
	tGen := time.Now()
	var firstTokenAt time.Duration
	generated := 0

	// Streaming detokenization.
	//
	// gollama's Token_to_piece returns the RAW byte-level-BPE vocab entry
	// ("ĠMerkle"), so every piece has to go through bpe.Decode. But decoding
	// pieces one at a time breaks multi-byte characters: a single CJK rune is
	// commonly split across 2-3 tokens, and each fragment on its own is invalid
	// UTF-8. So accumulate the raw pieces, decode the whole prefix each step,
	// and only emit the part that has become valid.
	var pieces []string
	emitted := 0
	stopped := false

	for i := 0; i < *maxTokens; i++ {
		if err := llama.DecodeTokens(lctx, cur); err != nil {
			fmt.Println()
			fatal("Decode failed at token %d: %v", i, err)
		}

		// Read the full logits row — nVocab entries, indexed by token id.
		row := llama.Logits(lctx, -1, nVocab)
		if row == nil {
			fmt.Println()
			fatal("no logits at token %d", i)
		}
		for j := range row {
			cands[j] = sampling.Candidate{ID: int32(j), Logit: row[j]}
		}

		tok := sampler.Sample(cands)
		if llama.IsEOG(vocab, tok) {
			stopped = true
			break
		}

		// The real detokenizer — no byte-level post-processing needed.
		piece := llama.TokenToPiece(vocab, tok, false)
		if generated == 0 {
			firstTokenAt = time.Since(tGen)
		}

		pieces = append(pieces, piece)
		full := strings.Join(pieces, "")

		// Stop markers arrive as ordinary text, so trim before display — else
		// the user sees the model's "<|im_end|>".
		full, done := tpl.TrimStop(full)

		if len(full) > emitted {
			pending := full[emitted:]
			// Hold back a trailing partial rune until the next token completes it.
			if utf8.ValidString(pending) {
				fmt.Print(pending)
				emitted = len(full)
			}
		}

		generated++
		if done {
			stopped = true
			break
		}
		cur = []llama.Token{tok}
	}

	genSec := time.Since(tGen).Seconds()
	fmt.Printf("\n\n── verdict ───────────────────────────────────\n")
	fmt.Printf("  generated      : %d tokens%s\n", generated,
		map[bool]string{true: " (hit stop marker)", false: " (hit -n limit)"}[stopped])
	fmt.Printf("  time to first  : %.2fs\n", firstTokenAt.Seconds())
	fmt.Printf("  throughput     : %.1f tok/s\n", float64(generated)/genSec)
	fmt.Printf("  model load     : %.2fs\n", loadSec)
	if generated > 0 {
		fmt.Printf("\n  ✅ PHASE 0 PASS — pure-Go inference works end to end.\n")
	} else {
		fmt.Printf("\n  ❌ generated nothing — investigate before proceeding.\n")
		os.Exit(1)
	}
}

func backendLabel(ngl int) string {
	if ngl == 0 {
		return "(pure CPU)"
	}
	return "(Metal offload expected)"
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "\n  ❌ "+format+"\n", args...)
	os.Exit(1)
}
