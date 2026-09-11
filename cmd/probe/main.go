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
		forceTpl  = flag.String("template", "", "force a built-in chat family instead of the model's own (chatml, llama3, mistral)")
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
	tpl := chat.FromModel(model, *modelPath)
	if *forceTpl != "" {
		t, err := chat.Get(*forceTpl)
		if err != nil {
			fatal("%v", err)
		}
		tpl = t
	}
	promptText := *prompt
	if !*raw {
		var msgs []chat.Message
		if *system != "" {
			msgs = append(msgs, chat.Message{Role: "system", Content: *system})
		}
		msgs = append(msgs, chat.Message{Role: "user", Content: *prompt})
		promptText = tpl.Render(msgs)
		fmt.Printf("  ✅ chat template          %s\n", tpl.Name)
		// Apple Silicon steers low-QoS threads onto efficiency cores, where
		// llama_decode's CPU half takes roughly twice as long (measured: 2035
		// µs/token clamped to background against 719 at the default). Worth
		// printing, because it is invisible otherwise and explains an otherwise
		// baffling halving of throughput.
		initQoS()
		fmt.Printf("  thread QoS     : %s\n", qosReport())

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
	defer sampler.Close()

	fmt.Printf("── output ────────────────────────────────────\n  ")

	cur := tokens
	tGen := time.Now()
	var firstTokenAt time.Duration
	generated := 0

	// Time decode+sample apart from the wall clock.
	//
	// Ollama reports eval_count/eval_duration, which is llama.cpp's own timer
	// around the forward pass. Comparing that against kinfer's wall clock would
	// flatter Ollama by whatever detokenisation and string handling cost here,
	// so measure both and say which is which.
	var decodeTime, syncTime, sampleTime, pieceTime time.Duration

	// Streaming detokenization.
	//
	// llama_token_to_piece already returns decoded text, but emitting each
	// piece as it arrives still breaks multi-byte characters: a single CJK rune
	// is commonly split across 2-3 tokens, and each fragment on its own is
	// invalid UTF-8. So accumulate the pieces and emit only the prefix that has
	// become valid.
	var pieces []string
	emitted := 0
	stopped := false

	for i := 0; i < *maxTokens; i++ {
		tStep := time.Now()
		if err := llama.DecodeTokens(lctx, cur); err != nil {
			fmt.Println()
			fatal("Decode failed at token %d: %v", i, err)
		}
		tAfterDecode := time.Now()
		decodeTime += tAfterDecode.Sub(tStep)

		// Touch the logits before sampling. On Metal llama_decode may only
		// enqueue the forward pass, in which case the wait for the GPU happens
		// at the first read — and would otherwise be charged to the sampler.
		if row := llama.Logits(lctx, -1, nVocab); row != nil {
			_ = row[0]
		}
		tAfterSync := time.Now()
		syncTime += tAfterSync.Sub(tAfterDecode)

		// Sampling runs inside llama.cpp over its own logit buffer — no copy
		// of the 151,936-entry row crosses into Go.
		tok := sampler.Sample(lctx, -1)
		sampleTime += time.Since(tAfterSync)
		if llama.IsEOG(vocab, tok) {
			stopped = true
			break
		}

		// The real detokenizer — no byte-level post-processing needed.
		tPiece := time.Now()
		piece := llama.TokenToPiece(vocab, tok, false)
		pieceTime += time.Since(tPiece)
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
	fmt.Printf("  throughput     : %.1f tok/s  (wall clock, everything included)\n", float64(generated)/genSec)
	if decodeTime > 0 {
		n := float64(generated)
		fmt.Printf("  decode+sample  : %.1f tok/s  (comparable to Ollama's eval_duration)\n",
			n/(decodeTime+syncTime+sampleTime).Seconds())
		fmt.Printf("  llama_decode   : %5.0f µs/token\n", float64(decodeTime.Microseconds())/n)
		fmt.Printf("  first logit read:%5.0f µs/token  (GPU wait, if decode is async)\n", float64(syncTime.Microseconds())/n)
		fmt.Printf("  sampler        : %5.0f µs/token  (after the GPU is known to be done)\n", float64(sampleTime.Microseconds())/n)
		fmt.Printf("  token_to_piece : %.0f µs/token\n", float64(pieceTime.Microseconds())/n)
	}
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
