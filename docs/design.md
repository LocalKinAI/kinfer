# Design

How the code is laid out, and why llama.cpp is bound directly — part of [kinfer](../README.md).

## How the code is laid out

```
      cmd/kinfer              pull · list · rm · run · serve
            │
      internal/server         Ollama, OpenAI, Anthropic, Responses dialects, streaming
      internal/engine         one loaded model: prompt → tokens → text
            │
   ┌────────┼─────────┬──────────────┬────────────────┐
   │        │         │              │                │
 chat   sampling   store          hub            nativelib
 templates  temp/    ~/.kinfer/   Hugging Face   embeds llama.cpp
            top-k/p  models       downloads      in the binary
            │
      internal/llama            llama.cpp's C API, bound directly
            │
      purego                    no CGO, so cross-compilation survives
            │
      libllama + 7 ggml libraries   (llama.cpp b11998, 12.4 MB, embedded)
```

**No CGO anywhere.** The native libraries load at runtime through `purego`,
which keeps `GOOS=linux go build` working from a Mac and keeps the binary free
of a C toolchain dependency.

**One dependency.** `go.mod` requires `purego` and nothing else — the entire
llama.cpp surface kinfer uses is about twenty entry points bound in
`internal/llama`, including the by-value structs that usually force a project
onto libffi.

## Why llama.cpp is bound directly

kinfer started on `gollama.cpp` v0.2.2 and found five bugs in it, every one of
the kind that does not announce itself. It now binds llama.cpp itself, which
deleted the dependency and all three workarounds along with it.

**Vocabulary size was hardcoded to 32.** `Token_data_array_from_logits`
contained `nVocab := int32(32)` with the comment *"to avoid corruption issues"*.
Qwen2.5 has 151,936 tokens, so every candidate list was truncated to the first
32 — sampling over that produces garbage no matter how correct the algorithm is.

**`llama_context_params` was shifted, twice.** The Go mirror still declared a
`seed` field that llama.cpp removed, and was missing `flash_attn_type`
entirely — so fields landed one and then two slots early. A size assigned to
`NCtx` arrived as `n_batch`. Nothing errored; the context simply stayed at
llama.cpp's 512-token default, which surfaced as "prompt is 8267 tokens but the
context holds 4096" no matter what `-ctx` said. Transcribing the struct from
`llama.h` fixed it: asking for 4096 now yields 4096, verified after every load.

**`Token_to_piece` returned raw vocabulary text.** It called
`llama_vocab_get_text` rather than `llama_token_to_piece`, so generated text
arrived byte-level encoded — `ĠMerkleĠtree` instead of ` Merkle tree`.

**Only a greedy sampler was bound.** No temperature, top-k, top-p or repetition
penalty — and greedy decoding degenerates into loops. kinfer implemented the
whole pipeline in Go, which worked and cost two thirds of the throughput:
picking the best 40 of 151,936 candidates meant sorting all of them, once per
token, after copying every logit across the boundary. `internal/sampling` is now
a builder over `llama_sampler_chain_*`, and sampling is free — 44.7 tok/s
became 150, the same speed as greedy.

**`Config.LibraryPath` was accepted and discarded.** `ApplyConfig` unloaded the
current library but never stored the new path, and `DYLD_LIBRARY_PATH` is read
at exec time so setting it in-process does nothing. kinfer worked around it by
`chdir`-ing into the library directory — process-wide state, changed underneath
every other goroutine — because gollama called `dlopen` with a bare name.
`internal/llama` passes an absolute path, and libllama's `LC_RPATH` is
`@loader_path`, so its seven ggml siblings resolve from the same directory with
no help at all.

What replaced all of it is one file of about twenty bound entry points. purego
handles llama.cpp's by-value structs (`llama_batch` at 56 bytes,
`llama_context_params` at 120) directly, so libffi is not needed either — `bind`
asserts each struct's size against `llama.h` at startup, because a silent layout
drift is exactly the failure mode this package exists to end.
