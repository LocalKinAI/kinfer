# kinfer

**A single-file local inference runtime for agent fleets.** No Docker, no
service to install, no separate llama.cpp — one binary, one copy, it runs.

```bash
$ kinfer pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
$ kinfer list
NAME                                                 SIZE  MODIFIED
qwen2.5-0.5b-instruct-q4_k_m                     468.6 MB  just now

$ kinfer run qwen "explain merkle trees in one sentence"
A Merkle tree is a data structure used in cryptography to store and
retrieve data in a way that is resistant to attacks…

$ kinfer serve
kinfer serving on :11500 — 1 model(s) in ~/.kinfer/models
  Ollama API : POST :11500/api/chat        GET :11500/api/tags
  OpenAI API : POST :11500/v1/chat/completions
```

```
$ ls -lh $(which kinfer)
-rwxr-xr-x  14M   # llama.cpp is inside — nothing else to install
```

> **Status: early but usable.** The inference core is done and the CLI works:
> pull, list, rm, run, ps, and an HTTP server speaking two dialects. What is
> *not* done is the reason the project exists — health probing, automatic
> failover, circuit breaking. See [CHANGELOG.md](CHANGELOG.md), which records
> exactly what works and what does not.

---

## Why this exists

LocalKin runs ~150 agents against cloud models. On 2026-07-23 one of those
models (`kimi-k2.5:cloud`) stopped responding — not an error, a *hang* — and the
whole fleet went silent for half a day. The failure was not in the model; it was
in having no second leg to stand on.

kinfer is that second leg: a local inference path that is **structurally
independent** of any cloud provider or background service.

## What makes it different

Existing local runners are built for *a person talking to a model*. kinfer is
built for *a fleet of agents that must not go down*.

|  | Ollama | Docker Model Runner | kinfer |
|---|---|---|---|
| Install | background service | Docker required | **copy one file** |
| Model source | registry + `hf.co/...` | Docker Hub / OCI / HF | HF directly |
| On-disk models | blob hashes | OCI layers | **plain filenames** |
| API | Ollama | Ollama + OpenAI | **Ollama + OpenAI** |
| Designed for | one user, one chat | container workflows | **agent fleets** |

Plain filenames matter more than it sounds: when something breaks at 2am you
want `ls ~/.kinfer/models` to tell you what you have, and you want `scp` to be a
valid way of moving a model to another machine.

## Commands

```bash
kinfer fit                   # what this machine can actually run
kinfer pull <repo>[:quant]   # Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
kinfer list                  # what is installed
kinfer rm <model>            # delete one
kinfer run <model> [prompt]  # chat — no prompt means interactive
kinfer ps                    # what the daemon is holding right now
kinfer serve [-addr :11500]  # serve over HTTP
```

### `kinfer run` keeps the model warm

`run` never loads a model itself. It talks to a background daemon, starting one
the first time, so the weights are loaded once instead of once per command —
the same reason `ollama run` feels instant.

```
$ kinfer run qwen "say hi in three words"    # 0.50s — daemon starts, model loads
Hi there!
$ kinfer run qwen "name three colors"        # 0.41s — warm
$ kinfer run qwen                            # no prompt: interactive
kinfer · qwen
  /bye, /exit    leave (the daemon keeps the model warm)
  /clear         forget this conversation
> 

$ kinfer ps
NAME                                               SIZE  UNTIL
qwen2.5-0.5b-instruct-q4_k_m                   468.6 MB  5m0s
```

An idle model is unloaded after `-keepalive` (default 5 minutes), because a 7B
left resident forever owns most of a 16 GB machine. `-local` skips the daemon
and loads in-process, which is how you see llama.cpp's own stderr when the
question is why a model failed to load.

Flags go before the model name. A flag after it would otherwise be joined into
the prompt and silently asked of the model, so `run` refuses instead.

### `kinfer fit`

Choosing a local model is guesswork: quantisation names carry no size, "7B"
says nothing about RAM, and getting it wrong does not fail cleanly — the machine
swaps until it is unusable. The runtime knows the answer, so it says it.

```
$ kinfer fit
Machine
  darwin/arm64 · 10 cores · 16 GB RAM · Apple GPU (Metal, unified memory)

Recommended
  7B at Q4_K_M  (~4.5 GB, comfortable)
  kinfer pull Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M

Also fits
  14B       9.1 GB  tight
  3B        1.9 GB  comfortable
  1.5B    998.4 MB  comfortable

Backend
  metal — offload every layer with -ngl 99

Installed
  qwen2.5-0.5b-instruct-q4_k_m       468.6 MB  comfortable

Notes
  · 7B also fits at Q6_K (~6.2 GB) — better quality at the same size
  · unified memory: the GPU draws from the same pool, so model size and
    everything else compete
  · estimates are weights only — a large context adds hundreds of MB of KV cache
```

Size estimates are calibrated against real files (Llama-3-8B Q4_K_M ≈ 4.9 GB,
Qwen2.5-7B ≈ 4.7 GB), and 40% of RAM is reserved for everything that is not the
model. The backend advice is measured rather than assumed — see below.

Model names accept unique prefixes — `kinfer run qwen` is enough when only one
model starts with it. Ambiguous prefixes are an error rather than a guess:
silently loading the wrong 7B model is worse than a message.

## Design

```
      cmd/kinfer              pull · list · rm · run · serve
            │
      internal/server         Ollama + OpenAI dialects, streaming
      internal/engine         one loaded model: prompt → tokens → text
            │
   ┌────────┼─────────┬──────────────┬────────────────┐
   │        │         │              │                │
 chat   sampling   store          hub            nativelib
 templates  temp/    ~/.kinfer/   Hugging Face   embeds llama.cpp
            top-k/p  models       downloads      in the binary
            │
      internal/llama            binds what gollama omits or gets wrong
            │
      gollama.cpp (purego)      no CGO, so cross-compilation survives
            │
      libllama.dylib + 7 ggml libraries   (4.6 MB, embedded)
```

**No CGO anywhere.** The native libraries load at runtime through `purego`,
which keeps `GOOS=linux go build` working from a Mac and keeps the binary free
of a C toolchain dependency.

## Build

```bash
# Fetch the native libraries once, into the package that embeds them
go run github.com/dianlight/gollama.cpp/cmd/gollama-download \
    -download -copy-libs -libs-dir internal/nativelib/libs

# The libraries end up inside the binary
go build -o kinfer ./cmd/kinfer
```

## Measured on M-series (Qwen2.5-0.5B Q4_K_M)

| Backend | Throughput | Model load |
|---|---|---|
| CPU (`-ngl 0`) | **123.7 tok/s** | 0.13 s |
| Metal, first run | 75.8 tok/s | 5.46 s ← one-off kernel compilation |
| Metal, warm | 114.5 tok/s | **0.14 s** |

**CPU beats Metal on a 0.5B model** — transfer overhead outweighs the compute
win at this size. `kinfer fit` already reports this (it recommends CPU for
anything at or below 1.5B); acting on it without being asked is Phase 3.

Sampling over the full 151,936-token vocabulary costs throughput (measured 40
tok/s during generation). `internal/sampling` sorts all candidates when top-k
only needs the best 40 — partial selection will bring that back.

## Roadmap

- [x] **Phase 0** — feasibility: pure-Go inference, Metal, single binary
- [x] **Inference core** — chat templates, sampling, stop detection, native bindings
- [x] **Phase 1** — `fit` / `pull` / `list` / `rm` / `run` / `serve`
- [ ] **Phase 2** — the reason this exists: health probing, automatic failover,
      circuit breaking, per-agent model routing
- [ ] **Phase 3** — apply `fit`'s backend choice automatically, honest benchmarks
      vs Ollama

## Upstream issues found

Five bugs in `gollama.cpp` v0.2.2. Four are solved properly — `internal/llama`
binds the real llama.cpp entry points through purego, so no CGO is introduced.
All of them deserve patches upstream.

**Vocabulary size is hardcoded to 32.** `Token_data_array_from_logits` contains
`nVocab := int32(32)` with the comment *"to avoid corruption issues"*. Qwen2.5
has 151,936 tokens, so every candidate list was truncated to the first 32 —
sampling over that produces garbage no matter how correct the algorithm is.
Solved by binding `llama_vocab_n_tokens`.

**Context parameters are shifted by one field.** llama.cpp removed `seed` from
`llama_context_params`, but gollama's Go mirror still declares it first, so
`NCtx` lands in `n_batch`, `NBatch` in `n_ubatch`, and so on. Nothing errors —
the settings just do not apply. Demonstrated with
`cp.Seed = 8192; cp.NCtx = 111` → `n_ctx = 8192`. Worked around in
`internal/engine/ctxparams.go`, with a self-check that fails loudly if a future
gollama fixes the layout and quietly re-breaks the workaround.

**`Token_to_piece` returns raw vocabulary text.** It calls
`llama_vocab_get_text` rather than `llama_token_to_piece`, so generated text
arrives byte-level encoded — `ĠMerkleĠtree` instead of ` Merkle tree`. Solved by
binding the real `llama_token_to_piece`.

**Only a greedy sampler is bound.** No temperature, top-k, top-p, or repetition
penalty — and greedy decoding degenerates into loops. Solved by
`internal/sampling`, which samples from the raw logits.

**`Config.LibraryPath` is accepted and discarded.** `ApplyConfig` unloads the
current library but never stores the new path. Setting `DYLD_LIBRARY_PATH`
in-process does not help either — the dynamic linker reads it at exec time.
Worked around in `internal/nativelib` by `chdir`-ing into the library directory
for the duration of `Backend_init`, since `dlopen` is called with a bare name.

## License

Apache 2.0
