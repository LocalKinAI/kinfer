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
-rwxr-xr-x  17M   # llama.cpp is inside — nothing else to install
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
      internal/llama            llama.cpp's C API, bound directly
            │
      purego                    no CGO, so cross-compilation survives
            │
      libllama + 7 ggml libraries   (llama.cpp b10901, 8.2 MB, embedded)
```

**No CGO anywhere.** The native libraries load at runtime through `purego`,
which keeps `GOOS=linux go build` working from a Mac and keeps the binary free
of a C toolchain dependency.

**One dependency.** `go.mod` requires `purego` and nothing else — the entire
llama.cpp surface kinfer uses is about twenty entry points bound in
`internal/llama`, including the by-value structs that usually force a project
onto libffi.

## Build

```bash
# Fetch the native libraries once, into the package that embeds them.
# Copy the versioned sonames, not the plain names: llama.cpp's macOS builds
# reference @rpath/libggml.0.dylib, and //go:embed cannot carry the symlinks
# that the release tarball uses for the unversioned names.
V=b10901
curl -sfL "https://github.com/ggml-org/llama.cpp/releases/download/$V/llama-$V-bin-macos-arm64.tar.gz" \
    | tar xz -C /tmp
mkdir -p internal/nativelib/libs/darwin_arm64_$V
for f in libllama libggml libggml-base libggml-cpu libggml-blas libggml-metal libggml-rpc libmtmd; do
    cp -L /tmp/llama-$V/$f.0.dylib internal/nativelib/libs/darwin_arm64_$V/
done

# The libraries end up inside the binary
go build -o kinfer ./cmd/kinfer
```

## Measured on M-series (Qwen2.5-0.5B Q4_K_M)

This machine's throughput drifts by tens of percent over a session, so a number
measured on its own says very little. Everything below is measured against
Ollama 0.34.0 running on the same machine at the same time — same GGUF, same
prompt, both over HTTP, greedy, 200 tokens, five interleaved rounds.

**Sampling used to cost two thirds of the throughput and now costs nothing.**
The pipeline runs inside llama.cpp over its own logit buffer, doing partial
selection rather than sorting all 151,936 candidates in Go. Sampled generation
went from 44.7 tok/s to the same speed as taking the argmax.

**Upgrading llama.cpp b6862 → b10901 bought nothing measurable.** Normalising
against Ollama to cancel the drift, the net is between −2% (median) and +8%
(best) — inside the noise. The upgrade is kept for eleven months of upstream
fixes, and because it proved the struct assertions work, not for speed. An
earlier +16% claim here was wrong: it compared runs from different sessions,
which on this machine is not a measurement.

**kinfer's own overhead is 46%, and it is all CPU-side work between GPU
dispatches.** Every layer measured in the same interleaved rounds, same GGUF,
greedy, 200 tokens:

| | median tok/s | of native |
|---|---|---|
| `llama-bench` (pipelined, never samples) | 181.1 | 103% |
| `llama-cli` (real autoregressive generation) | 176.6 | 100% |
| Ollama 0.34.0 | 180.2 | 102% |
| a Go loop that only calls `llama_decode` | 157.2 | 89% |
| + sampling and detokenisation (`cmd/probe`) | 136.2 | 77% |
| + HTTP streaming (`kinfer serve`) | 95.0 | 54% |

`llama-cli` is the honest ceiling: it generates autoregressively with a sampler,
exactly as kinfer does, on the same embedded b10901 libraries. `llama-bench` is
not a fair comparison — its generation loop never reads the logits, so its CPU
work overlaps the GPU in a way real generation cannot.

Reading the logits is free (164.8 tok/s against 161.9 without), so the cost is
not the GPU synchronisation. It is that **the GPU sits idle while Go works.**
`cmd/probe` breaks a token down: the GPU wait is a stable ~5.4 ms, the sampler
90–219 µs, detokenisation 2–5 µs — and `llama_decode`'s own CPU half swings
between 548 and 1422 µs. Each millisecond spent on the CPU between dispatches is
a millisecond the GPU is not computing, and kinfer serialises all of it.

That points at the HTTP layer — a JSON encode, a socket write and a flush per
token, all on the critical path between two GPU dispatches. **Moving it to a
writer goroutine behind a buffered channel was tried and made things worse.**
Measured against Ollama in every round, with the order rotated so no
configuration is systematically first: writing inline reaches 83% of Ollama,
writing from a second goroutine reaches 71%.

The likely reason is that this is not an I/O-bound problem at all. kinfer's
per-token cost is CPU work, and a second runnable goroutine doing syscalls
competes for the four performance cores — pushing generation onto an efficiency
core costs more than the inline write ever did. Whatever fixes this has to
*remove* CPU work from the token loop rather than move it elsewhere. It is also
a warning for the scheduler in Phase 2: extra goroutines are not free on this
hardware, and the assumption has to be measured rather than reasoned about.

Ruled out along the way, each with a measurement: Go's GC (`GOGC=off`),
`GOMAXPROCS`, async preemption, `runtime.LockOSThread`, `n_batch`, `go run`
versus a prebuilt binary, process nice level (native runs at the same one),
Metal residency (`GGML_METAL_NO_RESIDENCY`, `RESIDENCY_KEEP_ALIVE_S`), whether
Ollama is holding a model at the time, and thread QoS — kinfer already runs at
`user-interactive`, though clamping it to `background` does halve throughput,
which is why `cmd/probe` now prints it.

**A note on measuring this at all.** Sustained benchmarking heats the machine,
and throttling costs kinfer far more than it costs Ollama — over one session
kinfer fell from 178 to 104 tok/s while Ollama held near 175, which is itself
evidence that kinfer's critical path is CPU-bound where Ollama's is not. Run to
run the spread reaches ±40%, wide enough to swallow any change worth making. So
every comparison here measures Ollama in the same round and reports the ratio.
A kinfer number on its own, from this machine, means nothing.

**Ollama runs one request at a time.** Eight concurrent requests take eight
times as long as one, aggregate throughput pinned at 165 tok/s, and its log only
ever shows `slot id 0`. Still true in 0.34.0. Per-stream speed is the smaller of
the two opportunities.

Model load is 0.49 s warm. The first Metal run on a machine pays a one-off ~5 s
for shader compilation, which macOS then caches — every later process, not just
every later request, starts fast.

**CPU beats Metal on a 0.5B model** — transfer overhead outweighs the compute
win at this size. `kinfer fit` already reports this (it recommends CPU for
anything at or below 1.5B); acting on it without being asked is Phase 3.

## Roadmap

- [x] **Phase 0** — feasibility: pure-Go inference, Metal, single binary
- [x] **Inference core** — chat templates, sampling, stop detection, native bindings
- [x] **Phase 1** — `fit` / `pull` / `list` / `rm` / `run` / `serve`
- [ ] **Phase 2** — the reason this exists: health probing, automatic failover,
      circuit breaking, per-agent model routing
- [ ] **Phase 3** — apply `fit`'s backend choice automatically, honest benchmarks
      vs Ollama

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

## License

Apache 2.0
