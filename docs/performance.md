# Performance

Measured on Apple Silicon, against Ollama on the same files, and how far batching scales — part of [kinfer](../README.md).

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

**A note on measuring this at all.** Sustained benchmarking heats a laptop, and
throttling costs kinfer far more than it costs Ollama — over one session on the
M4 kinfer fell from 178 to 104 tok/s while Ollama held near 175. Run to run the
spread reaches ±40%, wide enough to swallow any change worth making. So every
comparison here measures Ollama in the same round and reports the ratio. A
kinfer number on its own, from that machine, means nothing.

### Against Ollama, on the same file

The box, ornith-1.5-35b (Q4_K_M, a 35B mixture of experts): kinfer's copy of it
is a link to Ollama's own blob, so both runtimes read the same bytes. Ollama
0.34.3 was measured as installed: one request at a time, a 262144-token context,
and its vision projector loaded. kinfer ran with `-slots 4`, which sizes the
same 262144 as 4 x 65536. Every request used `think: false`, temperature 0 and
a fixed `num_predict`. The box is shared — a ComfyUI pipeline starts a job every
few minutes — so each measurement waited for ComfyUI to be idle, and any during
which it started a job was thrown away and taken again.

| | kinfer | Ollama |
|---|---|---|
| memory | 26.9 GiB | 25.8 GiB |
| one request, decode | **106.6 tok/s** | 103.3 tok/s |
| 4 at once: aggregate / last one done | **166 tok/s / 9.6 s** | 102 tok/s / 15.8 s |
| 8 at once: aggregate / last one done | **154 tok/s / 20.8 s** | 102 tok/s / 31.5 s |
| 11,479-token prompt: prefill | **5.85 s** | 6.17 s |
| decode after it | **96.2 tok/s** | 90.8 tok/s |
| 103,641-token prompt: prefill | 126.4 s | **121.8 s** |
| decode after it | **61.4 tok/s** | 60.2 tok/s |

The replies were identical. Ollama still reads a 100k-token prompt 4% faster;
everything else is even or kinfer's, and several requests at a time kinfer
finishes in 60–66% of Ollama's time, because Ollama answers one request and
queues the rest.

It was not so before this build. With llama.cpp b10901 embedded, kinfer decoded
at 87 tok/s to Ollama's 103, and read the 11k prompt in 7.7 s to 5.9. Finding
out why, in order:

- **Not the slots.** kinfer with one slot of 262144, Ollama's exact layout, ran
  at the same 87.
- **Not the threads.** Ollama's own llama-server gave 103.1 tok/s with 4 threads
  and 102.2 with 20.
- **Not kinfer's own code.** `KINFER_DEBUG_SCHED` now splits each step: of
  11.5 ms per token, 1.32 went to queueing the forward pass, 10.06 to waiting
  for the GPU, 0.12 to sampling, and none to the scheduler.
- **The llama.cpp build.** The official b10901 llama-server, the same library
  kinfer embedded (byte for byte), ran at 83.8 tok/s. The official b11175 ran
  at 106–108. Ollama ships its own, somewhere between.

kinfer moved to b11175 (and since to b11998; see CHANGELOG.md). llama.h gained
functions between the two and changed none that kinfer binds. The prompt side also took Ollama's micro-batch: 2048
tokens a pass where the model and its cache leave room, 512 where they do not
(see `-batch`). Flash-Next on the same upgrade: 41.9 tok/s alone (was 33.6) and
66 together at four (was 53), still at 512 a pass since it fills the box.

### The same code on a Mac Studio

Repeating the protocol on an M3 Ultra (20 performance cores, 96 GB) settles what
the M4 could not. Same GGUF, byte for byte; eight rounds, order rotated:

| | median tok/s | spread | of Ollama |
|---|---|---|---|
| Ollama 0.34.0 | 293.7 | 291–319 | 100% |
| `llama-cli` | 288.9 | 267–310 | 98% |
| kinfer | 282.0 | **278–285** | **96%** |

kinfer is within 4% of Ollama and is the *steadiest* of the three, at ±1.3%. The
per-token breakdown is the explanation: its CPU work is ~540 µs on both machines
(448–470 µs in `llama_decode`, 88 µs sampling, 1 µs detokenising), stable here
and swinging between 548 and 1422 µs on the M4. **kinfer's cost was never the
work itself — it was whether a four-core laptop under thermal load could give
that work a performance core.**

The decoupling verdict flips with it: on the M3 Ultra a writer goroutine wins 7
rounds of 8, for +2.5% (95.5% → 97.9% of Ollama). It stays reverted because a
14% loss on a constrained laptop outweighs 2.5% on a workstation, but the
mechanism is confirmed and it should be revisited once Phase 2 has a scheduler —
by then the writer serves many streams and the generation loop must not block.

### The opportunity, measured

**Ollama runs one request at a time**, and a bigger machine does not change it:
on the M3 Ultra eight concurrent requests still take eight times as long as one,
its log still shows only `slot id 0`. kinfer no longer does — see *Continuous
batching* below. What it is worth — `llama-server --parallel 8 -cb`, same machine,
same model, same rounds:

| concurrent requests | llama-server `-cb` | Ollama | kinfer |
|---|---|---|---|
| 1 | 207.8 | 273.2 | 266.9 |
| 4 | **569.2** (2.74×) | 281.4 (1.03×) | 292.5 (1.10×) |
| 8 | **988.3** (4.76×) | 291.9 (1.07×) | 297.0 (1.11×) |

Aggregate tok/s; the multiplier is scaling against that server's own N=1.
Batching returns **3.4× more total throughput than Ollama at eight concurrent
requests.** For one person typing at a model that is a bad trade. For a fleet it
is the whole game, and neither runtime here is taking it.

### How far it scales, and the one batch size to avoid

Measured with `llama-batched-bench`, which puts no HTTP client in the way — a
350-token prompt and 200 generated, on the M3 Ultra:

| batch | aggregate tok/s | per stream | ms per step | vs 1 |
|---|---|---|---|---|
| 1 | 299.8 | 299.8 | 3.3 | 1.00× |
| 4 | 883.6 | 220.9 | 4.5 | 2.95× |
| **8** | **1336.9** | 167.1 | 6.0 | **4.46×** |
| 12 | 1103.8 | 92.0 | **10.9** | 3.68× |
| 16 | 1441.2 | 90.1 | 11.1 | 4.81× |
| 24 | 2092.1 | 87.2 | 11.5 | 6.98× |
| **32** | **2824.7** | 88.3 | 11.3 | **9.42×** |
| 64 | 4293.1 | 67.1 | 14.9 | 14.32× |
| 128 | 6483.0 | 50.6 | 19.7 | **21.63×** |

Step time nearly doubles between 8 and 12 — 6.0 ms to 10.9 ms — then stays flat
all the way to 32. Almost certainly Metal switching from a vector-matrix to a
matrix-matrix kernel. It splits the curve into three regimes:

- **at or below 8** the step is cheap: 4.5× aggregate, 167 tok/s per stream
- **12 to 16 is the worst place to be.** The step cost is paid and not yet
  amortised, so twelve concurrent requests produce *less* total throughput than
  eight
- **20 and above** the step time stops growing while the batch keeps growing, so
  throughput climbs nearly linearly: 9.4× at 32, 21.6× at 128

A slot count therefore comes from `{8, 32, 64, 128}` and never from 12–16 —
which intuition gets backwards, since 16 looks like it must beat 8. Margins do
fade past 32 (32→64 is +52%, 64→96 +32%, 96→128 +14%), but at 128 each stream
still gets 50 tok/s, far above what an agent needs.

That is Phase 2, and it is why per-stream speed is the smaller opportunity.

Model load is 0.49 s warm. The first Metal run on a machine pays a one-off ~5 s
for shader compilation, which macOS then caches — every later process, not just
every later request, starts fast.

**CPU beats Metal on a 0.5B model** — transfer overhead outweighs the compute
win at this size. `kinfer fit` already reports this (it recommends CPU for
anything at or below 1.5B); acting on it without being asked is Phase 3.

## Continuous batching

Conversations run side by side as separate sequences in one KV cache, advanced
together by a single forward pass per step. One goroutine owns the context and
everything else hands it work over a channel.

```
$ kinfer serve -slots 8
kinfer serving on :11500 — 1 model(s) in ~/.kinfer/models
  8 slots × 4096 tokens — conversations share one forward pass
```

Eight concurrent requests on the M4, with `KINFER_DEBUG_SCHED=1`:

```
sched:   44 steps/s   mean batch   8.0 tokens   mean active slots  8.0
```

Every step carries one token from each of the eight slots: 44 × 8 = **352 tok/s
aggregate against 164 for a single stream.** `llama-batched-bench` puts this
machine's own ceiling at a batch of eight at 393.5 tok/s, so the scheduler
reaches **89% of what llama.cpp achieves on the same hardware**.

The gain is the machine's to give, not the scheduler's. This M4 saturates early
— 2.39× at a batch of eight, where the M3 Ultra reaches 4.46× there and 21.6× at
128. Slot counts come from `{8, 32, 64, 128}`; see the table above for why never
12–16.

Two things the scheduler will not do. It never blocks on a client: a consumer
that stops reading stalls only its own slot, because a blocked send would freeze
every other conversation with it. And prompts are prefilled in 512-token chunks
rather than whole, so an agent arriving with an 8k system prompt shares a step
with the conversations already generating instead of stopping them.

Sequences must not leak into each other, so that is tested directly: eight
arithmetic questions with distinguishable answers, asked serially and then all
at once, must produce identical replies. They do.
