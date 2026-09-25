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
  Anthropic  : POST :11500/v1/messages          (Claude Code)
  Responses  : POST :11500/v1/responses         (Codex)
```

```
$ ls -lh $(which kinfer)
-rwxr-xr-x  17M   # llama.cpp is inside — nothing else to install
```

> **Status: early but usable.** The inference core is done and the CLI works:
> pull, list, rm, run, ps, and an HTTP server speaking four dialects. What is
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
| API | Ollama + OpenAI + Anthropic + Responses | Ollama + OpenAI | **Ollama + OpenAI + Anthropic + Responses** |
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
kinfer serve [-addr :11500]  # serve over HTTP; context and slots sized to the model
kinfer install [serve flags] # keep serve running: at login and after a crash (macOS)
kinfer uninstall             # stop it and remove the service
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
left resident forever owns most of a 16 GB machine. A request can say otherwise,
the way Ollama's do: `keep_alive` on a chat request is how long the model stays
after it (seconds, `"5m"`, or negative for forever), and
`POST /api/generate {"model": m, "keep_alive": 0}` — what `ollama stop` sends —
unloads it, answering once the memory is free. The flag's `0` means never; the
request's `0` means now, as it does in Ollama. `-local` skips the daemon
and loads in-process, which is how you see llama.cpp's own stderr when the
question is why a model failed to load.

Flags go before the model name. A flag after it would otherwise be joined into
the prompt and silently asked of the model, so `run` refuses instead.

### `serve` sizes itself, like Ollama does

`kinfer serve` with no flags reads the context the model was trained for, looks
at the memory its weights left, and picks a context and a slot count that fit:

```
sizing: 8 slots + 4 prefix x 65536 tokens each — the model was trained for 262144,
        the cache should cost about 15.7 GiB of the 45.7 GiB left. -slots and -ctx override this.
```

The policy: spend at most half of what is left after the weights on the cache
(the free-memory figure is per process, and a machine serving a fleet is rarely
running one thing); give each conversation the trained context, rounded down to
a power of two, as far as that half allows; and when that comes out below what a
real prompt needs, give up slots before giving up context — one slot of 32768
answers slowly, eight of 4096 answer nothing an agent sends. "What a real prompt
needs" is 32768, or the trained context if that is less: a coding agent's opening
prompt alone is 7,428 tokens (Codex) and about 25k (Claude Code).

The estimate reads the hybrid geometry from the GGUF header
(`full_attention_interval`, the `ssm.*` keys): on the Qwen3.5 family only every
fourth layer holds a KV cache, and the rest hold a per-sequence state that does
not grow with the context. Counting every layer put the 35B's cache at twice its
size and sized a server to 32768 tokens on a machine with room for 65536 — which
refused a 45k-token prompt that Ollama took without comment. That evening is why
this exists. After the context is built the cost is measured, and if the measure
leaves under 2 GiB it is rebuilt smaller — seconds, against the weights'
half-minute — in the same order of sacrifice: slots first, then the prefix
pool's own cells, then pool entries, and the context only after that. A pool
without cells of its own keeps its prompts in whatever the slots leave free and
gives them back when a slot needs the room, which on a machine with room for one
conversation still carries an agent's turns forward:

```
sizing: 1 slots + 1 prefix x 32768 tokens measured 2.6 GiB, leaving 1.7 GiB — the prefix pool shares the slots' cells instead, keeping 32768 tokens each
memory: MTL0   model 73.4 GiB + context 1.7 GiB -> 2.6 GiB free of 77.8 GiB (this process only)
        context is 32768 tokens: -ctx 32768 per conversation x 1 slots, with 1 prefix entries in whatever the slots leave free
```

### A conversation longer than its slot loses its oldest turns

Ollama never refuses a conversation for being long, and neither does kinfer. A
prompt that does not fit is fitted: the system prompt, the latest request and the
newest turn stay, and the oldest turns go until the rest fits with room left to
reply (what the caller asked for, up to an eighth of the slot). A tool call and
its results go together, so no result outlives the call that asked for it. Turns
are dropped in chunks at points that depend only on the turns before them, so the
kept part stays the same for several turns and the prefix pool keeps matching it.

```
context: dropped the oldest 16 of 28 messages so the prompt fits a 4096-token conversation with room to reply — now 3514 tokens
```

Only a prompt whose system prompt, request and newest turn do not fit by
themselves is refused, with a 413.

`kinfer plan <model>` prints the same decision without loading anything.
Explicit `-ctx` and `-slots` are never second-guessed: they are the layout the
server starts in and comes back to. The next section is the one way it leaves it.

### `-slots` without `-ctx`: each model's context sized the way Ollama sizes one

Ollama gives a request that names no context 262144 tokens on a machine with at
least 47 GiB of GPU memory, 32768 with 23, and 4096 below that, never more than
the model was trained for. The number is per machine, not per model: on the box
it gave ornith-1.5-35b and qwen3.8 262144 each, and would ask the same of a model
whose weights leave room for a quarter of it.

With `-slots N` and no `-ctx`, kinfer takes that number as the total the slots
share, halves it until it fits in what the weights left (keeping 1 GiB free), and
gives each slot an equal part. Flex hands one long prompt the whole total. A
server started once with `-slots 4` sizes each model as it loads:

```
$ kinfer plan -slots 4 <model>        on the box, 77.8 GiB GPU budget
Qwen3.8-Flash-Next  (73.5 GB)   4 x 16384, 65536 shared, cache about 1.9 GB
ornith-1.5-35b      (20.2 GB)   4 x 65536, 262144 shared, cache about 5.3 GB
qwen3.8-27b         (15.7 GB)   4 x 65536, 262144 shared, cache about 16.6 GB
qwen2.5-0.5b        (0.5 GB)    4 x 8192, 32768 shared (trained for 32768)
```

ornith gets the memory Ollama spends on its one request and serves four at once.
Flash-Next gets the 4 x 16384 that `-ctx 16384 -slots 4` used to fix for every
model, including the ones with room for more.

### The micro-batch, sized the way Ollama sizes it

llama.cpp reads a prompt a micro-batch at a time, 512 tokens unless told
otherwise, and a bigger pass keeps the GPU busier. kinfer picks it with Ollama's
rule (`automaticGenerationBatch` in its server/sched.go). It starts at 2048
tokens for a context over 32768 and at 1024 over 4096. It steps down while the
model and its cache are more than 60% of the GPU budget (75% for 1024), or the
extra memory the bigger pass takes (2 GiB, or 768 MiB) is not there:

```
batch: the GPU reads 2048 prompt tokens a pass, sized the way Ollama sizes it — the model and its cache are 26.9 GiB of 77.8 GiB
batch: the GPU reads 512 prompt tokens a pass, sized the way Ollama sizes it — the model and its cache are 76.2 GiB of 77.8 GiB
```

On the box that read an 11,479-token prompt into ornith-1.5-35b in 5.85 s
instead of 7.7. Flash-Next fills the machine and stays at 512. A load whose
measured memory comes out short gives up the bigger pass before anything a
conversation would notice. `-batch N` sets it outright.

### A prompt too long for its slot gets a longer one

llama.cpp splits a context evenly between its slots, so `-ctx 16384 -slots 4` is
four conversations of 16384, and the longest prompt it takes is a quarter of the
cache. On the box a 44,555-token prompt was refused with a 413 although the cache
held 65,536 tokens in all. Giving every slot 65,536 does not fit there: 8.2 GiB
of cache where the weights leave 4.4.

So the total stays the same and the shape changes. A prompt that will not run
whole in a slot waits for the slots to finish what they are doing, and the cache
is rebuilt as the most slots that still hold it:

```
slots: 65536 tokens of context as 4 x 16384; a prompt that needs a longer slot gets 3 x 21760, 2 x 32768, 1 x 65536 while it runs — the rest wait, and 4 x 16384 comes back when nothing needs the longer slots. -flex=false keeps 4 x 16384.
slots: 4 x 16384 -> 1 x 65536 for a 44555-token prompt (it queued 10.9s; the cache was rebuilt in 224ms, leaving 10.1 GiB free (this process only))
slots: 1 x 65536 -> 4 x 16384 — nothing running needs the longer slots (it queued 1m56.3s; the cache was rebuilt in 223ms, leaving 9.8 GiB free (this process only))
slots: 4 x 16384 -> 2 x 32768 for a 27174-token prompt (it queued 2m4s; the cache was rebuilt in 246ms, leaving 10.0 GiB free (this process only))
```

The weights stay where they are; only the cache is reallocated, in about a quarter
of a second on the box. (The free memory in those lines is with the box's GPU budget
raised to 86 GiB that day; at macOS's default the same cache leaves about 1.6.) The request at the head of the queue decides, and nothing
behind it is admitted until it has its slot, so short requests arriving after a
long one cannot starve it. While it runs, a short request that finds a free longer
slot takes it rather than wait. Once nothing running needs the longer slots, the
next request that wants more of them brings the layout back the same way.

What it costs is waiting. A long prompt waits for the slots to drain, and the
requests that arrive meanwhile wait for it. A prompt no layout holds whole gets the
longest slot on offer and loses its oldest turns there, as above. A layout that
would leave less memory than the loaded one did is refused before any batch runs
in it, and not tried again. The prefix pool lives only in the loaded layout.
`-flex=false` keeps the layout fixed.

### `kinfer install` keeps the server up

The daemon `run` starts is a plain background process: a reboot loses it, and
so does a crash. A machine that serves a fleet needs `serve` to come back on
its own, which is the operating system's job. `install` hands it over:

```
$ kinfer install -addr :11590 -keepalive 0
installed ~/Library/LaunchAgents/ai.localkin.kinfer.plist
  runs: /usr/local/bin/kinfer serve -addr :11590 -keepalive 0
  log : ~/.kinfer/serve.log
  starts at login and restarts if it exits; kinfer uninstall removes it
```

`-keepalive 0` keeps the model resident for good, which suits a machine that does
nothing else. On one that is shared, give it a time instead. The box's job ran with
`-keepalive 0`, so a 73.4 GiB model held most of a 96 GB machine around the clock,
and an Ollama model and a ComfyUI job that other sessions ran there stalled or failed
against it. Installed with `-keepalive 10m` now, it gives the memory back ten
minutes after its last request, and the next request loads it again.

It is a per-user launchd job — no sudo, the same arrangement Ollama's app uses —
started at login and restarted within seconds of any exit (measured: `kill -9`,
back in 10s). The flags are recorded verbatim, after being run through `serve`'s
own parser, so a typo is refused at the keyboard instead of by a job that fails
at every login. Running `install` again replaces the job; `uninstall` stops it
and removes the file. macOS only for now.

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
      libllama + 7 ggml libraries   (llama.cpp b11175, 8.4 MB, embedded)
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
# Fetch llama.cpp's libraries once, into the package that embeds them
# (8 files, 8.8 MB; they are not committed — every upgrade is a new set).
scripts/fetch-libs.sh

# The libraries end up inside the binary (19 MB in all)
go build -o kinfer ./cmd/kinfer
```

`scripts/fetch-libs.sh` takes the llama.cpp build kinfer is checked against
(b11175) from the official release, copies the eight `.0.dylib` files into
`internal/nativelib/libs/darwin_arm64_b11175/`, and removes any other version
there. Run it with `-f` to fetch again, or with a build number to try another
release — after diffing its `llama.h`; see `internal/nativelib/libs/README.md`.

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

kinfer now embeds b11175. llama.h gained functions between the two and changed
none that kinfer binds. The prompt side also took Ollama's micro-batch: 2048
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

## Prefix pool

An agent's system prompt is identical byte for byte on every turn it ever takes,
and a soul runs to thousands of tokens. Without help, each turn prefills all of
it again, and a hundred agents sharing a soul pay for it a hundred times.

Recently seen prompt prefixes stay resident in the KV cache under their own
sequence ids. A request that begins the same way adopts those cells rather than
recomputing them — and nothing is copied: `llama_memory_seq_cp` tags cells that
already exist with a second sequence id, so taking over a 4000-token prefix
costs a pointer walk. It needs llama.cpp's unified KV buffer, which is what
allows a cell to belong to more than one sequence.

Measured on the M3 Ultra with a 3,100-word system prompt, time to first token:

| | no pool | prefix pool |
|---|---|---|
| same soul, repeated | 266 ms | **17 ms** |
| multi-turn conversation, turns 2–5 | 283 ms | **22 ms** |
| **8 agents sharing a soul, concurrent** | 2.38 s wall, 1767 ms median | **0.16 s wall, 68 ms median** |
| six distinct souls through four pool slots | 91 ms | 94 ms |

The last row is the control that matters: prefixes that do not repeat get no
benefit, and pay no penalty for the attempt.

```
$ kinfer serve -slots 8 -prefix 4
  8 slots × 4096 tokens — conversations share one forward pass
  4 prefix slots — a repeated system prompt is prefilled once
```

Each pooled prefix costs a sequence's worth of KV cache, the same as a slot, so
this is memory traded for latency — unless memory is short and nothing was
asked for, in which case the pool shares the slots' cells (see sizing above). A match must leave at least one token to
decode — logits exist only for tokens that went through a forward pass, and a
prompt adopted whole would have nothing to sample the first reply token from.

**Reuse is correct but not bit-exact, and that is worth knowing before turning
it on.** How much of a prompt gets adopted depends on what the pool happens to
hold: the first request prefills its question in one batch of a dozen tokens,
and a repeat decodes a single token instead. Different batch shapes mean
different floating-point reduction orders, so a logit that was a near-tie can
land the other way.

Measured: with the pool off, the same question produced one output in ten runs;
with it on, two. Across twenty varied prompts nineteen were byte-identical and
one differed — an arithmetic question the model was genuinely torn on, where the
pooled answer happened to be the better one. Nothing is misaligned: a prompt
reused in full still lets the model quote the last sentence of its own system
prompt, which a position error would destroy. What is lost is the guarantee that
a fixed seed reproduces a previous run exactly, so `-prefix off` turns the pool
off for callers who need it.

`-prefix` takes `auto` (the default), `off`, or a number of entries. A number is
taken as given. `auto` gets only the memory the conversations leave: with
`-slots` and `-ctx` both set, the pool takes cells of its own if they fit, shares
the slots' cells if only that fits, and is left out otherwise — `-prefix 0`
used to mean "sized with the rest", and on a 96 GB box running a 73.4 GiB model
that sized four entries of 16384 into 0.0 GiB. `0` now means none, like `off`.

## Chat templates come from the model

An instruct model answers properly only when the prompt is wrapped the way it
was fine-tuned to see. kinfer used to guess that from the filename and knew
three families, so anything else silently got ChatML — a format it had never
seen. The template it was actually trained on is right there in the GGUF's
metadata, so kinfer reads it.

`llama_chat_apply_template` does not run Jinja. It matches the stored template
string against the families llama.cpp implements in C++ and applies that one, so
binding it buys **54 families** without shipping a template engine.

The difference, on Gemma 3 — which has no system role at all:

```
from the GGUF   <start_of_turn>user\nYou are terse.\n\nHi.<end_of_turn>\n<start_of_turn>model\n
guessed         <|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\n…
```

The guess does not merely look wrong, it shows: asked the same question through
each, the model's own template stops cleanly while the guessed one leaks
`<|im_end` into the reply, because ChatML's marker is not what this model was
taught to stop at.

Generation stops on an end-of-generation token — `llama_vocab_is_eog` reports
whichever terminator the model uses — so a template read from metadata needs no
stop strings of its own. The three hand-written families remain as a fallback
for a file that carries no template, or one llama.cpp does not implement, and
`-template` still forces one.

## Tool calls

A model can ask for a function to be run, in both dialects:

```
POST /api/chat   {"tools": [{"type":"function","function":{"name":"get_current_weather", …}}]}

{"message": {"role":"assistant", "content":"",
             "tool_calls":[{"function":{"name":"get_current_weather",
                                        "arguments":{"city":"Berlin","unit":"celsius"}}}]},
 "done": true, "done_reason": "tool_calls"}
```

Send the result back as a `tool` message and the model answers from it — *"The
current weather in Berlin is light rain with a temperature of 7 degrees
Celsius."*

`llama_chat_apply_template` takes only `{role, content}` — it has no tools
parameter — so the tool branches of a model's Jinja template are unreachable
through the C API. What is reachable is the convention those branches encode,
and kinfer reproduces it exactly as Qwen's own template writes it: the
signatures join the end of the system prompt, an assistant's calls become
`<tool_call>` blocks in its text, and a result becomes a user turn wrapped in
`<tool_response>`, because no instruct model has a `tool` role of its own. The
template then has nothing unusual left to render.

**There is no single convention**, so kinfer reads which one a model knows out
of its own chat template and both declares and parses in that one. Two are
implemented:

| | declaration | a call looks like |
|---|---|---|
| **hermes** | "You may call one or more functions…" | `<tool_call>{"name":…,"arguments":{…}}</tool_call>` |
| **function-xml** | "You have access to the following functions:" | `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` |

Getting this wrong is not a near miss. Told to answer in JSON when it had been
taught the XML form, `ornith-1.5:35b` produced `{"name":_get_weather"` —
deterministically, four runs out of four. It was trying to obey an instruction
that contradicted its training. Reading the convention from the template instead
makes the same model answer correctly, with `days: 3` arriving as a number
rather than the string the XML form carries.

Both conventions wrap their calls in `<tool_call>`, so only the inner form tells
them apart — checking the outer tag first would misread every XML model as
Hermes.

A model whose template describes neither is refused rather than answered
uselessly, since it would reply in prose however the functions are declared:

```
$ curl … -d '{"model":"gemma","tools":[…]}'
{"error":"this model's chat template does not use the <tool_call> convention,
          so it cannot answer with tool calls"}
```

A reply carrying functions is not streamed. A tool call is JSON spread over many
tokens, and streaming it would hand the client fragments of syntax; with
functions on the table the reply is buffered and delivered once, parsed. Without
them, streaming is unchanged.

## Reasoning models

A reasoning model writes its working out before it answers, wrapped in `<think>`
tags. That text is not the reply: a client showing replies verbatim shows the
model talking to itself, and — worse for a fleet — a tool call *decided on*
inside the thinking block is not a tool call, so a caller waiting for one sees
prose. kinfer separates the two, as Ollama does.

```
POST /api/chat  {"think": true, …}

{"message": {"role":"assistant", "content":"4",
             "thinking":"The user asks \"What is 2+2?\" and wants just the number…"}}
```

Without `think` the model thinks, and the working out comes back in `thinking`
as it does from Ollama; `content` is the answer and nothing else either way.
`"think": false` keeps the model out of its think block altogether. Dropping the
thinking when the field was absent — the old behaviour — returned an empty reply
with nothing to show for it whenever a reasoning model ran out of time.

A reasoning model can spend an entire token budget thinking and return an empty
answer, so a reply cut short by the budget reports `done_reason: "length"` —
`finish_reason` in the OpenAI dialect — rather than `"stop"`. Told a truncated
reply completed normally, a caller has no way to tell it from a model that had
nothing to say. Streaming separates them as they arrive rather than
buffering the reply. The OpenAI dialect returns it as both `reasoning`, the
field Ollama's own `/v1` writes, and `reasoning_content`, the one DeepSeek
introduced and most clients now look for.

Thinking is split before tool calls are parsed, so a `<tool_call>` the model
merely considered while reasoning is not mistaken for one it made. All three
arrive separately:

```
tool_calls  [{"function":{"name":"get_weather","arguments":{"city":"Berlin"}}}]
content     ""
thinking    "The user is asking about the weather in Berlin. I have a tool available…"
```

## What a reply cost

Every reply carries the counts and timings Ollama reports, under the same names
and in the same integer nanoseconds. Everything that measures a local model
reads these and none of it asks kinfer how it spells them — `ollama run
--verbose`, benchmark scripts, dashboards. Reporting nothing does not read as
fast, it reads as unmeasurable, and leaves a stopwatch as the only way to
compare kinfer against the runtime it stands in for.

```
POST /api/chat  {"stream": false, …}

{"message": …, "done": true, "done_reason": "stop",
 "total_duration": 637676958, "load_duration": 507375,
 "prompt_eval_count": 15,    "prompt_eval_duration": 19558000,
 "eval_count": 101,          "eval_duration": 617452708}
```

`eval_count / eval_duration` is the tokens-per-second figure all of those tools
quote — 163.6 tok/s for the reply above, an M4 running Qwen2.5-0.5B. Streaming
carries them on the final `done` object only, as Ollama does. The OpenAI dialect
reports the same counts as `usage`, and on a stream when
`stream_options: {"include_usage": true}` asks for them.

`prompt_eval_duration` ends at the first sampled token and `eval_duration` starts
there, which is where Ollama draws the line: prefill is in one and generation in
the other, never both. Getting that boundary right needs `llama_synchronize`,
because `llama_decode` queues the forward pass and returns while the GPU is still
working — llama.cpp waits at the first read of the logits, inside the sampler.
Timed without it, the same M4 that prefills at 3,400 tok/s reported a 209-token
prompt processed in 2.9 ms — 71,000 tok/s, which the hardware cannot do — and the
single token sampled from it as having taken 60 ms. One measurement was short by
exactly what the other was long.

Two honest surprises remain in the numbers:

- Under concurrency these are wall clock, not exclusive GPU time. A request
  shares every forward pass with the other busy slots, so its `eval_duration`
  counts their tokens too. That is what the caller waited, and what Ollama
  reports as well.
- A [pooled prefix](#prefix-pool) leaves `prompt_eval_count` unchanged while
  `prompt_eval_duration` collapses. The tokens really were in the prompt; they
  just did not have to be prefilled again.

## Staying up under load

The outage this exists for was a hang, not an error: `kimi-k2.5:cloud` stopped
answering on 2026-07-23 and ~150 agents waited half a day. Everything here is
aimed at the same shape of failure — a server that is not working but does not
say so.

**A bounded queue that refuses.** `-queue` (default 128, sixteen per slot) caps
what may wait. Past it the answer is `503`, not a held connection. Measured with
four slots and a queue of eight, flooded with 64 requests: 12 answered at a 3.2s
median, 52 refused at 0.06s. A `Retry-After` is sent only when the server has
measured its own completion rate — a guessed interval brings every refused
caller back at the same invented moment.

**A clock on every request.** `-max-gen` (default 5m) bounds one reply;
`-max-wait` (same) bounds how long it may queue before being refused rather than
given a slot. There is no token ceiling unless a caller names one — no
`num_predict` means until the model stops or its slot is full, as in Ollama — so
the clock is what bounds a runaway reply. Without it a single request could hold
a slot for over 400 seconds — eight of those is the whole server. A reply cut off by the clock returns its text with
`done_reason: "timeout"`, because `"stop"` would claim the model finished and
`"length"` would claim it hit the budget it was given.

**A queued request survives a backend failure.** llama.cpp reports a fatal
decode for the whole context, so kinfer retires the model — and used to fail
every request it held. Measured: 32 concurrent requests against a dying backend,
0 answered before, 28 after. The four that had already streamed text are still
failed, because re-running them would repeat it.

## Seeing what it is doing

```
curl localhost:11500/metrics
```

```
kinfer_queue_depth 0
kinfer_slots_busy 0
kinfer_slots_total 2
kinfer_prompt_tokens_total 9
kinfer_eval_tokens_total 16
kinfer_model_loaded{model="…/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"} 1
kinfer_requests_total{outcome="ok"} 2
kinfer_requests_total{outcome="refused"} 0
kinfer_requests_total{outcome="failed"} 0
```

Prometheus text format. The three that matter together: a deep queue beside idle
slots is a different problem from slots that are always full, and `refused` is
how you know which.

`KINFER_DEBUG_SCHED=1` adds a line a second on what the scheduler did, and where
each step's time went:

```
sched:   87 steps/s   mean batch   1.0 tokens   mean active slots  1.0   per step 11.50 ms: decode 1.32, gpu wait 10.06, sample 0.12, scheduler 0.00
```

`decode` is queueing the forward pass, `gpu wait` is the GPU finishing it,
`sample` is choosing the next tokens, and `scheduler` is whatever is left of the
wall clock. That line is how the single-request gap to Ollama was found to live
in the llama.cpp build rather than in kinfer (see *Against Ollama* above). The
timers run only when the variable is set.

## Sizing a model before loading it

```
kinfer plan Qwen3.8-Flash-Next-UD-Q2_K_XL -ctx 16384
```

```
  weights           73.5 GB
  GPU budget        77.8 GB   (this process's share, not the machine's RAM)
  left for cache     4.3 GB

  -slots   total tokens   largest -ctx that fits
  1        32768          32768
  2        32768          16384
  4        32768          8192
  …
  For -ctx 16384: kinfer serve -ctx 16384 -slots 2
```

`-ctx` is per conversation and gets multiplied by the slot count, which is the
part that surprises people — raising `-slots` from 2 to 4 doubles the cache. The
figure kinfer prints at load time is the measurement; `plan` is an estimate, and
says so, because a hybrid architecture holds per-sequence state that does not
scale with context and costs more than the arithmetic predicts at high slot
counts.

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
