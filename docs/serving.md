# Serving

How `kinfer run` and `kinfer serve` load and size a model, keep it up, and say what it costs — part of [kinfer](../README.md).

## `kinfer run` keeps the model warm

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

## `serve` sizes itself, like Ollama does

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

## A conversation longer than its slot loses its oldest turns

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

A model's files are not always what it holds in memory. DeepSeek-V4.1,
Qwen3.8-Flash-Next and Gemma 4 carry a table llama.cpp reads from disk a few rows
a token instead of loading — V4.1's n-gram "Engram" is 111 GB of a 249 GB set.
`plan`, `fit`, `can-run` and `/api/ps` read the GGUF headers (ranged, for a repo
on Hugging Face) and count only what stays resident:

```
$ kinfer plan Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S          on the box, 77.8 GiB budget
  weights           51.1 GB
  read from disk    26.8 GB   (a table read a few rows a token: it stays on the SSD, not in memory)
  left for cache    26.7 GB
```

Counted as weights, that file was 117.8 MB too large for the box, and `plan` said
so. `serve` never believed it: it sizes from what the load measures, llama.cpp
already left the table on disk, and the box's server has run this file since
the evening it was downloaded — 42 loads from October 11 to 28, 51.0 GiB of model
each time. Only the estimate was wrong. On b11998 it takes 50.7 GiB, leaves
27 GiB for 8 slots + 4 prefix x 32768, and decodes at 44.5 tok/s.

## `-slots` without `-ctx`: each model's context sized the way Ollama sizes one

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

## The micro-batch, sized the way Ollama sizes it

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

## A prompt too long for its slot gets a longer one

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

## `kinfer install` keeps the server up

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

## `kinfer fit`

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
- A [pooled prefix](agents.md#prefix-pool) leaves `prompt_eval_count` unchanged while
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
kinfer plan -ctx 16384 Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S
```

```
  weights           51.1 GB
  read from disk    26.8 GB   (a table read a few rows a token: it stays on the SSD, not in memory)
  GPU budget        77.8 GB   (this process's share, not the machine's RAM)
  left for cache    26.7 GB

  -slots   total tokens   largest -ctx that fits
  1        262144         262144  (capped at what it was trained for)
  2        524288         262144  (capped at what it was trained for)
  4        1048576        262144
  8        524288         65536
  …
  Unasked, serve picks: 8 slots + 4 prefix x 32768 tokens, cache about 10.3 GB
  For -ctx 16384: kinfer serve -ctx 16384 -slots 8
```

The UD-Q2_K_XL this example used before was 73.5 GB of weights by the same
arithmetic, with 4.3 GB left for cache and room for two slots of 16384.

`-ctx` is per conversation and gets multiplied by the slot count, which is the
part that surprises people — raising `-slots` from 2 to 4 doubles the cache. The
figure kinfer prints at load time is the measurement; `plan` is an estimate, and
says so, because a hybrid architecture holds per-sequence state that does not
scale with context and costs more than the arithmetic predicts at high slot
counts.
