# Changelog

## Unreleased

Everything below was built and measured on one 96 GB Mac Studio against a
73.4 GiB model, which is why the numbers are specific.

### A long prompt gets a longer slot

`-ctx 16384 -slots 4` on the box refused a 44,555-token prompt with a 413 while
the cache held 65,536 tokens in all: llama.cpp splits a context evenly between
its sequences, so the longest prompt a server took was a quarter of its cache.
Giving every slot the whole 65,536 was measured and does not fit: 8.2 GiB of
cache where the weights leave 4.4 under macOS's default GPU budget. With the
budget raised to 86 GiB it loaded, and the first decode still died with
`kIOGPUCommandBufferCallbackErrorOutOfMemory`, because another process on the
machine was holding 20 GB at the time.

So the cache keeps its total and changes its shape. The layout the server loaded
with is home. A prompt that will not run whole in a home slot waits for the slots
to drain, and the context is rebuilt as the most slots that still hold it: 3 x
21760, 2 x 32768 or 1 x 65536 on the box. Once nothing running needs the longer
slots, home is rebuilt. The request at the head of the queue decides and nothing
behind it is admitted until it has its slot, so short requests arriving after a
long one cannot starve it. While it runs, a short request that finds a free
longer slot takes it.

Measured on the box with Flash-Next. A 44,555-token prompt queued behind three
short ones waited 10.9 s for them, got 1 x 65536 in 224 ms, and answered whole:
95 s to read it, then 24 tok/s. The two short requests queued behind it got 4 x
16384 back in 223 ms. A 27,174-token prompt got 2 x 32768 in 246 ms, with a short
request riding in the other slot. Four short requests at once in the home layout
still run at 14 tok/s each, as before. A layout that would leave less memory than
home did is refused before any batch runs in it, and not tried again. The prefix
pool lives only in home. `-flex=false` keeps the layout fixed.

Also:

- **`/api/ps` named a split model by its first file.** It reported
  `…-00001-of-00003`, which `/api/tags` does not list and a request does not
  resolve, so an unload by the name `ps` gave found nothing. It also sized the
  model at the first file's 11 MB of 73.5 GB. It now names and sizes the model
  the way `/api/tags` does.
- **`kinfer_slots_total` in `/metrics` reports the current layout**, not the one
  the model loaded with.
- **The context belongs to the scheduler.** It frees and rebuilds the context on
  a layout change, and hands whichever one it holds last to the engine to free.

### Four places Ollama answered differently, and a caller paid for it

A night of deep-study runs on the box — a local worker writing citation cards
from source texts through `/api/chat` — hit four differences from Ollama in a
row. Each is now what Ollama does, or, for `-prefix`, what it looks like it
does.

- **Thinking comes back unless it is turned off.** With no `think` field the
  model still thought — absent is not off — but the thinking was split off and
  dropped. A reply cut short by the clock then came back with nothing at all:
  measured on Flash-Next, 3,143 tokens generated, 0 characters of thinking and
  0 of content. Ollama 0.34.3, asked the same way, returns the thinking in
  `message.thinking` beside a clean `content` (measured, streamed and not). Now
  so does kinfer. `think: false` still keeps the model out of its think block.
  The OpenAI dialect returns it too, as both `reasoning` (Ollama's `/v1` name)
  and `reasoning_content` (DeepSeek's).
- **No token ceiling unless the caller names one.** The default was 512, which
  contradicted the scheduler's own premise that a caller who names no budget
  gets the rest of its slot, bounded in time by `-max-gen`. Card-writing replies
  ran past 512 tokens and came back cut off mid-JSON, `done_reason: "length"`,
  which parsed as zero cards. Ollama's default is `num_predict: -1`. `kinfer run
  -n` defaults to 0 as well: until the model stops.
- **A model can be unloaded with one request.** Freeing the box's 73.4 GiB for a
  second model meant `launchctl bootout` — the whole service down — and a
  bootstrap afterwards. Ollama does it with the request `ollama stop` sends:
  `POST /api/generate {"model": m, "keep_alive": 0}`. kinfer now answers that,
  a prompt-less `/api/generate` that loads a model ahead of use, and both as
  `/api/chat` with no messages, in Ollama's exact shape (`done_reason` "load" or
  "unload"). An unload answers once the memory is free; a request still
  generating finishes first, and nothing loads a second copy meanwhile.
  `keep_alive` on an ordinary request now governs how long the model stays
  after it — seconds, `"5m"`, or negative for forever, as Ollama reads it. Its
  zero means unload; the `-keepalive` flag's zero still means never. Each keeps
  its own convention, and the code says so where they meet.
- **`-prefix` takes `auto`, `off` or a number, and `auto` fits.** It was an int
  where 0 meant "sized with the rest". Typed as `-ctx 16384 -slots 4 -prefix 0`
  to turn the pool off, it sized four entries of 16384 instead: 0.0 GiB left
  after loading, and the first request failed inside llama.cpp (code -3). Two
  things were wrong. The spelling: `0` now means no pool, `-1` still does for
  installs that recorded it, and unset is `auto`. And `auto` did no arithmetic
  when `-slots` and `-ctx` were both given — it returned four entries before any
  check, and the measured check after loading skipped it because the slots and
  context were not its to change. Now an automatic pool around fixed
  conversations gets only what they leave: cells of its own if they fit, the
  slots' cells if only that fits, none otherwise; and the measured check sheds
  it when the estimate was short. An explicit count is still never
  second-guessed.

### Fixed — coding agents run against `serve` with no flags

Codex, pointed at the box's `serve` with Qwen3.8-Flash-Next loaded, got
`413: prompt is 7428 tokens but each slot holds 4096 (the context is split
across 4 slots; lower -slots or raise -ctx)` on its first request — naming two
flags nobody in a terminal tab can set. Ollama ran the same session. Two
things were different, and both are now what Ollama does.

- **The context an agent gets.** The floor below which slots are given up
  instead was 8192, so 4.3 GiB left after a 73.4 GiB model sized 4 slots of
  8192, and the measured cost halved that to 4096. The floor is now 32768 (or
  the trained context, if less) — Codex's opening prompt is 7,428 tokens and
  Claude Code's about 25k, before any tool has run — and when the measured cost
  is too high the rebuild gives up slots, then the prefix pool's own cells,
  then pool entries, and halves the context only after that. `-prefix` now
  defaults to 0, sized with the rest; its old default of 4 counted as an
  operator's choice and kept the pool out of that order.
- **A pool that shares the slots' cells.** Its entries live in whatever the
  slots leave free; a batch that finds the cache full takes them back and runs
  again — llama.cpp finds room for every micro-batch before running any, so the
  failed attempt changed nothing. An agent's next turn continues what the slot
  already holds, so it costs no extra room. On the box: 1 slot of 32768 with
  its turns reused, 2.6 GiB free, where the choice was otherwise 16384 or no
  pool.
- **A conversation longer than its slot loses its oldest turns** instead of
  being refused. The system prompt, the latest request (the newest user turn
  not riding along with a tool result — Claude Code sends reminders that way)
  and the newest turn stay; a tool call goes with its results and a plain reply
  with its question; room is left to reply in, up to an eighth of the slot.
  Turns are dropped in chunks at points fixed by the turns before them, so the
  kept part is the same for several turns and a hybrid model's whole-entry
  pool keeps matching it. Only a prompt whose system prompt, request and newest
  turn do not fit by themselves is still a 413. Measured with Qwen2.5-0.5B at
  `-ctx 4096 -slots 1`: 28 messages, the oldest 16 dropped, 3,514 tokens, and
  the answer read from the tool result that was kept.

- **The five-minute reply limit no longer counts prefill.** It started at
  admission, so on a machine prefilling 50 tokens a second Claude Code's
  30k-token opening prompt was still being read when time ran out: the reply
  came back empty with `stop_reason: max_tokens`, Claude Code sent the same
  prompt again, and the same thing happened again. The limit is for a reply
  that never ends; it now starts at the first sampled token.

On the box, after this: Codex (`codex exec`, a task that reads a file) ran end
to end — tool call, result, answer — in 33 s with the weights in the page
cache and 68 s with them read from disk, where it had been refused. Claude Code answered the same task through `/v1/messages`; its
second turn adopted the first from the shared pool (3.5 s to the first byte),
and the pool gave its cells back each time a different prompt needed them,
with no request failing.

The slow numbers above are the box with Qwen3.8-Flash-Next, and not the box
itself. A 73.4 GiB model leaves that machine 14% of its memory, and for a
while after such a model is loaded or released everything on it runs slow —
ornith-1.5:35b under Ollama 0.34.0, same GGUF, went 79 → 34 → 18 tok/s over
three short requests and back to 90 after 30 s idle, exactly as under kinfer.
Once memory had settled, the same model under kinfer's own sizing (8 slots of
65536) prefilled a 6,360-token prompt at 1,637 tok/s and generated at 83–88
tok/s before and after it, with no drop. Codex did the task above in 7.3 s;
Claude Code in 37 s, 27 s of it the first read of its 30k-token prompt, with
0.6 s to the first byte of its second turn.


### Fixed — prefix reuse on models with recurrent layers

Measured on a 16 GB MacBook with ornith-1.5:9b and confirmed on ornith-1.5-35b;
both are hybrids, attention plus recurrent layers.

- **What was wrong.** The pool shares a cached prompt's first n tokens with a
  new request. Attention cells can be shared up to any position, but a
  recurrent layer's state is one value for the whole sequence, so the request
  got a state that had already read the rest of the pooled prompt. llama.cpp
  said so on every such reuse (`find_slot: non-consecutive token position 257
  after 257`), and the output showed it: the same prompt sent twice came back
  the second time with its tool call written as `<invoke name="shell">…` prose
  instead of a `<tool_call>` — through every dialect, `/v1/chat/completions`
  included. The warning was in the box's log for other clients' traffic too.
  A test for content crossing between conversations — one told a codeword, a
  second with the same system prompt asked for it — came back clean: the stale
  state perturbed output, it did not carry text across.
- **What it does now.** For a model llama.cpp reports as recurrent or hybrid, a
  pooled prompt is adopted only whole, by a request that continues past its
  end. To keep that worth having, such prompts are pooled at the two places
  later requests repeat whole — found by rendering the conversation a second
  way and seeing where the two part: the end of the system prompt, for another
  conversation of the same agent, and the end of the conversation so far, for
  its next turn. Both are taken back to just after a control token, because an
  ordinary token at the edge tokenizes with what follows it: the `\n` that
  ended `<|im_start|>assistant\n` came back in the next turn merged into the
  `\n\n` before a `<tool_call>`, one token short of whole. Prefill ends a batch
  at each so the state is pooled at exactly that position, and the system
  prompt is kept under the conversations that extend it and evicted after
  them. Pure attention models are unchanged.

| ornith-1.5:9b, ~1,000-token agent prompt | reused | prefill |
|---|---|---|
| first turn | 0% | 4.95s |
| second turn | 94% | 0.42s |
| third turn | 94% | 0.43s |
| another conversation, same system prompt | 99% | 0.26s |
| back to the first conversation | 97% | 0.27s |
| the same prompt again (tool call intact 3/3) | 99% | — |
| control: a different system prompt | 0% | 6.53s |

Whole entries pooled at the prompt's end — the first version of this fix —
kept output right and reused 0% of the second turn, because the pooled prompt
ended in an opening of the reply the next turn renders differently. That is
what the checkpoints are for.

### The dialects coding agents speak

Measured on a 16 GB MacBook against ornith-1.5:9b — a 5.6 GB GGUF symlinked
out of an Ollama blob store — not on the Studio above.

- **`POST /v1/messages`, Anthropic's Messages API**: what Claude Code speaks to
  a server named in `ANTHROPIC_BASE_URL`. **`POST /v1/responses`, OpenAI's
  Responses API**: the only thing Codex speaks to a custom provider since 0.154
  refused `wire_api = "chat"`. Claude Code finishing a turn that ran one Bash
  call: 62s. Codex listing a directory through `exec_command`: 49s with a cold
  prefix cache — its opening prompt is 6,350 tokens — and 15s warm.
- **Written against captured traffic, not the reference.** Both agents were run
  through a recording proxy against Ollama 0.34, which each accepts unmodified,
  and the handlers follow what was on the wire. It differs from the docs in ways
  that would each have been a 400 or a hang: Claude Code posts to
  `/v1/messages?beta=true`, puts turns with role `system` inside `messages`, and
  sends `thinking`, `output_config`, `context_management`, `metadata` and
  `cache_control` on every turn; Codex sends a 17K-character `instructions`, a
  `developer` message, `namespace` and hosted `web_search` tools beside its
  function tools, and history in the order function_call → reasoning →
  function_call_output. The streams follow Ollama's event order too: no `ping`
  and no thinking signature in one dialect, reasoning summaries without
  `summary_part` events in the other.
- **One split for every dialect.** Thinking, prose and calls are separated by
  the same `ThinkSplitter` and `CallSplitter` pass the OpenAI dialect streams
  with, so no dialect can disagree with another about where a think block ends
  or a call begins.
- **Failures in each dialect's own envelope.** 413 `request_too_large`, 503
  `overloaded_error` with `Retry-After`, 404 `not_found_error`. A streamed
  request that fails before its first token still gets a status; one that fails
  after gets `event: error` or `response.failed` rather than a stream that just
  stops. A reply cut off by a limit ends in `stop_reason: max_tokens` or
  `response.incomplete`, never in a finished turn.
- **Not done.** No `count_tokens` (Claude Code estimates instead, as it does
  against Ollama). Namespace, custom and hosted tools are dropped: offered to a
  local model they can only produce calls nothing here answers.
  `output_config.format` is stated to the model, not enforced with a grammar —
  Claude Code uses it for a session title, so a model that ignores it costs a
  title. Images arrive as a note saying they were omitted.

### Staying up under load

- **`serve` sizes itself.** With no `-ctx`/`-slots`, the context comes from
  what the model was trained for and the memory its weights left, spending at
  most half of that on the cache; when that is too small for a real prompt,
  slots are shed before context is. The hybrid geometry is read from the GGUF
  header (`full_attention_interval`, `ssm.*`), so the 35B's cache is estimated
  at its measured size rather than twice it — the error that sized a server to
  32768 tokens on a machine with room for 65536, and refused a 45k-token
  prompt Ollama accepted. After building the context the cost is measured and
  the context halved if under 2 GiB remains. `kinfer plan` prints the same
  decision; explicit flags are never overridden.
- **`kinfer install` / `kinfer uninstall`.** A per-user launchd job (no sudo)
  that starts `serve` at login and restarts it after any exit — `kill -9` came
  back in 10s. The serve flags are recorded verbatim after `serve`'s own parser
  has accepted them, so a mistyped flag is refused now rather than retried
  forever by launchd. Reinstalling replaces the job — and waits for the old one
  to actually stop first: `launchctl bootout` returns while the server is still
  in its shutdown grace, and a bootstrap in that window fails with an I/O error
  and leaves nothing running, which is how the first reinstall of a 20 GiB
  model went. macOS only.
- **`/metrics` counts 4xx as `rejected`.** A client sending prompts larger
  than a slot got 413 on every request while the counters read all zeros; the
  operator's question was "why does it not answer" and there was no number to
  point at.
- **Bounded queue with backpressure.** `-queue` (default 128) caps what may wait;
  past it the answer is `503` in about 60ms rather than a held connection. A
  `Retry-After` is sent only when the server has measured its own completion
  rate, never guessed.
- **A clock on every request.** `-max-gen` bounds one reply and `-max-wait` how
  long it may queue (both 5m; `0` removes either). A caller could previously pass
  `num_predict` meaning "keep going" and hold a slot for over 400 seconds. A
  reply cut off by the clock returns its text with `done_reason: "timeout"`.
- **Queued requests survive a backend failure.** A fatal decode retires the
  model; requests that had never started are now re-run on its replacement
  instead of failing with it. Measured: 32 concurrent requests against a dying
  backend, 0 answered before, 28 after.

### Seeing and sizing

- **`/metrics`** in Prometheus format: queue depth, busy and total slots, prompt
  and eval tokens, request outcomes, forwarding and fallback counts.
- **`kinfer plan <model>`** sizes `-slots` and `-ctx` against the GPU's real
  budget before loading anything, reading the shape from the GGUF header.
- **A memory line at load time**, with the arithmetic spelled out:
  `model 73.4 GiB + context 1.7 GiB -> 2.7 GiB free of 77.8 GiB (this process
  only)`. The parenthesis is load-bearing — see below.
- **Timings on every reply** (`eval_count`, `eval_duration` and the rest, as
  integer nanoseconds) in both dialects, so kinfer can be measured the same way
  Ollama is.

### Fixed

- A mid-request model reload no longer vanishes from `load_duration`.
- `llama_synchronize` on steps that produce logits: without it, splitting prefill
  from eval charged the GPU wait to the wrong side and reported 71,000 tok/s
  prompt processing.
- A request that expired *in the queue* is refused rather than returned as an
  empty reply with a finish reason — it never ran, and reporting it as a
  completed reply says the model had nothing to say.
- `-max-gen 0` removes the limit. `-1` cannot be typed: Go parses a duration and
  it has no unit, so the flag errored and the process exited.

### Tried and withdrawn

- **Routing cloud models through kinfer.** For one day kinfer recognised
  Ollama's cloud entries, forwarded them, and could fall back to a local model
  when the upstream said *not now*. It was removed after its first real outage.
  When the weekly cloud quota ran out, every request fell back to a local model
  whose per-slot context could not hold most of the fleet's prompts (37% of
  turns came back empty), the substitution was invisible to the quota sentinel
  because the reply still carried a `message`, and cloud names absent from the
  local machine's manifests were answered 404 where Ollama would have resolved
  them. None of that is a bug in the runtime; it is what happens when a local
  inference server is put in front of traffic it does not serve. kinfer serves
  local models. Which endpoint a caller uses, and when, is the caller's policy.

### Learned, and written into the code

- **`free` from ggml-metal is per process.** It is
  `recommendedMaxWorkingSetSize - currentAllocatedSize`, and
  `currentAllocatedSize` belongs to the calling process. With 73.4 GiB resident
  in one kinfer, a second one loading a 0.5B reports 76.9 of 77.8 GiB free. Every
  out-of-memory failure seen here happened when the sum across processes crossed
  the budget, which no process can see — measured at 78.4 GiB against a 77.8 GiB
  limit while each reported comfortable headroom. See `internal/engine/budget.go`.
- **Batching amortises much less on a mixture of experts.** Doubling the tokens
  in a decode step costs 1.5-1.8x, not 1.0x, because each token routes to its own
  experts. Which is why **speculative decoding was measured and not built**: it
  would buy +47% for a single stream on ornith-1.5-35b and nothing at all at
  eight, and below ~90% acceptance it costs. See `internal/engine/scheduler.go`.
- **KV cost does not scale with tokens alone.** At 32768 tokens either way, 2
  sequences cost 1.7 GiB and 32 cost 4.3. Sizing `-slots` is about the
  per-sequence term, not `-ctx`.
