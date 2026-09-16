# Changelog

## Unreleased

Everything below was built and measured on one 96 GB Mac Studio against a
73.4 GiB model, which is why the numbers are specific.

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
