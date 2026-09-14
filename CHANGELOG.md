# Changelog

## Unreleased

Everything below was built and measured on one 96 GB Mac Studio against a
73.4 GiB model, which is why the numbers are specific.

### Staying up under load

- **`kinfer install` / `kinfer uninstall`.** A per-user launchd job (no sudo)
  that starts `serve` at login and restarts it after any exit — `kill -9` came
  back in 10s. The serve flags are recorded verbatim after `serve`'s own parser
  has accepted them, so a mistyped flag is refused now rather than retried
  forever by launchd. Reinstalling replaces the job. macOS only.
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
