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
-rwxr-xr-x  23M   # llama.cpp is inside — nothing else to install
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

## Install

On an Apple Silicon Mac:

```bash
curl -fsSL https://raw.githubusercontent.com/LocalKinAI/kinfer/main/scripts/install.sh | sh
```

It puts the latest release in `~/.local/bin`, checked against its sha256: one
file with llama.cpp inside, the build that reads experts from the SSD at speed.
It is not signed. Downloaded this way macOS runs it, and a firewall that is on
asks once before `serve` or `worker` takes connections from another machine.
Elsewhere, or to build it yourself, see [Build](#build).

Or with Homebrew, which upgrades it with everything else (`brew upgrade`):

```bash
brew tap localkinai/tap
brew trust --formula localkinai/tap/kinfer
brew install localkinai/tap/kinfer
```

With other taps not trusted yet, Homebrew 7 can end the install with an error
about one of them, after kinfer is in place; `kinfer version` says whether it is,
and `HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK=1` before `brew install` skips the
check that raised it.

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
kinfer worker -addr <ip:port> # lend this machine's GPU to a serve -peers on another
kinfer version               # this build, and the llama.cpp inside it
```

What each does, and how serve sizes itself: [docs/serving.md](docs/serving.md).

## Models bigger than the machine

A model the GPU cannot hold needs no flags. A mixture of experts uses a few of
its experts for each token — 10 of Flash-Next's 512 — so kinfer reads the
model's header, keeps on the GPU what every token uses and a cache of the
experts in use, and leaves the other experts in host memory, read from the SSD
as the cache misses them. Given another machine's GPU (`-peers`), it gives that
machine only the layers that do not fit here.

Qwen3.8-Flash-Next IQ3_S — 51 GiB of weights, and a 27 GiB table read a few
rows a token — with no flags; a 7,753-token prompt and three 200-token replies:

| machine | GPU memory | where the model went | prompt | decode |
|---|---:|---|---:|---:|
| Mac Studio M3 Ultra, 96 GB | 88 GiB | all of it on the GPU | 748 tok/s | 43 tok/s |
| MacBook Pro M5 Max, 48 GB | 40 GiB | 28 of 48 layers' experts in host memory and on the SSD, a 9 GiB cache | 706-726 tok/s | 27-29 tok/s |
| the MacBook Pro, the Mac Studio as its peer over USB 2.0 | 40 + 88 GiB | layers 0-17 on the Mac Studio | 258 tok/s | 23 tok/s |

`kinfer plan <model>` shows where a model would go, without loading it. Reading
experts from the SSD at that speed takes kinfer's build of llama.cpp
(`scripts/build-libs.sh`, below); with the release libraries the MacBook Pro
decoded at 9 tok/s. Over that USB 2.0 cable the laptop's own SSD beat the other
machine's GPU; a Thunderbolt 5 link is the next measurement.

Against Ollama on the same MacBook Pro, from the same files, the same requests
back to back, and the same reply from all three:

| | prompt | decode |
|---|---:|---:|
| Ollama 0.40.13, as it sets itself up | 17 tok/s | 6.9-8.4 tok/s |
| Ollama's `llama-server` with kinfer's hand-tuned settings (`--n-cpu-moe 24 --moe-cache-mib 8192 -ub 2048`) | 81 tok/s | 6.3-8.2 tok/s |
| kinfer, no flags | 716 tok/s | 27.9-28.4 tok/s |

The settings are not the difference; kinfer's build of llama.cpp is. It reads
the experts the cache misses with a pool of threads instead of a page fault at a
time, reads the next ones ahead, and streams a prompt's passes.

Part of the way there is Strata's, whose own engine runs a mixture of experts
from the SSD on a Mac: a GPU cache of the experts in use with the rest read from
the file, misses read by a pool of threads, the next layer's experts guessed
from its router and read ahead, and a profile of the experts generation uses
filling the cache at load. kinfer does those in llama.cpp, whose MoE cache it
patches; the rest is its own. Step by step on the MacBook Pro, the 7,753-token
prompt (each step measured as it was made, and the laptop wanders by a quarter
from run to run):

| step | whose | prompt |
|---|---|---:|
| llama.cpp's release, 24 layers' experts on the host, an 8 GiB cache | | 62 tok/s |
| misses read in parallel, the next layer read ahead, a profile at load | Strata's | 159 tok/s |
| everything mapped read with `pread` by a pool of threads | Strata's way, kinfer's code | 306 tok/s |
| a prompt's passes streamed through host memory, cached experts copied on the host | kinfer | 391 tok/s |
| passes of 4096 | kinfer | 445 tok/s |
| the rows of the per-layer table read in parallel | kinfer | 701 tok/s |
| placed by itself, no flags | kinfer | 706-726 tok/s |

Decode is mostly Strata's way: 9 tok/s with the release, 23 with its three
parts, 31 once misses were read with `pread`. Reading a prompt is mostly
kinfer's, and so is choosing where the model goes, on one machine or across
several. How it works, and every measurement behind it:
[docs/big-models.md](docs/big-models.md).

## What else it does

- **Serving** ([docs/serving.md](docs/serving.md)): `run` keeps a model warm in
  a daemon; `serve` sizes the context, the slots and the micro-batch to the model
  and the memory left; a prompt too long for its slot gets a longer one;
  `install` keeps the server up; every reply carries its timings; `/metrics`; a
  bounded queue that refuses rather than hangs; `fit`, `plan` and `can-run` size
  a model before it is loaded or downloaded.
- **Agents** ([docs/agents.md](docs/agents.md)): a prefix pool, so a repeated
  system prompt is read once; chat templates read from the model; tool calls in
  the Ollama and OpenAI dialects; a reasoning model's thinking kept apart from
  its answer.
- **Performance** ([docs/performance.md](docs/performance.md)): continuous
  batching, and measurements against Ollama on the same files.
- **Design** ([docs/design.md](docs/design.md)): how the code is laid out, and
  why llama.cpp is bound directly.

## Build

```bash
# Fetch llama.cpp's libraries once, into the package that embeds them
# (8 files, 12.4 MB; they are not committed — every upgrade is a new set).
scripts/fetch-libs.sh

# The libraries end up inside the binary (24 MB in all)
go build -o kinfer ./cmd/kinfer
```

`scripts/fetch-libs.sh` takes the llama.cpp build kinfer is checked against
(b11998) from the official release, copies the eight `.0.dylib` files into
`internal/nativelib/libs/darwin_arm64_b11998/`, and removes any other version
there. Run it with `-f` to fetch again, or with a build number to try another
release — after diffing its `llama.h`; see `internal/nativelib/libs/README.md`.

`scripts/build-libs.sh` instead builds b11998 from source with kinfer's patch
(`patches/llama.cpp/b11998/`; about two minutes, cmake and Xcode's command line
tools) into `darwin_arm64_b11998k3/`, replacing the release set. It matters
for a model whose experts or tables come from the SSD — see
[When the experts do not fit in memory either](docs/big-models.md#when-the-experts-do-not-fit-in-memory-either);
with the release set `-host-experts` and `-moe-cache` work as before and there is
no profile — and for a model split across machines, whose GPUs it keeps at speed
between tokens; with the release set a split decodes at about 25 tok/s where the
patched one reaches 42.

`scripts/release.sh v0.1.0` builds what a release ships: those patched
libraries, the version stamped in, packed with its sha256 into `dist/`.
`scripts/brew-formula.sh v0.1.0 > ../homebrew-tap/Formula/kinfer.rb` writes the
formula for that release into the tap,
[LocalKinAI/homebrew-tap](https://github.com/LocalKinAI/homebrew-tap).

## Roadmap

- [x] **Phase 0** — feasibility: pure-Go inference, Metal, single binary
- [x] **Inference core** — chat templates, sampling, stop detection, native bindings
- [x] **Phase 1** — `fit` / `pull` / `list` / `rm` / `run` / `serve`
- [x] **DeepSeek-V4.1** — its own prompt renderer and DSML tool calls, checked
      against DeepSeek's encoder; sizing that leaves lazily read tables on disk
- [x] **Two machines, one model** — `kinfer worker` and `serve -peers`, through
      llama.cpp's RPC backend; measured over USB 2.0 between the box and a MacBook Pro
- [x] **Experts in host memory** — `-host-experts`, `-moe-cache`; llama.cpp's MoE
      cache measured on Metal, where it costs more than it saves for a model that fits
- [x] **Experts from the SSD, at speed** — a patched MoE cache: parallel reads,
      next-layer read-ahead, a profile that warms it; 9 → 23 tok/s on a MacBook Pro
- [x] **Prompts from the SSD, at speed** — `pread` for everything mapped, large
      batches streamed through host memory, per-layer tables read ahead; a prompt
      62 → 705 tok/s and decode 23 → 31 on the same laptop
- [x] **Placed by itself** — a mixture of experts the GPU cannot hold leaves as
      many layers' experts in host memory as it must, with a cache sized for
      them: no flags on the MacBook Pro, as fast as the hand-tuned ones
- [x] **Split by itself** — with `-peers`, the peers take only the layers that do
      not fit here, and what does not fit anywhere stays in this machine's host memory
- [ ] **Bigger than one machine, at speed** — a Thunderbolt 5 link, and choosing
      between a peer's GPU and this machine's SSD by what each costs a token
- [ ] **Phase 2** — the reason this exists: health probing, automatic failover,
      circuit breaking, per-agent model routing
- [ ] **Phase 3** — apply `fit`'s backend choice automatically, honest benchmarks
      vs Ollama

## License

Apache 2.0
