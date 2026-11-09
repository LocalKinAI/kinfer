# kinfer

[![PyPI](https://img.shields.io/pypi/v/kinfer-ai)](https://pypi.org/project/kinfer-ai/)
[![Downloads](https://img.shields.io/pepy/dt/kinfer-ai)](https://pepy.tech/project/kinfer-ai)
[![Downloads a month](https://img.shields.io/pypi/dm/kinfer-ai)](https://pypistats.org/packages/kinfer-ai)
[![GitHub downloads](https://img.shields.io/github/downloads/LocalKinAI/kinfer/total)](https://github.com/LocalKinAI/kinfer/releases)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**A single-file local inference runtime for agent fleets.** No Docker, no
service to install, no separate llama.cpp — one binary, one copy, it runs.

```bash
$ kinfer pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
$ kinfer list
NAME                                                 SIZE  MODIFIED
qwen2.5-0.5b-instruct-q4_k_m                     468.6 MB  just now

$ kinfer run qwen2.5-0.5b-instruct-q4_k_m "explain merkle trees in one sentence"
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

Or with pip, into whatever Python environment you work in (`kinfer` on PyPI is
another project, so the package is `kinfer-ai`; the command is still `kinfer`):

```bash
pip install kinfer-ai
```

The wheel holds the same release binary, for macOS 26 on Apple Silicon or the
Linux and Windows builds below, and `pip install -U kinfer-ai` upgrades it.

### Linux and Windows: a preview, on the CPU

kinfer is developed and tested on the Mac. The Linux and Windows builds run on
the CPU only for now: x86-64 and arm64 pass the test suite and serve a small
model on real machines in CI, and nobody here runs them daily. Reports are
welcome, and so is help with GPU support.

On Linux, the same script as on a Mac (glibc 2.34 or newer on x86-64, 2.38 or
newer on arm64; not Alpine):

```bash
curl -fsSL https://raw.githubusercontent.com/LocalKinAI/kinfer/main/scripts/install.sh | sh
```

On Windows, in PowerShell (it needs the Visual C++ runtime, and warns if that is
missing):

```powershell
irm https://raw.githubusercontent.com/LocalKinAI/kinfer/main/scripts/install.ps1 | iex
```

MLX models, reading experts from the SSD, splitting a model across machines and
`kinfer install` are Mac only.

## Commands

```bash
kinfer fit                   # what this machine can actually run
kinfer pull <repo>[:quant]   # Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M, or an MLX repo: mlx-community/Qwen3.8-27B-4bit
kinfer list                  # what is installed
kinfer rm <model>            # delete one
kinfer run <model> [prompt]  # chat — no prompt means interactive
kinfer ps                    # what the daemon is holding right now
kinfer stop                  # stop that daemon
kinfer serve [-addr :11500]  # serve over HTTP; context and slots sized to the model
kinfer install [serve flags] # keep serve running: at login and after a crash (macOS)
kinfer uninstall             # stop it and remove the service
kinfer worker                # lend this machine's GPU over a Thunderbolt cable
kinfer worker -on <user>     # start one on the Mac at the other end of the cable
kinfer version               # this build, and the llama.cpp inside it
```

`kinfer pull` fetches a file in pieces over eight connections at once
(`-connections n`), and keeps what has arrived when it stops — Ctrl-C, a dropped
network, a full disk: the same pull again fetches only the rest, and until then
`kinfer list` shows it unfinished. A split model arrives with every piece, and
each file takes its name only once it matches the sha256 Hugging Face publishes
for it.

What each does, and how serve sizes itself: [docs/serving.md](docs/serving.md).

## MLX models

A model can also be a directory of safetensors as Hugging Face publishes it,
and kinfer serves it on Apple's MLX: Qwen3.5 and Qwen3.8, at bf16 or MLX 4-bit, such as
mlx-community's conversions, or NVFP4. `kinfer pull mlx-community/Qwen3.8-27B-4bit` fetches
one into the store; from there it is listed, run and served like a GGUF, through
the same four dialects. Ollama's own MLX models (`qwen3.8:27b-mlx`) are listed
and served where they lie, as its GGUFs are. The engine is Go over mlx-c, with no cgo and no Python,
and checked against mlx-lm layer by layer. MLX's libraries (78 MB) are fetched
once, the first time an MLX model loads, and need macOS 26.2.

Where it stands against llama.cpp on the M3 Ultra: a small model at many replies
at once is MLX's (Qwen3.5-0.8B at 64: 2,833 tok/s against 1,031, in half the
memory); Qwen3.8-27B decodes alone at 36 tok/s against 31, but llama.cpp's MTP
drafting reaches 51, and at 16 replies llama.cpp leads, 130 against 105. A model
that keeps its MTP head drafts with it on MLX too: Ollama's qwen3.8:27b-mlx on a
MacBook Pro (M5 Max) went from 30 tok/s to 40–107, 72.5 on average against Ollama's 78. Pictures and embeddings run only
on llama.cpp so far. A mixture of experts runs on MLX too: Ollama's
qwen3.5:35b-a3b-nvfp4 writes 104 tok/s alone against Ollama's 125, and four
replies at once 188 against 112. The numbers: [CHANGELOG.md](CHANGELOG.md).

## Models bigger than the machine

A model the GPU cannot hold needs no flags. A mixture of experts uses a few of
its experts for each token — 10 of Flash-Next's 512 — so kinfer reads the
model's header, keeps on the GPU what every token uses and a cache of the
experts in use, and leaves the other experts in host memory, read from the SSD
as the cache misses them. With another Mac on a Thunderbolt cable running
`kinfer worker`, `kinfer run` finds it by itself and borrows its GPU for the
layers that do not fit here — 44 tok/s on the MacBook Pro where its SSD gives 30
— and a model that fits never touches it.

Qwen3.8-Flash-Next IQ3_S — 51 GiB of weights, and a 27 GiB table read a few
rows a token — with no flags; a 7,753-token prompt and three 200-token replies:

| machine | GPU memory | where the model went | prompt | decode |
|---|---:|---|---:|---:|
| Mac Studio M3 Ultra, 96 GB | 88 GiB | all of it on the GPU | 748 tok/s | 43 tok/s |
| MacBook Pro M5 Max, 48 GB | 40 GiB | 44 of 48 layers' experts in host memory and on the SSD, a 15 GiB cache (v0.2.5) | 876 tok/s | 30-32 tok/s |
| the MacBook Pro, the Mac Studio as its peer over Thunderbolt 5 | 40 + 88 GiB | 17 GB of it on the Mac Studio's GPU | 1,274-1,325 tok/s | 43.6-44.4 tok/s |

`kinfer plan <model>` shows where a model would go, without loading it. Reading
experts from the SSD at that speed takes kinfer's build of llama.cpp
(`scripts/build-libs.sh`, below); with the release libraries the MacBook Pro
decoded at 9 tok/s. Over Thunderbolt 5 the other machine's GPU beats the laptop's
own SSD; over the USB 2.0 cable used first it did not (23 tok/s against 28,
[docs/big-models.md](docs/big-models.md#over-thunderbolt-5-and-the-gpu-that-waits)).

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
  `install` keeps the server up; embeddings (`/api/embed`, `/v1/embeddings`) from a
  model loaded beside the chat model; pictures, for a model with a projector; a
  model with an MTP head drafts its next tokens (Qwen3.8-27B: 31.5 → 37–69 tok/s);
  every reply carries its timings; `/metrics`; a
  bounded queue that refuses rather than hangs; `fit`, `plan` and `can-run` size
  a model before it is loaded or downloaded.
- **Agents** ([docs/agents.md](docs/agents.md)): `kinfer launch claude` or
  `kinfer launch codex` starts Claude Code or Codex on a local model (and Pi
  and Hermes Agent, untested so far; OpenClaw is told how); a prefix
  pool, so a repeated system prompt is read once; chat templates read from the
  model; tool calls in the Ollama and OpenAI dialects; a reasoning model's
  thinking kept apart from its answer.
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
tools) into `darwin_arm64_b11998k7/`, replacing the release set. It matters
for a model whose experts or tables come from the SSD — see
[When the experts do not fit in memory either](docs/big-models.md#when-the-experts-do-not-fit-in-memory-either);
with the release set `-host-experts` and `-moe-cache` work as before and there is
no profile — and for a model split across machines, whose GPUs it keeps at speed
between tokens (with the release set a split decodes at about 25 tok/s where the
patched one reaches 42), and whose connections notice a machine that went away
within about 25 s instead of two hours.

`scripts/release.sh v0.1.0` builds what a release ships: those patched
libraries, the version stamped in, packed with its sha256 into `dist/`, and then
the Linux and Windows packages, on llama.cpp's own CPU builds.
`scripts/brew-formula.sh v0.1.0 > ../homebrew-tap/Formula/kinfer.rb` writes the
formula for that release into the tap,
[LocalKinAI/homebrew-tap](https://github.com/LocalKinAI/homebrew-tap).

`cmd/kinlab` is a bench for editing a small model's weights and measuring what
an edit does — `kinlab edit` writes an edited model into kinfer's store, `kinlab
eval` runs a set of problems through it against its base. It builds from the
same tree (`go build -o kinlab ./cmd/kinlab`) and is not part of a release.

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
- [x] **Bigger than one machine, at speed** — over Thunderbolt 5 with the GPUs
      kept warm a split decodes as fast as the box alone, and `kinfer run` borrows a
      worker it finds on the cable only for a model that does not fit here
- [x] **MLX** — safetensors served on MLX from Go: Qwen3.5 and Qwen3.8, batched, with a
      prefix pool and MTP drafting; a small model's many replies at once 2.7 times llama.cpp's
- [ ] **Phase 2** — the reason this exists: health probing, automatic failover,
      circuit breaking, per-agent model routing
- [ ] **Phase 3** — apply `fit`'s backend choice automatically, honest benchmarks
      vs Ollama

## The LocalKin family

kinfer is one of [LocalKin](https://www.localkin.app)'s open-source tools. `pip install localkin`
installs it with its two siblings; each is also a package of its own.

| Package | Command | What it does |
|---|---|---|
| [kinfer-ai](https://pypi.org/project/kinfer-ai/) | `kinfer` | local LLM serving for agent fleets: one binary, the Ollama, OpenAI and Anthropic APIs, models bigger than the machine (this project) |
| [ollamadiffuser](https://pypi.org/project/ollamadiffuser/) | `ollamadiffuser` | local image and video generation with an Ollama-style CLI: Stable Diffusion, FLUX, Qwen-Image, LTX |
| [localkin-service-audio](https://pypi.org/project/localkin-service-audio/) | `kin audio` | local speech: speech-to-text and text-to-speech, voice cloning, an MCP server for Claude |

The rest of the tools, written in Go: [localkin.app](https://www.localkin.app).

## License

Apache 2.0
