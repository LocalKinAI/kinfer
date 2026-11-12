<div align="center">

# kinfer

**A private AI server in one file.**

Chat, speech, pictures and embeddings behind one address, in four API dialects.<br>
Built for fleets of agents, and for models bigger than the machine.

[![Release](https://img.shields.io/github/v/release/LocalKinAI/kinfer)](https://github.com/LocalKinAI/kinfer/releases)
[![PyPI](https://img.shields.io/pypi/v/kinfer-ai)](https://pypi.org/project/kinfer-ai/)
[![Downloads](https://img.shields.io/pepy/dt/kinfer-ai)](https://pepy.tech/project/kinfer-ai)
[![Downloads a month](https://img.shields.io/pypi/dm/kinfer-ai)](https://pypistats.org/packages/kinfer-ai)
[![GitHub downloads](https://img.shields.io/github/downloads/LocalKinAI/kinfer/total)](https://github.com/LocalKinAI/kinfer/releases)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

<sub>macOS on Apple silicon · Linux and Windows as a preview</sub>

[Quickstart](#quickstart) · [The numbers](#the-numbers) · [For agents](#for-agents) · [Speech and pictures](#it-hears-speaks-and-paints) · [Bigger than the machine](#models-bigger-than-the-machine) · [Docs](#read-more) · [Changelog](CHANGELOG.md)

</div>

```console
$ kinfer pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
$ kinfer run qwen "explain merkle trees in one sentence"
A Merkle tree is a data structure used in cryptography to store and
retrieve data in a way that is resistant to attacks…

$ kinfer serve
kinfer serving on :11500 — 1 model(s) in ~/.kinfer/models
  Ollama API : POST :11500/api/chat        GET :11500/api/tags
  OpenAI API : POST :11500/v1/chat/completions
  Anthropic  : POST :11500/v1/messages          (Claude Code)
  Responses  : POST :11500/v1/responses         (Codex)
  Speech     : POST :11500/v1/audio/transcriptions   :11500/v1/audio/speech
  Pictures   : POST :11500/v1/images/generations
```

One binary of 23 MB, with llama.cpp inside. No Docker, no Python, no background
service: copy it and it runs.

| 28 tok/s | 17 ms | 2,833 tok/s | 0.09 s |
|:---:|:---:|:---:|:---:|
| from a 51 GiB model on a 48 GB laptop, with no flags.<br>Ollama on the same files: 7–8 | to the first token when a 3,100-word system prompt is already in the prefix pool.<br>Without the pool: 266 ms | from 64 replies at once on MLX, where llama.cpp on the same Mac writes 1,031 | to the first audio of streamed speech, from a model that holds 1.5 GB |

<sub>Measured on our own Macs. Each figure's machine, model and conditions are under [The numbers](#the-numbers), and where kinfer is behind, that is there too.</sub>

## What is in the file

<table>
<tr>
<td width="33%" valign="top">

**📦 One file**

23 MB, with llama.cpp inside. Models are plain files in `~/.kinfer/models`:
`ls` says what you have, and `scp` moves one to another machine. Ollama's own
models are served where they lie.

</td>
<td width="33%" valign="top">

**🔌 Four dialects, one port**

Ollama, OpenAI, Anthropic Messages and OpenAI Responses, from the same engine.
A fleet moves over by changing one endpoint, and back the same way. `kinfer
launch claude` starts Claude Code on a local model.

</td>
<td width="33%" valign="top">

**🧠 Bigger than the machine**

A mixture of experts the GPU cannot hold needs no flags: the experts in use are
cached on the GPU and the rest read from the SSD, at speed. A second Mac on a
Thunderbolt cable lends its GPU.

</td>
</tr>
<tr>
<td width="33%" valign="top">

**🐝 Many agents at once**

Continuous batching, a prefix pool that reads a shared system prompt once,
drafting with a model's own MTP head, and a bounded queue that answers 503
rather than hang.

</td>
<td width="33%" valign="top">

**🎙 Hears, speaks, paints, embeds**

Qwen3-ASR, Qwen3-TTS, Z-Image-Turbo and embedding models at the chat model's
own address. Each is fetched the first time it is asked for.

</td>
<td width="33%" valign="top">

**🧱 A process for every model**

A model that crashes takes only itself down, and the next request starts it
again. One unloaded gives back all its memory. Before one loads, room is made
on the GPU for it.

</td>
</tr>
</table>

## Why it exists

LocalKin runs ~150 agents against cloud models. On 2026-07-23 one of those
models (`kimi-k2.5:cloud`) stopped responding — not an error, a *hang* — and the
whole fleet went silent for half a day. The failure was not in the model; it was
in having no second leg to stand on.

kinfer is that second leg: a local inference path that is **structurally
independent** of any cloud provider or background service. Switching a fleet to
it is one endpoint change, and switching back, when something goes wrong at 2am,
is the same one change.

Existing local runners are built for *a person talking to a model*. kinfer is
built for *a fleet of agents that must not go down*.

|  | Ollama | Docker Model Runner | kinfer |
|---|---|---|---|
| Install | background service | Docker required | **copy one file** |
| Model source | registry + `hf.co/...` | Docker Hub / OCI / HF | Hugging Face directly, and Ollama's own store |
| On-disk models | blob hashes | OCI layers | **plain filenames** |
| Designed for | one user, one chat | container workflows | **agent fleets** |

Plain filenames matter more than it sounds: when something breaks at 2am you
want `ls ~/.kinfer/models` to tell you what you have, and you want `scp` to be a
valid way of moving a model to another machine.

## Quickstart

```bash
# macOS on Apple silicon, or Linux
curl -fsSL https://raw.githubusercontent.com/LocalKinAI/kinfer/main/scripts/install.sh | sh

kinfer fit                                              # what this machine can actually run
kinfer pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M      # 469 MB, from Hugging Face
kinfer run qwen "explain merkle trees in one sentence"  # the model stays warm in a daemon
kinfer serve                                            # everything, at :11500
```

Or with Homebrew, which upgrades it with everything else, or with pip, into
whatever Python environment you work in:

```bash
brew tap localkinai/tap
brew trust --formula localkinai/tap/kinfer
brew install localkinai/tap/kinfer

pip install kinfer-ai    # `kinfer` on PyPI is another project; the command is still kinfer
```

<details>
<summary><b>What the installers do</b></summary>

The script puts the latest release in `~/.local/bin`, checked against its
sha256: one file with llama.cpp inside, the build that reads experts from the
SSD at speed. It is not signed. Downloaded this way macOS runs it, and a
firewall that is on asks once before `serve` or `worker` takes connections from
another machine.

With other taps not trusted yet, Homebrew 7 can end the install with an error
about one of them, after kinfer is in place; `kinfer version` says whether it is,
and `HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK=1` before `brew install` skips the
check that raised it.

The wheel holds the same release binary, for macOS 26 on Apple Silicon or the
Linux and Windows builds below, and `pip install -U kinfer-ai` upgrades it.

</details>

<details>
<summary><b>Linux and Windows: a preview</b></summary>

kinfer is developed and tested on the Mac. The Linux and Windows builds are a
preview: x86-64 and arm64 pass the test suite and serve a small model on real
machines in CI, and nobody here runs them daily.

They run on the CPU, and on a GPU through Vulkan as an experiment. On a machine
with a Vulkan driver kinfer fetches the backend the first time a model loads
(about 35 MB, once, into `~/.kinfer/gpu`); a model that does not fit the GPU, or
fails on it, runs on the CPU instead, and `KINFER_GPU=off` leaves the GPU out.
Here it has only run on a software Vulkan driver on Linux, so real GPUs, and
Windows, are yours to try: `kinfer selftest` runs a small model on the GPU and on
the CPU and says whether the answers make sense. Please send what it prints.

CUDA on NVIDIA cards and ROCm on AMD ones are hundreds of megabytes, so they are
fetched only when asked for, and they have never run on a real card here either:

```bash
kinfer gpu install cuda   # says the size first (about 600 MB) and asks
kinfer gpu install rocm   # about 250 MB, 1.2 GB unpacked; on Linux AMD's ROCm 10 must be installed
kinfer gpu                # what is fetched, and what models load on
```

A backend that does not start, or finds no GPU, is left unused.

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

</details>

Then point what you already have at it:

| What you have | What it needs |
|---|---|
| Anything that speaks Ollama | its host set to `http://localhost:11500` |
| An OpenAI SDK | its base URL set to `http://localhost:11500/v1` |
| Claude Code | `kinfer launch claude -model qwen3.8` |
| Codex | `kinfer launch codex -model ornith-1.5:35b` |
| A LocalKin soul | its `brain.endpoint` set to the server's address |

## The numbers

Every figure here was measured on the machines named, against the other runtime
on the same files where one is named, and is written up with its conditions in
the docs. Where kinfer is behind, that is written down too.

### A model the machine cannot hold

Qwen3.8-Flash-Next IQ3_S — 51 GiB of weights, and a 27 GiB table read a few
rows a token — with no flags; a 7,753-token prompt and three 200-token replies:

| machine | GPU memory | where the model went | prompt | decode |
|---|---:|---|---:|---:|
| Mac Studio M3 Ultra, 96 GB | 88 GiB | all of it on the GPU | 748 tok/s | 43 tok/s |
| MacBook Pro M5 Max, 48 GB | 40 GiB | 44 of 48 layers' experts in host memory and on the SSD, a 15 GiB cache (v0.2.5) | 876 tok/s | 30-32 tok/s |
| the MacBook Pro, the Mac Studio as its peer over Thunderbolt 5 | 40 + 88 GiB | 17 GB of it on the Mac Studio's GPU | 1,274-1,325 tok/s | 43.6-44.4 tok/s |

Against Ollama on the same MacBook Pro, from the same files, the same requests
back to back, and the same reply from all three:

| | prompt | decode |
|---|---:|---:|
| Ollama 0.40.13, as it sets itself up | 17 tok/s | 6.9-8.4 tok/s |
| Ollama's `llama-server` with kinfer's hand-tuned settings (`--n-cpu-moe 24 --moe-cache-mib 8192 -ub 2048`) | 81 tok/s | 6.3-8.2 tok/s |
| **kinfer, no flags** | **716 tok/s** | **27.9-28.4 tok/s** |

Ollama took 459 seconds to read that prompt, and kinfer 11. DeepSeek-V4.1, at
129 GiB, decodes at about 15 tok/s on the same laptop. How it works, and every
measurement behind it: [docs/big-models.md](docs/big-models.md).

### Many conversations at once

ornith-1.5-35b on the Mac Studio, the same file in both runtimes, the replies
identical (Ollama 0.34.3, as installed):

| | kinfer | Ollama |
|---|---:|---:|
| one request | **106.6 tok/s** | 103.3 tok/s |
| four at once | **166 tok/s**, the last one done in 9.6 s | 102 tok/s, in 15.8 s |
| eight at once | **154 tok/s**, in 20.8 s | 102 tok/s, in 31.5 s |

- A small model has more to give. Qwen2.5-0.5B, through HTTP on the same Mac
  Studio, writes 295 tok/s alone, 2,588 with 16 at once, and 4,803 with 128 at
  once and the prefix pool off (`-prefix -1`).
- On MLX, Qwen3.5-0.8B writes 2,833 tok/s with 64 at once against llama.cpp's
  1,031, in half the memory.
- A model with an MTP head drafts its own next tokens. Qwen3.8-27B goes from
  31.5 tok/s to 37–69, and at temperature 0 its replies came out identical with
  drafts and without, fifteen of fifteen.

More in [docs/performance.md](docs/performance.md) and
[docs/serving.md](docs/serving.md#drafting-with-the-models-mtp-head).

### Where it is behind

- Ollama 0.34.3 reads a 103,641-token prompt 4% sooner: 121.8 s against 126.4.
- With MTP drafting on MLX, Ollama 0.40.15 averaged 78 tok/s over six replies
  where kinfer averaged 72.5 (qwen3.8:27b-mlx on the MacBook Pro).
- A mixture of experts on MLX writes 104 tok/s alone against Ollama's 125. With
  four replies at once it is 188 against 112.
- Between 17 and about 48 conversations at once, Metal's larger step is not yet
  paid for: 24 at once write less than 16. Slot counts are 8, 16, 64 or 128.

## For agents

```bash
kinfer launch claude -model qwen3.8          # Claude Code on a local model
kinfer launch codex -model ornith-1.5:35b    # Codex
```

`launch` gives the agent an address, a model and the context window it really
has, and writes nothing to the agent's configuration. From a Mac mini against
the Mac Studio, on ornith-1.5:35b, the same task — write a word counter and
three tests, run them, fix what fails — took Claude Code 37 s, model load
included, and Codex 15 s. The tests passed both times.

**A system prompt is read once.** An agent's soul is the same bytes on every
turn, and a hundred agents share it. Recently seen prefixes stay in the KV
cache, and a request that begins the same way adopts those cells instead of
computing them again. With a 3,100-word system prompt, on the Mac Studio:

| | no pool | prefix pool |
|---|---:|---:|
| the same soul, repeated: time to the first token | 266 ms | **17 ms** |
| a conversation's turns 2–5: time to the first token | 283 ms | **22 ms** |
| 8 agents sharing a soul, at once: until all have begun | 2.38 s | **0.16 s** |

**What else an agent gets:**

- **Tool calls held to their schema.** A call is drawn under a grammar made from
  the request's functions: well formed, closed, and each declared parameter the
  type its schema says. On the LocalKin swarm's seats it turned `"hourly": True`
  into `true`. GGUF models only, for now.
- **Thinking kept apart.** A reasoning model's working out comes back in its own
  field in every dialect, and a `<tool_call>` it only considered is not mistaken
  for one it made.
- **Templates read from the model**, 54 families of them, and tool conventions
  read from the template. A model told to answer in a form it was not taught
  gets it wrong four runs in four.
- **A server that says no.** A bounded queue answers 503 instead of holding a
  connection: flooded with 64 requests at four slots and a queue of eight, 12
  were answered and 52 refused in 0.06 s. A clock bounds every reply, and one
  cut short says `timeout` or `length`, never `stop`.
- **A reply that survives a crash.** When llama.cpp fails under a model, the
  model is loaded again and the requests go on: of 32 at once against a dying
  backend, 28 were answered, and the ones already streaming continued where
  they stopped.
- **Timings on every reply**, in Ollama's fields, and `/metrics` for Prometheus.

All of it, with the measurements: [docs/agents.md](docs/agents.md) and
[docs/serving.md](docs/serving.md#staying-up-under-load).

## It hears, speaks and paints

On a Mac, the chat model's address answers OpenAI's speech and picture
endpoints too. Each model runs in a process of its own, so the chat model stays
loaded and all of them can be used at once.

```bash
kinfer serve                                        # chat, hearing, speaking and painting at :11500

curl localhost:11500/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"一个地址，什么都能做。"}' -o one.wav

curl -F file=@one.wav -F model=whisper-1 localhost:11500/v1/audio/transcriptions
{"text":"一个地址，什么都能做。"}

curl localhost:11500/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"黄昏时的江南水乡，小船，灯笼，水墨画","size":"768x1024","seed":3}' \
  -o picture.json                                   # the PNG, in base64, at data[0].b64_json
```

| | the model | on the Mac Studio | |
|---|---|---|---|
| Speech to text | Qwen3-ASR 1.7B | a 163 s recording in 2.9 s | 30 languages; SenseVoice adds the emotion and the sounds |
| Text to speech | Qwen3-TTS 0.6B | 5.2× real time, the first audio in 0.09 s, 1.5 GB | nine voices in ten languages, or a voice told in words (VoiceDesign) |
| Pictures | Z-Image-Turbo | 1024×1024 in 31 s, 512×512 in 7.5 s | 256 to 2048 a side |
| Embeddings | any embedding GGUF; EmbeddingGemma 2 on MLX | EmbeddingGemma 2's vectors are Ollama's, to a cosine of 0.9998 or better | text, and for EmbeddingGemma 2 pictures and sound |

- OpenAI's names mean this machine's default: `whisper-1`, `tts-1`, `dall-e-3`
  and `gpt-image-1` all answer.
- A default that is not here is fetched the first time it is asked for —
  Qwen3-ASR 2.5 GB, Qwen3-TTS 1.8 GB, Z-Image-Turbo 11.0 GB — and the request is
  answered 503 with how far the fetch has got. `kinfer pull speech` fetches the
  speech models beforehand, and `-auto-pull=false` turns fetching off.
- A chat model with a projector reads pictures in all four dialects.
- Speech to text runs on llama.cpp everywhere. Text to speech and pictures from
  words run on MLX, so on a Mac.

Every parameter, and how close each port is to the code it was ported from:
[docs/serving.md](docs/serving.md#one-address-for-chat-and-speech).

## Models bigger than the machine

A model the GPU cannot hold needs no flags. A mixture of experts uses a few of
its experts for each token — 10 of Flash-Next's 512 — so kinfer reads the
model's header, keeps on the GPU what every token uses and a cache of the
experts in use, and leaves the other experts in host memory, read from the SSD
as the cache misses them. With another Mac on a Thunderbolt cable running
`kinfer worker`, `kinfer run` finds it by itself and borrows its GPU for the
layers that do not fit here — 44 tok/s on the MacBook Pro where its SSD gives 30
— and a model that fits never touches it. The measurements are
[above](#a-model-the-machine-cannot-hold).

`kinfer plan <model>` shows where a model would go, without loading it. Reading
experts from the SSD at that speed takes kinfer's build of llama.cpp
(`scripts/build-libs.sh`, below); with the release libraries the MacBook Pro
decoded at 9 tok/s. Over Thunderbolt 5 the other machine's GPU beats the laptop's
own SSD; over the USB 2.0 cable used first it did not (23 tok/s against 28,
[docs/big-models.md](docs/big-models.md#over-thunderbolt-5-and-the-gpu-that-waits)).

Against Ollama the settings are not the difference; kinfer's build of llama.cpp
is. It reads the experts the cache misses with a pool of threads instead of a
page fault at a time, reads the next ones ahead, and streams a prompt's passes.

Part of the way there is Strata's, whose own engine runs a mixture of experts
from the SSD on a Mac: a GPU cache of the experts in use with the rest read from
the file, misses read by a pool of threads, the next layer's experts guessed
from its router and read ahead, and a profile of the experts generation uses
filling the cache at load. kinfer does those in llama.cpp, whose MoE cache it
patches; the rest is its own.

<details>
<summary><b>Step by step, and whose each step is</b></summary>

On the MacBook Pro, the 7,753-token prompt (each step measured as it was made,
and the laptop wanders by a quarter from run to run):

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
several.

</details>

## Two engines

A model is a GGUF, served by llama.cpp, or a directory of safetensors as Hugging
Face publishes it, served on Apple's MLX: Qwen3.5 and Qwen3.8, at bf16 or MLX
4-bit, such as mlx-community's conversions, or NVFP4. `kinfer pull
mlx-community/Qwen3.8-27B-4bit` fetches one into the store; from there it is
listed, run and served like a GGUF, through the same four dialects. Ollama's own
MLX models (`qwen3.8:27b-mlx`) are listed and served where they lie, as its
GGUFs are. The MLX engine is Go over mlx-c, with no cgo and no Python, and
checked against mlx-lm layer by layer. MLX's libraries (78 MB) are fetched once,
the first time an MLX model loads, and need macOS 26.2.

Neither engine wins everywhere, which is why kinfer carries both. On the Mac
Studio:

| | MLX | llama.cpp |
|---|---:|---:|
| Qwen3.5-0.8B, 64 replies at once | **2,833 tok/s**, in half the memory | 1,031 tok/s |
| Qwen3.8-27B, one reply | 36 tok/s | 31 tok/s, **51** with MTP drafting |
| Qwen3.8-27B, 16 replies at once | 105 tok/s | **130 tok/s** |

A model that keeps its MTP head drafts with it on MLX too: Ollama's
qwen3.8:27b-mlx on the MacBook Pro went from 30 tok/s to 40–107. Pictures read
by a chat model run only on llama.cpp so far, and so do embeddings, but for
EmbeddingGemma 2, which Ollama keeps in NVFP4: that one runs on MLX. The
numbers: [CHANGELOG.md](CHANGELOG.md).

## How it is built

```
          Claude Code · Codex · Ollama clients · OpenAI SDKs · your agents
                                          │
                 ┌────────────────────────┴───────────────────────┐
                 │             kinfer serve   :11500              │
                 │    Ollama · OpenAI · Anthropic · Responses     │
                 │     chat · embeddings · speech · pictures      │
                 └────────────────────────┬───────────────────────┘
                                          │   a process for every model
  ┌───────────────┬───────────────┬───────┴───────┬───────────────┬───────────────┐
  │      chat     │   embeddings  │     hears     │     speaks    │     paints    │
  │   GGUF · MLX  │   GGUF · MLX  │   Qwen3-ASR   │   Qwen3-TTS   │ Z-Image-Turbo │
  └───────────────┴───────────────┴───────┬───────┴───────────────┴───────────────┘
                                          │
            llama.cpp on Metal, inside the binary   ·   MLX, fetched once
                                          ┆
               kinfer worker on a second Mac, over a Thunderbolt cable
```

- **No cgo anywhere.** The native libraries load at run time through `purego`,
  which keeps `GOOS=linux go build` working from a Mac and the build free of a C
  toolchain.
- **One dependency.** `go.mod` requires `purego` and nothing else. The llama.cpp
  surface kinfer uses is about twenty entry points bound in `internal/llama`,
  and each struct passed by value is checked against `llama.h`'s size at
  startup, because a silent layout drift is the failure that binding exists to
  end.
- **llama.cpp patched in the open.** Eight patches in `patches/llama.cpp`:
  experts read from the SSD in parallel and ahead, the GPU kept warm between
  tokens, connections that notice a machine gone, free memory counted for the
  whole GPU.
- **A process for every model.** The server loads none itself. Each model runs
  in this same binary on a loopback port, started on the first request for it
  and stopped once idle. Killed with `-9` in the middle of a reply, a model
  answered the next request 2.3 s later.

Why llama.cpp is bound directly, and the five bugs that decided it:
[docs/design.md](docs/design.md).

## Commands

```bash
kinfer fit                    # what this machine can actually run
kinfer can-run <hf-repo>      # whether a model on Hugging Face will run here
kinfer pull <repo>[:quant]    # Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M, or an MLX repo: mlx-community/Qwen3.8-27B-4bit
kinfer pull speech            # this machine's default speech models
kinfer list                   # what is installed, Ollama's models included
kinfer rm <model>             # delete one
kinfer run <model> [prompt]   # chat — no prompt means interactive
kinfer launch [agent]         # Claude Code or Codex on a local model
kinfer ps                     # what the daemon is holding right now
kinfer stop                   # stop that daemon
kinfer plan <model>           # where a model would go, and the -slots and -ctx that fit
kinfer serve [-addr :11500]   # serve over HTTP; context and slots sized to the model
kinfer install [serve flags]  # keep serve running: at login and after a crash (macOS)
kinfer uninstall              # stop it and remove the service
kinfer worker                 # lend this machine's GPU over a Thunderbolt cable
kinfer worker -on <user>      # start one on the Mac at the other end of the cable
kinfer selftest [model]       # a small model on the GPU and the CPU: are the answers right?
kinfer gpu                    # GPU backends off a Mac: install cuda or rocm, use, remove
kinfer version                # this build, and the llama.cpp inside it
```

`kinfer pull` fetches a file in pieces over eight connections at once
(`-connections n`), and keeps what has arrived when it stops — Ctrl-C, a dropped
network, a full disk: the same pull again fetches only the rest, and until then
`kinfer list` shows it unfinished. A split model arrives with every piece, and
each file takes its name only once it matches the sha256 Hugging Face publishes
for it.

What each does, and how serve sizes itself: [docs/serving.md](docs/serving.md).

## Status

- **In daily use** on Apple silicon Macs, where it serves LocalKin's own agents.
  Each release is run on our own Macs before it ships, and its notes in
  [CHANGELOG.md](CHANGELOG.md) say what was run and what was not.
- **A preview** on Linux and Windows, on the CPU: nobody here runs them daily,
  and each release's notes say whether they were run at all.
- **An experiment** off the Mac's GPU: Vulkan has only run on a software driver,
  and CUDA and ROCm on no real card. `kinfer launch` for Pi, Hermes Agent and
  OpenClaw is written from their documentation and not yet run.
- **MLX serves Qwen3.5 and Qwen3.8** as chat models; anything else is a GGUF.
- **No keys yet.** `serve` answers whatever reaches its port, and listens on
  every interface unless told otherwise: `kinfer serve -addr 127.0.0.1:11500`
  keeps it to this machine. Put it on a network you trust.
- **Not built:** failing over between providers, and probing their health.
  kinfer is the local leg; when to stand on it is still the fleet's decision.

## Build

<details>
<summary><b>From source, and how a release is made</b></summary>

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
tools) into `darwin_arm64_b11998k8/`, replacing the release set. It matters
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

</details>

## Read more

| | |
|---|---|
| [Serving](docs/serving.md) | how `serve` sizes itself, speech, pictures, embeddings, `install`, metrics |
| [For agents](docs/agents.md) | `launch`, the prefix pool, templates, tool calls, reasoning models |
| [Models bigger than the machine](docs/big-models.md) | experts from the SSD, two machines, placement, and every measurement |
| [Performance](docs/performance.md) | continuous batching, and measurements against Ollama on the same files |
| [Design](docs/design.md) | how the code is laid out, and why llama.cpp is bound directly |
| [Changelog](CHANGELOG.md) | what each release did, what was measured, and what was not run |

## The LocalKin family

kinfer is one of [LocalKin](https://www.localkin.app)'s open-source tools. `pip install localkin`
installs it with its two siblings; each is also a package of its own.

| Package | Command | What it does |
|---|---|---|
| [kinfer-ai](https://pypi.org/project/kinfer-ai/) | `kinfer` | local LLM serving for agent fleets: one binary, the Ollama, OpenAI and Anthropic APIs, models bigger than the machine (this project) |
| [ollamadiffuser](https://pypi.org/project/ollamadiffuser/) | `ollamadiffuser` | local image and video generation with an Ollama-style CLI: Stable Diffusion, FLUX, Qwen-Image, LTX |
| [localkin-service-audio](https://pypi.org/project/localkin-service-audio/) | `kin audio` | local speech: speech-to-text and text-to-speech, voice cloning, an MCP server for Claude |

The rest of the tools, written in Go: [localkin.app](https://www.localkin.app).

## Standing on

- [llama.cpp](https://github.com/ggml-org/llama.cpp), the engine inside the
  binary, and [MLX](https://github.com/ml-explore/mlx) with mlx-c, the one
  fetched beside it.
- Strata, for the way a mixture of experts is run from the SSD (see
  [above](#models-bigger-than-the-machine)).
- mlx-audio, whose Qwen3-ASR and Qwen3-TTS kinfer's are ports of; mflux, for
  Z-Image-Turbo; Ollama, for EmbeddingGemma 2's picture and sound towers, and
  for the API a fleet already speaks.
- [purego](https://github.com/ebitengine/purego), which is why there is no cgo.

## License

Apache 2.0
