# Changelog

## v0.8.0 — unreleased

### Added — the machine's other programs, under one manager

On the box, beside kinfer, ComfyUI, a talking face and a navigation model held
memory each on their own. The face held 10 GB and the navigation model 9.8 GB
while nothing used them, and none of them knew the others were there.
`kinfer service` takes them in.

- `kinfer service scan` lists the programs listening here as service entries.
  - It reads, from each process itself: its exact command line, its folder,
    its environment less the session's, its launchd job, and its memory
    footprint (the GPU's share included).
  - Its port and host become `{port}` and `{host}`, whether written after a
    flag, as `--flag=value`, as a bare word, or in the environment.
  - `-save` writes the new ones to `~/.kinfer/services.json`, watched.
- `/api/ps` shows every service as `services`: its process, what it holds now,
  and the most it was seen to hold.
- `kinfer service adopt NAME` hands one to `kinfer serve`:
  - its own process is ended, and the front answers on its address;
  - the first request starts it on a loopback port, and HTTP and WebSocket go
    through, so its clients change nothing;
  - idle for `-idle`, it stops;
  - the room it held last time is made before it starts, and the runner or
    service unused longest goes first when anything needs room.
- `release` gives it back, running on its own; `stop` stops kinfer's copy.
  These are asked from the machine itself only.

Tested on the box with a stand-in: found and templated; adopted, with its own
process ended; started by the first request in 0.4 s; an upgrade request passed
through; stopped after 15 s idle; started again by the next request; released
and running on its own again. Unit tests cover:
- requests arriving together while the service is stopped, which start it once;
- refusing an address another program holds;
- ending what a crashed front left running.

## v0.7.2 — 2026-11-11

### Added — pictures and sound in EmbeddingGemma 2

kinhome places its pictures and recordings by meaning, beside its words, with
EmbeddingGemma 2, so to move it off the box's Ollama kinfer has to read them
as Ollama does. `/api/embed` now takes an input as Ollama 0.40 takes it for
its MLX embedding models: `{"image": base64}` or `{"audio": base64}` (a file
or a list, a data URL too), with an optional `"text"` before them. A
text-only model answers one with a 400, as before.

The release's own towers read them (internal/mlxembed), ported from Ollama:
- **Pictures:** resized to whole 48-pixel cells within 280 soft tokens with
  the Catmull-Rom kernel Ollama scales with, cut into 16-pixel patches, read by
  the 16-layer vision tower with a rotary position on each axis, and pooled 3×3.
- **Sound:** a WAV at any rate, resampled to 16 kHz with the same Kaiser sinc,
  cut into pieces of at most 30 s at their quietest moments, and read by the
  12-layer conformer as log-mel frames.

Beyond Ollama, a picture Go does not decode (WebP, HEIC, TIFF) goes through
macOS's `sips`, and a sound that is not a WAV (mp3, m4a, aac, ogg) through
`afconvert`. Ollama refuses those sounds.

Measured on the box against Ollama, cosine of the two vectors:

| Input | Cosine |
|---|---|
| Six made-up pictures, from 120×90 to 3000×2000, one with alpha | 0.99990–0.99999 |
| Six of kinhome's own PNGs | 0.99996–0.99999 |
| Six of kinhome's own WAVs, one of 1,304 tokens | 0.99981–0.99996 |
| Four of kinhome's WebPs, which macOS decodes slightly differently from Ollama's Go decoder | 0.9972–0.9990 |

Every input read as the same number of tokens. Three mp3s that Ollama refuses
were read.

## v0.7.1 — 2026-11-11

What kinhome's test home found running on the box for a night, put right.

The regression ran through runners on both machines, built as a release is:

- **Box:** all six models (the 0.5B, ornith, qwen3.8-27B as GGUF and on MLX,
  Flash-Next, V4.1), and V4.1 split with the laptop's worker over Thunderbolt.
- **Laptop:** four models (ornith, qwen3.8-27B on MLX, Flash-Next, V4.1).
- **Everywhere, the same checks:** a reply, the same after a reload, 4 at once
  and 16 at once, a tool call, and all four dialects.
- **Other paths:** room made for a picture, a runner killed mid-reply, `kinfer
  run` (daemon and `-local`), `kinfer launch claude` and `codex`.

Every check passed but one already known: a Responses reply cut at a short
token limit, which completes at 700. Against v0.7.0 on the laptop, back to
back, the replies were the same and so were the speeds: ornith 3,280 tok/s on a
7,470-token prompt and 136 decoding; Flash-Next 903–987 and 13.7–17.0, as before.

Not run:
- the laptop borrowing the box as a worker;
- the new sizing step on a GPU as short of memory as on the night it was found:
  the box had more room this time, and that step is covered by unit tests;
- Linux and Windows beyond building.

The one-model-one-runner fix came after the regression; it was tested on its
own.

### Fixed — tool calls written back without their arguments

A client in OpenAI's dialect sends an earlier turn's call back with its
arguments as a string holding the object. kinfer wrote such calls into the
prompt of a model of the `<function=…>` kind — ornith, Flash-Next — with no
parameters at all, and the model, seeing its own earlier calls empty, wrote its
next ones empty too: `remind {}`, the largest cause of ornith's failures in
kinhome's test home. The string is read now (as V4.1's markup already did), and
the parameters are written in the order they came: read through a map they had
come out in a different order on every turn, so a prompt no longer repeated the
one before it token for token. On the laptop, twelve of the test home's kept
requests sent three times each: v0.7.0 answered with 31 calls, 22 of them
empty; v0.7.1 with 28, none empty that needed arguments.

### Fixed — a hybrid model beside another gave up its whole prefix pool

On the box, qwen3.8-27B loaded beside ornith with the service's `-slots 4` had
too little memory for its pool and gave the last entry up to keep four slots of
32,768 tokens. A model with recurrent state reuses nothing without one, so each
of the test home's 13,000-token turns was read whole again: 43–50 s to a first
token, where ornith's took 0.4. Now the context is halved first (not below
16,384 a slot; a longer prompt still gets a longer slot while it runs), and the
pool keeps its last entry.

### Fixed — replies lost on the box to clients that ask Connection: close

Python's urllib asks every server to close the connection after its reply, and
on the box most of those replies never arrived: six of eight of the test home's
requests sent from the mini came back as `RemoteDisconnected`, and the server
logged nothing. Ollama on the box lost them the same way, and nothing was lost
from the box's own processes or on the other Macs; the box runs network filters
of its own, which drop a reply whose connection is closed the moment it is
written. kinfer's addresses now close a connection the orderly way, its writing
half first, then the socket once the client has closed its own: 0 of 10 lost
through urllib after, 20 of 20 answered for curl told to close.

### Added — EmbeddingGemma 2

`embeddinggemma-2` from Ollama's library is safetensors in NVFP4, not a GGUF,
and kinfer listed it and could not load it. It is served now on MLX
(internal/mlxembed, after Ollama's own): Gemma 4's text stack read both ways,
per-layer inputs, mean pooling, length 1; up to 8,192 tokens; text only, not
the pictures and sound the release can also embed. Its SentencePiece-style
tokenizer is read too (internal/tokenizer), and gave the tokenizers library's
ids on all of 25 test texts. Measured on the box against Ollama: cosine 0.99994
on average and 0.99979 at worst over eleven texts, the same 1,192 tokens, and
the same nearest neighbours; 0.19 s for the eleven where Ollama took 0.32.

### Fixed — one model file, one runner, whatever it is called

A model often has two names here: Ollama's (`ornith-1.5:35b`, a blob in its
store) and kinfer's own (`ornith-1.5-35b`, a link to that blob). A runner was
started for each name, so the swarm, which asked for one, and kinhome, which
asked for the other, would have held the same weights twice. A request now goes
to the runner already up for the same file under either name; each reply still
says the name it was asked by. On the box: the two names asked for in turn,
three times, one runner.

### Fixed — /api/version

It reports the release (`0.7.1`); it had said 0.2.0 since v0.2.0.

### Known

A prompt longer than one slot's context runs in a layout of fewer, longer slots,
and those have no prefix pool: on the laptop with `-slots 4` (8,192 tokens a
slot) the test home's 12,900-token requests reused nothing.

## v0.7.0 — 2026-11-10

### Changed — every model in a process of its own

`kinfer serve` now loads no model itself: a chat or embedding request goes, by
the model it names, to a runner of that model — this binary's serve on the
loopback, taking this server's own sizing flags — as speech and pictures
already did. A model that crashes takes only its runner down, and the next
request starts it again; one unloaded is a process ended, so all its memory
comes back; and before a runner starts, this server makes room on the GPU for
it, from what the whole GPU has free: it stops the runner unused longest that
nothing is using (the default speech models kept with `-speech keep` stay),
and once none is left waits up to five minutes for one in use to finish. A
request's keep_alive is kept here; `keep_alive` 0 stops the runner. `/api/ps`
and `/metrics` gather the runners'; a stream cut short by a runner's end ends
with an error in its own format. `-runners=false` serves chat in this process,
one model at a time, as v0.6.0 did.

Measured on the box: the regression through runners passed on four of the six
models v0.6.0's was run with there (the 0.5B, ornith, and qwen3.8-27B as GGUF
and on MLX; not Flash-Next or V4.1, nor the two-machine split), at the same
speeds (ornith 98.7 tok/s, 0.5B 289.8); decode
through a runner 94.6–104.1 tok/s against 95.7–99.1 in-process; a runner
killed with -9 mid-reply answered by the next request 2.3 s later; ten unloads
and reloads, every reply right and no runner left. With the budget pretended
at 45 GB (`KINFER_RUNNER_BUDGET`, for testing): ornith in, then qwen3.8-27B
stopping it; ornith asked for while qwen3.8 wrote a long reply waited for it,
then stopped it; a picture stopped ornith. A chat model sizes its context and
prefix pool to the memory it finds, so a second one usually stops the first,
as one model at a time did before.

On the laptop (M5 Max, 48 GB): the same regression passed with four models
(ornith, qwen3.8-27B on MLX, Flash-Next and V4.1 with experts on the SSD).
Against v0.6.0, back to back in the order new, old, old, new with no flags,
the replies were the same and so were the speeds: ornith 3,300 tok/s on a
7,470-token prompt and 136 tok/s decoding; Flash-Next 987 tok/s on the prompt,
its experts placed the same, its GPU cache 106 MiB smaller. With ornith and
both speech models loaded, a picture found 4.8 GB free of the 19.6 GB it
needed and stopped ornith; a chat runner killed mid-reply ended its stream with
an error, and the next reply came 3.9 s later.

### Added — pictures, behind the same address

`POST /v1/images/generations`, as OpenAI has it, on a Mac: Z-Image-Turbo
(Tongyi-MAI, Apache-2.0), a port of mflux's to Go on the MLX library
(internal/mlximage) — the Qwen3-4B text encoder read to its second-to-last
layer, the 6B single-stream transformer, nine flow-matching steps and Flux's
VAE decoder — reading mflux's own saved weights
(`kinfer pull mflux-community/z-image-turbo-mflux-q8`, 11.0 GB, 8-bit). Like
speech it runs in a runner of its own behind the chat address, started on the
first request and stopped once idle; the default is fetched on first use. The
answer is PNG, base64 (`b64_json`, or a `data:` URL for `url`); `size` from
256 to 2048 a side in multiples of 16, `n` up to 4, and kinfer's `seed` (the
same seed, the same picture) and `steps`. Checked against mflux on the M3
Ultra: the tokens, the caption's features, the schedule and the starting
noise the same, the decoder within 1e-5 on mflux's latents, and the pictures at
40.5–40.8 dB PSNR from mflux's own; 1024×1024 in 30.9 s (29.7 s of
denoising, mflux's the same 29.7 s), 1024×768 in 22.5 s, 10.8 GB resident and
an 18.4 GB peak. On the laptop's M5 Max, 1024×1024 in 13.1 s (11.2 s of
denoising) and 1024×768 in 9.6 s. Each stage's intermediates are freed before it is evaluated:
held, a 1024×1024 picture's attention took a runner to 84 GB.

### Added — the Python audio service's addresses

`-speech-addr asr=:8100,tts=:8101,design=:8102` has kinfer answer each kind of
speech at an address of its own, in localkin-service-audio's API
(`/transcribe`, `/synthesize`, `/voices`, `/health`) as well as OpenAI's, in
the model this server chooses for the kind and whatever model a request names,
as the Python service ignored it — so programs and devices written against
those ports keep working once the Python services stop. On the box it replaced
all three; Pixelle's and OpenMontage's calls, and a device's, answered as
before.

### Added — Qwen3-TTS VoiceDesign

A voice told in words: `instructions` (`instruct` at `/synthesize`) describe
the voice, with no voice by name (`Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`,
2.9 GB). Against mlx-audio: the prompt the same bits, the decoder within 1e-6,
the talker's logits cos ≥ 0.99996 fed the same codes, one near tie in 73
frames drawn otherwise (the two MLX builds' rounding); 4.3–4.6× real time
against mlx-audio's 4.6×. A 104-character narration: 22.6 s of speech in 5.0 s,
heard back word for word.

### Changed

With `-speech keep`, only the default speech models stay; a picture's runner
and any other speech model's stop once idle as long as `-keepalive`.
`kinfer pull speech` fetches hearing and speaking only.

## v0.6.0 — 2026-11-10

### Added — one address for chat and speech

`kinfer serve` answers `/v1/audio/transcriptions`, `/v1/audio/translations`,
`/v1/audio/speech` and `/v1/audio/voices` at the chat model's own address. Each
speech model runs in a process of its own, a runner as Ollama runs each model,
started on the first request for it and stopped once idle as long as
`-keepalive`: the chat model stays loaded, the three can be used at once, and a
speech model that crashes takes only its runner down. A request naming a speech
model of the right kind gets it; any other name (whisper-1, tts-1, "tts") gets
this machine's default, which is fetched on first use (`-auto-pull`, on): the
request is answered 503 with how far the fetch has got and a Retry-After.
`-speech keep` starts the defaults with the server, loaded; `-speech off` serves
speech in the server's own process as v0.5.0 did. `kinfer pull speech` fetches
the defaults. `/api/ps` lists the speech models. See docs/serving.md.

### Fixed — the same seed gave different speech

MLX keeps its global PRNG key for each thread, and Go moves work between
threads, so the draws followed whichever thread's key they met. Each answer now
keeps a key of its own, split before each draw as MLX's own sequence is, so
`seed` (new, kinfer's) gives the same speech every time, and the draws follow
it as mlx-audio's follow `mx.random.seed`.

### Changed — an answer that talks on is bounded, and said again

Qwen3-TTS 0.6B now and then talks on past its text, mlx-audio's as much as
kinfer's, and would go on for minutes before its 4,096 frames ran out. An
answer is now bounded by its text — 2 s and three times its ordinary reading —
and one that reaches the bound is said again once with another seed (cut, if it
is streamed). Over 30 seeds of an English sentence, 3–5 s to say, the longest
came back at 9.7 s where three had run to the test's 12 s.

## v0.5.0 — 2026-11-09

### Added — text to speech

`POST /v1/audio/speech`, as OpenAI has it, on a Mac: Qwen3-TTS CustomVoice
(`kinfer pull mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`), a port of
mlx-audio's to Go on the MLX library (internal/mlxtts) — the talker, its code
predictor and the speech tokenizer's decoder. Nine voices, ten languages,
`instructions` for the tone; wav or pcm; `speed` by time-stretching; streamed
(`stream_format` audio or sse) with the first audio out in 0.09 s. Answers the
Python service's clients as it does: a voice it lacks in the nearest or the
default, the language told from the script. Checked against mlx-audio: the
prompt and the first logits the same bits, the decoder within 1.5e-6, then the
two MLX builds' rounding (cos 0.9997 fed the same codes); four sentences six
times each, heard back by Qwen3-ASR, word for word on both. On the M3 Ultra
5.2× real time against Python's 5.1×, 1.5 GB against 2.4. `GET
/v1/audio/voices?model=` lists the voices. `kinfer pull` fetches the
speech_tokenizer/ folder beside the model (1.8 GB in 94 s on the box, checked).
Speech is MLX only: on Linux and Windows there is no text to speech yet. See
docs/serving.md.

### Added — speech to text

`POST /v1/audio/transcriptions`, as OpenAI has it, for a model whose projector
hears: Qwen3-ASR's GGUF with its audio mmproj (`kinfer pull
ggml-org/Qwen3-ASR-1.7B-GGUF:Q8_0`). The recording goes where a picture would,
through libmtmd, and the reply is the transcript; the chat path is unchanged.
wav, mp3 and flac; `language` (a code or a name) starts the reply as Qwen3-ASR
is told one, `prompt` is its context, and `response_format` is `json`, `text` or
`verbose_json`. Run it as its own server, sized by hand (`kinfer serve -addr
:11591 -slots 1 -ctx 16384 -prefix off`, 2.6 GB). On the M3 Ultra, against the
box's Python service with the same model at 8 bits: the same words in six
recordings of seven, the same time on short ones, 5–20% slower on long ones
(3.20 s against 2.65 s for 163 s of Chinese), 1 GB less memory.

On a Mac the same model also runs on MLX (`kinfer pull
mlx-community/Qwen3-ASR-1.7B-8bit`): a port of mlx-audio's Qwen3-ASR to Go on
the MLX library (internal/mlxasr), checked against it step by step — the
features within 1.4e-4, the encoder within 1.3e-6, the transcripts word for
word. 163 s of Chinese in 2.88 s against Python's 2.64, 54 s in 0.96 against
0.91, short ones the same; 2.6 GB against 3.4. A 16 kHz WAV is read in Go, other
recordings through macOS's afconvert. `kinfer pull` and `kinfer can-run` know
the MLX conversion, which ships vocab.json and merges.txt in place of
tokenizer.json.

`stream=true` streams a transcript as OpenAI does, without the language the
model opens with. `POST /v1/audio/translations` gives it in English: Qwen3-ASR
does not translate, so the transcript goes to a chat model on another server
(`KINFER_TRANSLATE_URL`, `KINFER_TRANSLATE_MODEL`).

SenseVoice Small runs on MLX too (`kinfer pull mlx-community/SenseVoiceSmall`),
on its own or beside Qwen3-ASR (`KINFER_EMOTION_MODEL=SenseVoiceSmall`), adding
`emotion` and `audio_events` to the transcript as the Python service's
`--emotion` does. Checked against mlx-audio — the same token at every position
of nine recordings — with the model's am.mvn applied, which mlx-audio leaves
out; on six recordings spoken happily, sadly and angrily it told five, and gave
the same emotion as FunASR on all six. See docs/serving.md.

## v0.4.1 — 2026-11-09

### Fixed — a model with experts on the SSD answered wrongly after it was reloaded

A model too big for the GPU keeps some of its experts in the file and reads them
as they are needed, looking up where in the file a mapped address comes from. That
table lasted as long as the process. The model unloads after five idle minutes,
and when the next question loaded it again, in the same process, its files could
land at other addresses, and the old entries sent every read to the wrong place
in the file. The weights it read were wrong, and the reply came back garbled or
empty. DeepSeek-V4.1 on a MacBook Pro did this after a pause; loading another model
first did it as well. It has been there since v0.1.0, and went unseen because each
test had started a new process for each model.

Each new MoE cache now forgets the table and looks up what it reads again (patch
0008, libraries b11998k8). On the MacBook Pro, V4.1 loaded three times in one
process, then Flash-Next, then V4.1 again, gave the same reply word for word each
time, where v0.4.0 gave an empty one from the second load on. A model that fits on
the GPU never used this table.

### Fixed — a Mac lending its GPU was given a layer too many

A model split across two Macs keeps room on the peer for a pass of activations and
2 GiB. A pass of DeepSeek-V4.1 took 4.6 GiB where the estimate said 2.7, so a
MacBook Pro with nearly all of its 40 GiB free was given a layer more than it had
room for, and paged its GPU on every token: V4.1 on the Mac Studio, borrowing the
laptop with the Studio's other services running, decoded at 4 tok/s. The
activations now count twice. Layers 0–9 go to the laptop rather than 0–10, and the
same run decodes at 20–21 tok/s. With the laptop as the master, V4.1 and Flash-Next
stay within a few percent of what they were.

## v0.4.0 — 2026-11-09

### Added — GPUs on Linux and Windows, as an experiment

On a machine with a Vulkan driver kinfer fetches llama.cpp's Vulkan backend the
first time a model loads (25–37 MB, once, checked against a pinned sha256, into
`~/.kinfer/gpu`). A backend is tried first in a process of its own, because a
driver that crashes takes the process with it. A model runs on the GPU if its
weights fit with 2 GiB to spare, and otherwise on the CPU, and one that fails to
load on the GPU is tried on the CPU. `KINFER_GPU=off` leaves all of it out.

CUDA and ROCm are hundreds of megabytes, so they are fetched only when asked for:
`kinfer gpu install cuda` says the size and asks first, and picks the CUDA build
the NVIDIA driver can run; `kinfer gpu install rocm` needs AMD's ROCm 10 on Linux
and brings its own runtime on Windows. `kinfer gpu` shows what is fetched and in
use, and `use` and `remove` change it. A backend that does not start, or finds no
GPU, is left unused.

**None of this has run on a real graphics card.** On Linux, x86-64 and arm64, CI
serves a real model through Mesa's software Vulkan driver with the right answers,
and fetches CUDA and ROCm on a machine with no card to see that each arrives whole
and steps aside. Windows has loaded none of the GPU backends. Reports are welcome.

### Added — kinfer selftest

`kinfer selftest [model]` runs a small model on the GPU and on the CPU, each in a
process of its own, and says whether the answers make sense, in a form to paste
into an issue. A broken forward pass goes on producing tokens at full speed and
looks fine until they are read. It works on a Mac too.

Nothing changes in how models run on a Mac.

## v0.3.0 — 2026-11-09

### Added — Linux and Windows, on the CPU, as a preview

kinfer is developed and tested on the Mac. This release adds packages for Linux
and Windows, x86-64 and arm64, built on llama.cpp's own CPU libraries and
carrying them in the one binary, as on the Mac. Linux installs with the same
`curl … install.sh | sh`, Windows with `irm … install.ps1 | iex`, and both with
`pip install kinfer-ai`.

Each passes the test suite and serves a real model (Qwen2.5-0.5B: four APIs,
three at once, a repeated prompt reused, `pull`, `run`, `stop`) on real machines
in CI, installed from its package; on Linux that is a container with no other
libraries. On four-core CI machines the 0.5B model decodes at 40–65 tok/s. Larger
models have not been tested on the CPU and there is no GPU support yet: reports
are welcome, and so is help with it.

Linux needs glibc 2.34 or newer on x86-64 and 2.38 on arm64, so not Alpine.
Windows needs the Visual C++ runtime. MLX models, reading experts from the SSD,
splitting a model across machines and `kinfer install` are Mac only.

### Fixed — a repeat penalty that a request asks for now applies

`llama_sampler_init_penalties` has taken `n_vocab` as its first argument since
llama.cpp b11998, and kinfer still passed the four it took before. On macOS that
left the window at zero, which turned the penalty off: a GGUF model has ignored
`repeat_penalty` in every release since that upgrade. It now applies. The other
bindings to llama.cpp were compared with its headers and match.

### Changed — the repeat penalty is off unless a request asks for it

The default was 1.1 and is now 1.0. A GGUF model answers as it did, having had
no penalty in practice: on the box Flash-Next, Qwen3.8-27B and ornith give the
same replies and speeds as v0.2.6. **An MLX model does change**, since kinfer's
MLX backend applied the 1.1 itself. Drafted tokens are no longer turned away by
it, so Qwen3.8-27B on MLX goes from 56 to 65 tok/s on the same five prompts. Code
and tool calls repeat tokens on purpose, which is the reason to leave it off.

### Changed underneath

purego 0.11.3, which every call into llama.cpp goes through. On the box against
v0.2.6: Flash-Next 1 to 16 at once 42.1→150.9 and 42.4→152.2 tok/s, V4.1 and
ornith as before, pictures in all four dialects, embeddings at cosine 1.0 to
Ollama, and the swarm's 214-tool seats with no leaked calls and no errors. A model
split across two machines was not retested for this release.

## v0.2.6 — 2026-11-07

### Added — function-xml tool calls held to their form and their schema

A GGUF model that writes its calls in the XML form (Qwen3.5 and later, and
what is built on them) now writes each `<tool_call>` outside its thinking under
a grammar made from the request's tools: a well-formed, closed call; a declared
parameter that says integer, number, boolean, enum, object or array reads as
one; nothing after the last call but the end of the turn. Other values, a
file's contents among them, are free as long as they do not close the
parameter early, and so are functions and parameters the request did not
declare, which the caller goes on reporting: a grammar of declared names only
would have turned the 4 calls to a missing `bash` or `shell`, in the 19,813 the
LocalKin swarm logged, into calls to something else. Every one of the swarm's
241 skills gets through with values of each kind its schema allows. Tokens are
drawn as before and drawn again under the grammar only when it rules one out,
as llama-server does: 0.1 ms a token, 9–25 ms for one drawn again; making the
grammar costs a request 4 ms with 57 tools. A value may not run into
`</function>` or `</tool_call>` either, so a model that leaves a parameter
open is led back to `</parameter>` rather than writing past its turn.

Played the way the box's 22 swarm seats play it — their prompts and tools,
214 tools for the family seats, a canned answer to each call, up to four
rounds — on the box and the laptop, the grammar on and off:

| | calls | invalid, on / off | tokens drawn again | decode, on / off |
|---|---|---|---|---|
| ornith-1.5:35b, box | 78 / 72 | 6 / 7 | 0 | 120 / 112 tok/s |
| ornith-1.5:35b, laptop | 82 / 87 | 3 / 8 | 0 | 138.4 / 138.6 |
| ornith, laptop, 8 at once | 80 | 5 | 0 | |
| qwen3.8-27b, box | 107 / 105 | 0 / 2 | 2 | 52.2 / 53.1 |
| Flash-Next, laptop | 95 / 108 | 4 / 10 | 7 | 9.5 / 9.8 |
| qwen2.5-0.5b (Hermes), box | 8 / 8 | 0 / 0 | not held | 311 / 309 |

What is left invalid with the grammar on is a call to a function the seat
does not have (`bash`, `weather`, `lean_verify`), passed through as before.
What it changed: qwen3.8-27b's and Flash-Next's `"hourly": True` and `False`,
read as strings, became booleans; told to break the schema, ornith wrote
`"unit": "K"`, `"days": "到月底"`, `"hourly": "要"` and a comma-separated
string for an array, and under the grammar `celsius`, `14`, `true` and a JSON
array. Two runs with the grammar off differ from each other as much as on
from off (11 of 13 cases on fresh servers, greedy), so replies that differ
are not the grammar's doing; the tokens it drew again are. Hermes, DSML and
MLX models are unchanged; `KINFER_NO_TOOL_GRAMMAR=1` turns it off.

### Added — `kinfer launch pi`, `hermes` and `openclaw`, untested

Written from each agent's documentation (2026-11-07) and not run against kinfer.
Pi and Hermes Agent take their provider only from files, so each is pointed, by
`PI_CODING_AGENT_DIR` and `HERMES_HOME`, at a directory of kinfer's under
`~/.kinfer/agents` holding kinfer as its one provider, the model and its context
window; their own `~/.pi` and `~/.hermes` are neither read nor changed. Pi picks
the model by a name without colons, which its `--model` reads as a thinking
level. OpenClaw runs as a background service of its own, so `launch openclaw`
prints the configuration to merge and runs nothing. launch installs none of
them.

### Added — what agents read to fit themselves to a server

- `/api/show` lists `tools`, `thinking` and `vision` among a model's
  capabilities, from its chat template (now read from the GGUF header) and its
  projector; `model_info` gives its architecture and trained context; and while
  it is loaded `parameters` holds `num_ctx`, the window each conversation has,
  which Hermes sizes its history by.
- `/props`, as llama-server answers it, with that window, the slots and the
  template's capabilities.
- `/v1/models` gives the loaded model's `context_length`.
- `/metrics`: `kinfer_prompt_reused_tokens_total`, the prompt tokens the prefix
  pool gave, and `kinfer_prefix_entries`, the pool's entries now — 0 reads every
  prompt whole, as the box's Flash-Next did for an hour with nothing to say so.

### Fixed — `-host-experts` without `-moe-cache` loads again

v0.2.5 gives experts in host memory passes of 4096, and with no GPU cache that
pass asks llama.cpp for buffers the GPU may not have: Flash-Next with
`-host-experts 36` on the MacBook Pro asked for 36.9 GiB of its 40 and failed to
load, where v0.2.4 served it in passes of 2048. A context that cannot be made
with a pass kinfer chose is now tried with a smaller one, 2048 first, before
the load fails; that one loaded with 4 slots of 32768 and 6.2 GiB to spare.

### Fixed — experts placed by hand that leave a little room are not refused

`-host-experts 24 -moe-cache 8192 -batch 2048` and `-host-experts 36
-moe-cache 20000` on the MacBook Pro left 1.7 and 2.2 GiB after the weights
and were refused ("no room for a context"), while the same flags leaving 0.0
GiB were loaded with a warning, as chosen on 11-06. Experts placed by hand are
now sized like that whatever is left, measured down to what fits, and loaded
with the warning: both answer.

### Fixed — a pool lost to the larger pass for experts in host memory

v0.2.5 reads prompts in passes of 4096 when experts are in host memory, and a
prefix pool needs the KV cache unified, whose pass reads every conversation's
cells. On the box, Flash-Next under the service's `-slots 4 -prefix auto` lost
its pool to that: every turn of an agent's conversation was read whole again,
19–41 s to a first token with 11–21k-token prompts. Before the pool's last
entry gives way — after the pool has shared the slots' cells and the expert
cache has given up what it may, as in v0.2.5 — the pass now goes back to 2048.
The same box with its services running, back to back with the same flags:
v0.2.5 kept no pool and read the 7,813-token second turn in 9.6 s; now one entry
stays and the turn is read in 0.4 s. Where v0.2.5 kept a pool, nothing changes:
Flash-Next with `-slots 4` on the MacBook Pro still reads an 11,000-token prompt
at 761–846 tok/s (v0.2.5 804–844), and DeepSeek-V4.1 placed on the box at 222.
Models whose experts are all on the GPU are untouched.

### Fixed — an Ollama MLX model served without its chat template

kinfer serves Ollama's MLX models from a directory of links to their blobs,
with the chat template written out from `tokenizer_config.json`, and stamps the
directory with the manifest. One made while that blob was still being copied
into Ollama's store read no template, was stamped all the same, and stayed so:
qwen3.8:27b-mlx, copied to the box over Thunderbolt, was served with chatml,
which does not know Qwen's tool calls — empty replies, and a call returned as
text. A directory that lacks the template its `tokenizer_config.json` carries is
now made again; on the box the same model parses its tool calls.

## v0.2.5 — 2026-11-06

### Faster — drafting on MLX: one wait a step, five drafts, a newer MLX

Ollama's qwen3.8:27b-mlx on a MacBook Pro (M5 Max), six greedy replies: 62.2 tok/s
on average before, **72.5** now (40–107), Ollama 0.40.15 78. Two changes:

- **A step waits on the GPU once.** It waited four times: for the drafts, for the
  pass that checks them, for the tokens taken from that pass, and for what was
  taken back after a draft that did not hold — and the GPU sat idle for the 4 ms
  the 64-layer pass took to build. Now the drafts are worked out while the pass is
  built, and the pass is evaluated with them, with what was taken back and with
  the tokens the reply takes: a step went from 48 ms to 43–46, and the six replies
  to 67.5, word for word the same.
- **MLX 0.32.5-0f4740d, and five drafts a step.** MLX 0.32.4 multiplies a
  quantized matrix by five rows or more at a higher cost per row than by four, so
  three drafts were the most that paid. The MLX after it has kernels for six to
  sixteen rows (ml-explore/mlx#4712): a pass checking five drafts costs little more
  than one checking three, and five drafts give 72.5 against three's 67.4. That
  MLX is not released; the libraries are what MLX's own CI built of commit
  0f4740d, the MLX Ollama 0.40.15 ships, with mlx-c 0.8.0 and a one-line patch
  (patches/mlx-c). kinfer fetches them once, 79 MB, the first time it serves an
  MLX model, into `~/.kinfer/mlx/0.32.5-0f4740d`; the 0.32.4 directory can go. On
  the M3 Ultra they are no slower anywhere: Qwen3.5-0.8B alone 117 → 145 tok/s,
  at 64 the same 2,850; Qwen3.8-27B at 4 bits alone 29 → 32, at 16 the same 105.

Measured and left as it is: choosing how many tokens to draft from the time each
number takes and how often each position holds, as Ollama does. It gained
nothing on 0.32.4 and less than a fixed five on the newer MLX, and a reply then
depends on the drafts before it: run twice, the same prompt could give two
replies. With the number fixed it gives one.

### Faster — many requests at once, with `-prefix -1`

With the prefix pool off, each conversation has a stream of the cache to itself,
and llama.cpp reads the sequences of separate streams together only while their
ids run on without a gap. As replies ended they left gaps, and every gap split
the step: Qwen2.5-0.5B on the M3 Ultra took 21 ms a step with 128 slots busy,
then 36, then 60. Now the highest busy slot moves into each one that frees up —
llama.cpp copies its stream, 12 MB for a 1024-token slot of that model — and the
step stays at 21 ms. 128 at once went from 2,467 tok/s to 4,803, 81% of what
`llama-batched-bench` gets on that machine; 64 from 1,983 to 3,405. The default,
with the pool, is unchanged at 3,029 and 2,328: the pool needs llama.cpp's unified
buffer, in which each conversation's attention reads every conversation's cells
(`llama-batched-bench -kvu` gets 3,436 at 128). docs/performance.md says when to
turn the pool off.

### Faster — a prompt read with experts in host memory, in passes of 4096

Every pass of a prompt reads the experts kept in host memory again, so the pass
those models get unasked is now 4096 tokens, not 2048: half as many passes, half
as many reads. Its compute buffer takes 2–3 GiB from the expert cache. On the
MacBook Pro (M5 Max, the GPU at 40 GiB), a 7,753-token prompt and three replies,
the replies the same word for word:

| model | prompt, 2048 → 4096 | decode, 2048 → 4096 |
|---|---|---|
| Qwen3.8-Flash-Next | 715 → 917 tok/s | 31.5–32.8 → 30.4–31.6 |
| DeepSeek-V4.1 | 129 → 235 | 13.8–14.1 → 13.4–13.9 |
| Ornith-1.5-397B | 58 → 139 | 3.04–3.56 → 3.13–3.61 |

An agent rereads long prompts more than it writes, so the 3% of decode is worth
it; `-batch 2048` takes it back. Placement keeps room for the larger pass. 8192
read Ornith's prompt at 328 tok/s but left a 9 GiB cache, which missed 98% of
the time: decode 27% slower, and 18 GiB of swap.

### Checked — a mixture of experts on MLX, end to end

Ollama's qwen3.5:35b-a3b-nvfp4 (Qwen3.5's mixture of experts, 21.9 GB) on a
MacBook Pro (M5 Max): 101–105 tok/s alone against Ollama's 123–127, a 7,753-token
prompt read at 3,772 tok/s against 3,989, and four replies at once 188 tok/s in
all against Ollama's 112, which serves one at a time on MLX. One reply of six
was Ollama's word for word; the others were for a few words to a few sentences,
then went their own way where a word was a near tie. The package has no MTP head, so nothing drafts.

## v0.2.4 — 2026-11-06

### Faster — a mixture of experts read from the SSD, on a MacBook Pro most

DeepSeek-V4.1 on a MacBook Pro (M5 Max, the GPU at 40 GiB), every layer's
experts in host memory and read from the SSD as the cache misses them, generated
14.5–18.7 tok/s instead of 11.5–16.6, the replies the same word for word. On the
M3 Ultra, whose larger cache misses less, 3% more. Two changes to kinfer's patch
of llama.cpp (0007; the libraries are now k7):

- **A miss is read straight into its slot.** On a Mac the cache is memory the GPU
  shares with the host, and the misses were read into host memory, then copied
  into it: 5 ms of the laptop's 78 a token, now gone.
- **The GPU is kept busy while a layer's misses are read.** V4.1's misses keep
  the GPU waiting most of a millisecond, forty times a token, and a GPU left idle
  that long lets its clock down: the layer after ran 0.96 ms by the GPU's own
  clock, against 0.65 with a small kernel queued on the graphs' own queue for
  most of the reads' expected time. A token came 14% sooner. Flash-Next, whose
  misses take 0.2 ms, decodes the same either way (32.7 and 32.8 tok/s on the
  laptop). `LLAMA_MOE_FILL=0` turns it off.
- `LLAMA_MOE_TIMING=1`, with `GGML_METAL_GPU_TIME=1`, logs when the model
  unloads where generation spent its time at each cached layer — the GPU, the
  reads, the gaps — and per layer the hit rate, the share of misses read ahead
  and the experts the previous token also chose. That is how the two were found.

Measured while doing this, and left as it is: reading the guessed experts into
slots of their own while the GPU computes raised the hits from 83% to 92% but
slowed the GPU beside the writes by 12 ms a token, 30% slower in all; guessing
more or fewer experts a layer, or keeping the read-ahead to fewer threads, moved
nothing beyond the laptop's run-to-run drift, the SSD already reading at about
14 GB/s; and a cache holding the experts the usage profile counts most would hit
73% of the laptop's selections where the cache that follows the conversation
hits 83%. The numbers and the rest: docs/big-models.md, "DeepSeek-V4.1 on the
laptop, and the GPU that idles".

## v0.2.3 — 2026-11-06

### Added — drafting with the MTP head, for MLX models that carry one

An MLX Qwen3.5 or Qwen3.8 model whose weights keep the multi-token-prediction
head (`mtp.*`: Ollama's NVFP4 models and the HF releases; mlx-community's
conversions strip it) now drafts with it, as the GGUF engine does. The head
reads each token beside the hidden state the model ends it with; while a reply
writes alone it guesses the next three tokens, one pass of the model checks
them all, and as many as the model would have written are kept. The linear
attention layers cannot be cut back to a shorter pass, so the kernel that
runs them also writes its state after each token of a checking pass, and a
step that keeps k of n drafts takes the state after k.

On Ollama's qwen3.8:27b-mlx on a MacBook Pro (M5 Max), six replies decoded at
38–75 tok/s against 30 without, one to four drafts giving 44, 55, 60 and 59 on
average; 724 of 1,051 drafts were kept. The replies are the model's replies:
five of six were the same word for word, the sixth the same until a near tie
141 characters in, as a pass of four tokens and four passes of one round
differently. Prompts read at the same speed. Only a reply writing alone drafts:
when a second request arrives the step goes back to one token a reply, and the
one that was drafting goes on in the batch. `-draft -1` turns it off, `-draft N`
drafts N.

A model under 4 GB of weights does not draft unless `-draft N` says so: its
step is quick beside the head's, which runs the output layer over the whole
vocabulary for every draft. On the M3 Ultra, Qwen3.5-0.8B in bf16 wrote 221
tok/s alone without drafts and 141 with three. Where between it and the 27B
drafting starts to pay is not measured.

### Faster — many requests at once, and prompts where memory is left

- **Sampling shared out across the cores.** Each step's slots were sampled
  one after another on one thread, and at 128 slots that was a third of a step:
  8 ms of 24 on a MacBook Pro (Qwen2.5-0.5B, 152k-token vocabulary). A step
  with eight slots or more now samples them all at once, one sampler per slot
  as before, and harvests them in slot order. 128 at once went from 3,350 to
  4,259 tok/s greedy and from 3,183 to 3,829 at temperature 0.7; 32 at once
  from 1,959 to 2,250.
- **A larger pass where the context leaves room.** The micro-batch is chosen
  before the context exists, by Ollama's rule, which is cautious: on the M3
  Ultra, Flash-Next's model and cache came to 71.9 GiB of the 77.7 free, so it
  read prompts 512 tokens a pass, and the context then left 5.8 GiB. Where the
  context leaves twice the 2 GiB headroom, a pass twice as large is tried, and
  kept if the headroom is still there; the `batch:` line says so. On ornith-1.5:35b
  an 11,000-token prompt read 13–18% faster in passes of 2048 than of 512. Not
  with experts in host memory, whose pass is sized for them, nor across
  machines, where the peers' buffers grow too.
- `KINFER_DEBUG_SCHED` also prints llama.cpp's own counters each second,
  among them how often a compute graph was reused.

Measured while doing this, and left as it is: the KV cache the prefix pool needs
is unified, and attention over a unified cache reads every conversation's
cells. llama-batched-bench at 128 sequences gave 9,932 tok/s with a cache per
sequence and 5,338 unified, 256 tokens in. Without the pool (`-prefix off`) each
conversation gets a cache of its own, but llama.cpp then splits a step into as
many passes as there are gaps between the conversations still writing, and a
step took 47 ms instead of 22. For agents the difference is small: eight
ornith-1.5:35b conversations with 7,800-token prompts each took 42.6 ms a step
either way within 10%, the experts' weights being most of a step. And drafting
with the MTP head costs about 5% of a prompt, which the head reads too
(ornith-1.5:35b: 3.80 s against 3.63 for 11,000 tokens).

### Fixed — settings given by hand that leave a model no room

- **Said at load, and again in the error, why a first pass may run out of
  memory.** On a MacBook Pro at the GPU's default limit, Flash-Next with
  `-host-experts 24 -moe-cache 8192 -batch 2048` left 0.0 GiB, and its first
  pass died on Metal's out-of-memory error with nothing saying why. A load
  that leaves under 256 MiB now says so, and names the settings that were set
  by hand: "… loaded with 0.0 GiB of the GPU left, which its first pass may not
  fit in — -batch 2048 -host-experts 24 -moe-cache 8192 set by hand may leave it
  too little; without them kinfer places the model and sizes its context
  itself". If the pass does fail, the reply's error carries the same words. It
  is not refused: the reading cannot tell a load that fails from one that runs.
  `-host-experts 36 -moe-cache 20000` on the laptop read 0.0 GiB too and
  answered, and so did DeepSeek-V4.1 with `-host-experts 32 -moe-cache 47542`
  on the M3 Ultra, at 17–19 tok/s.
- **A model that will not load is a 500, not a 404.** A load failure was
  answered as if the name were unknown; a client told 404 looks for another
  model instead of at the server. It is now 500 in every dialect, as Ollama
  answers it (`api_error` in Anthropic's), with the reason; an unknown name is
  still 404.
- **`-host-experts` without `-moe-cache` says what it costs.** It gives the host
  experts no GPU cache, so every token reads them from host memory:
  DeepSeek-V4.1 with `-host-experts 32` decoded at 3.6 tok/s on the M3 Ultra,
  and at 17–19 with `-moe-cache 47542`. The log now says so, and how to cache
  them or have the model placed.
- **A context squeezed small by a hand-set cache says so.** When `-moe-cache` or
  `-host-experts` leaves a conversation less than 32,768 tokens, the log says
  how many fit and that a longer prompt is refused.

### Changed — the slot counts to use

`-slots` said to use 8, 32, 64 or 128 and never 12–16. That was llama.cpp
b11175's curve. With b11998, the libraries kinfer embeds, throughput climbs to
16 and then falls: past sixteen sequences the step time more than doubles, and
it takes 48 to 64 to pay for it. On an M3 Ultra 32 at once gave 2,505 tok/s
against 3,494 at 16, on an M5 Max 24 gave 2,464 against 4,269
(`llama-batched-bench`, Qwen2.5-0.5B). The advice is now 8, 16, 64 or 128,
never 17–48, and docs/performance.md has both machines' curves.

### Added — MLX MoE models of Qwen3.5's architecture, not yet checked whole

`qwen3_5_moe` models load on MLX: each layer's experts as MLX's SwitchGLU
keeps them, stacked (`switch_mlp`, as mlx_lm.convert and Ollama write them),
the router's softmax in float32 and its top k scaled to sum to one, gathered
matmuls for the experts (`gather_qmm` for quantized ones, NVFP4 among them),
sorted by expert once a pass has 64 token-expert pairs, and the gated shared
expert. Ollama's qwen3.5:35b-a3b-nvfp4 and ornith-1.5:122b-nvfp4 are such
models. The block is tested against a reference written out in Go, expert by
expert, both gathered as it comes and sorted, and `gather_qmm`'s binding
against `quantized_matmul` of each expert alone; no MoE model has been served
end to end yet. Hugging Face's own files keep the experts as `gate_up_proj`
or one by one, which is not read: `kinfer can-run` and `kinfer pull` say so
from the repo's index of tensors before anything is downloaded (Qwen/Qwen3.5-
35B-A3B is refused, mlx-community/Qwen3.5-35B-A3B-4bit accepted), and the
loader says an MLX conversion is needed. Qwen3.8-Flash-Next is another architecture (`qwen4_exp`) and stays
on llama.cpp.

## v0.2.2 — 2026-11-05

### Added — Ollama's MLX models, NVFP4 among them, read where they lie

Ollama 0.40 keeps an MLX model as no single file: its manifest names a blob for
each tensor, a safetensors file holding it and its scales, and blobs for
config.json, tokenizer.json and the rest by their names. kinfer listed only
Ollama's GGUFs, so `kinfer list` did not show one. Now it does (`qwen3.8:27b-mlx
… ollama, MLX`) and serves it: for each such model kinfer keeps a directory of
symbolic links to the blobs, under the names they stand for, in
`~/Library/Caches/kinfer/ollama-mlx`, with the chat template Ollama keeps in
tokenizer_config.json written out beside them. Nothing is copied, nothing of
Ollama's is touched, the directory is made again when the manifest changes, and
`kinfer rm` points at `ollama rm`.

Ollama's are NVIDIA's NVFP4, as ModelOpt exports it: 4-bit floats eight to a
uint32, an FP8 scale for each 16 and one scale for the whole weight. The MLX
engine now reads that alongside bf16 and MLX's own 4-bit: MLX's nvfp4 product,
then the weight's own scale, in float32 (MLX's dequantize takes a global scale
too, but means by it the weight's largest value, 2688 times the stored one).

Measured on a MacBook Pro (M5 Max), Ollama 0.40.14's qwen3.8:27b-mlx from the
same blobs, back to back, greedy:

| | Ollama | kinfer |
|---|---:|---:|
| decode | 53–99 tok/s, with MTP drafting | 30.2–31.0 tok/s |
| a 7,753-token prompt | 794 tok/s | 662 tok/s |
| memory | 20.0 GiB | 21 GB |

The replies are the same answers, word for word for their first 5 to 196
characters and then in other words, as two engines' rounding makes greedy
replies part; a misread scale gives no sense at all. Ollama drafts with the
model's MTP head and kinfer's MLX engine does not yet, which is most of the
decode difference: 31 tok/s is about what the laptop's memory gives a 27B dense
model reading every weight once a token.

## v0.2.1 — 2026-11-05

### Fixed — two models named but for case were one name

A name was matched without regard to case, and the first match taken, newest
first. An MLX directory and a GGUF of the same weights are often named so —
`kinfer pull mlx-community/Qwen3.5-0.8B-bf16` and `kinfer pull
ggml-org/Qwen3.5-0.8B-GGUF:BF16` give Qwen3.5-0.8B-bf16 and Qwen3.5-0.8B-BF16 —
and with both in the store, both names ran the newer one, the other could not be
run at all, and `kinfer rm` of the older one deleted the newer. A name now means
the model of exactly that name, and only failing that one named so but for case;
a name that is two models but for case, and neither exactly, is refused as
ambiguous, as a prefix already was. A pull of exactly the name, unfinished, is
that model too: asked for, it says it has not finished, and `rm` takes what
arrived of it rather than the finished model beside it. A name that matches one
model but for case finds it as before.

## v0.2.0 — 2026-11-05

### Added — MLX: a directory of safetensors served in Go, no Python

A model can now be a directory as Hugging Face publishes it: config.json,
tokenizer.json and safetensors. kinfer serves such a directory on Apple's MLX,
and a GGUF on llama.cpp as before. The server makes no difference between them:
the same four dialects, the same chat template read from the model, and the same
thinking kept apart from the answer. The MLX engine is Go over mlx-c, through
purego, as llama.cpp is bound: no cgo and no Python.

- **What it runs.** Qwen3.5 dense (`model_type` qwen3_5), at bf16 or MLX 4-bit,
  such as the mlx-community conversions. Checked against mlx-lm layer by layer
  on Qwen3.5-0.8B (bf16 and 4-bit) and Qwen3.8-27B 4-bit: the same hidden states
  at every layer, the same greedy tokens. The tokenizer reads tokenizer.json; it
  matched the tokenizers library on 537 texts, 36,390 tokens.
- **Getting one.** `kinfer pull mlx-community/Qwen3.8-27B-4bit` pulls a repo
  with no GGUF and a model_type MLX serves as a directory of the store, named
  for the repo. It takes the weights and the files that describe them, goes on
  where it stopped, and checks the weights against Hugging Face's sha256.
  `kinfer list` shows it as "kinfer, MLX". It resolves by name or prefix, and
  `kinfer rm` deletes it whole. `kinfer can-run` sizes such a repo against the
  machine, and for a safetensors repo MLX does not serve, it now gets as far as
  suggesting GGUF conversions; it used to stop with an error.
- **The libraries.** MLX's libraries (78 MB) are not inside kinfer. The first
  MLX model fetches them once from this repository's mlx-0.32.4 release into
  `~/.kinfer/mlx/0.32.4`, checked against a pinned sha256. They need macOS 26.2.
- **Serving.** Replies run batched, one forward pass for all. They are sampled
  on the GPU in llama.cpp's order, repeat penalty to temperature, and the next
  step starts before the host reads the last one's tokens. A finished reply's
  row is reused in place. Each row's attention reads its own columns only, so
  one long reply does not make the others read its length. Recent prompts are
  pooled as the GGUF engine pools a model with recurrent state: an agent's next
  turn and its next conversation skip what they share. MLX's buffer cache is
  capped at 4 GB; unbounded, it had held 76 GB.
- **Measured** on the M3 Ultra. Qwen3.5-0.8B, the same bf16 weights both ways:

  | | kinfer on llama.cpp | kinfer on MLX |
  |---|---|---|
  | 1 / 4 / 16 / 64 replies at once | 201 / 576 / 1,160 / 1,031 tok/s | 217 / 671 / 1,270 / 2,833 |
  | memory | 15 GB | 8.3 GB |

  Qwen3.8-27B, Q4_K_M on llama.cpp against MLX 4-bit:

  | | llama.cpp, no drafts | llama.cpp, MTP | MLX |
  |---|---|---|---|
  | greedy decode | 31.0 tok/s | 35.7–51.3 | 36.0 |
  | a prompt of 11,245 tokens | 36.8 s | 39.2 s | 35.0 s |
  | 1 / 4 / 16 at once | 31 / 63 / 130 | 40 / 87 / 121 | 35 / 83 / 105 |

  With a 9,000-token system prompt, a 27B agent's second and third turns, and
  its next conversation, read their prompts in 0.28–0.37 s instead of 28 s.
- **Not yet on MLX:** a mixture of experts, MTP drafting, pictures, embeddings,
  and architectures besides Qwen3.5. A GGUF of any of them runs on llama.cpp.

### Changed — a Qwen model thinks as its template says, as in Ollama

Qwen's templates have the model think by ending the prompt in `<think>\n`.
Qwen3.8's do it unless told not to; Qwen3.5's only when told to. kinfer renders
templates through llama.cpp's C API, which runs no Jinja, and the opener was
left out. So qwen3.8-27b answered without thinking even when a request asked
for it, and Qwen3.5 on MLX almost never thought.

Requests are now rendered as Ollama renders them, from the model's default:
- a request that says nothing gets what the template does by default;
- one that asks to think gets the opener: `think: true` or a level, OpenAI's
  `reasoning_effort`, Anthropic's `thinking`, the Responses API's
  `reasoning.effort`;
- `think: false` gets the empty think block, as before.

On the box, characters of thinking with think absent / true / false:
- qwen3.8-27b: 360 / 360 / 0, where it was 0 / 0 / 0;
- ornith-1.5-35b: 508 / 508 / 0, unchanged (it writes its own `<think>`);
- Qwen3.5-0.8B on MLX: 0 / 1,474 / 0.

Ollama 0.40.11 defaults qwen3.8 to thinking (medium), so this matches it. **A
client of a Qwen3.8 model that sent no `think` now gets thinking**, and with it
longer replies; it should send `think: false` (or Anthropic's thinking disabled,
or a reasoning effort of none) to answer straight away as before.

### Fixed — a kinfer built without cgo died on its first BF16 GGUF

Metal compiles llama.cpp's kernels at the language version of the SDK the
executable says it was linked against, not the one the Mac has. Go's own
linker, which links kinfer when cgo is off (`CGO_ENABLED=0`, or a Mac without a C
compiler), says macOS 12, and that gets Metal 2.4. Below 3.1 llama.cpp's kernels
leave out their bfloat versions while the GPU still reports bfloat, so the first
BF16 matrix product found no kernel (`kernel not found in any metal library`)
and the server crashed. The released binaries come from the system linker and
were not affected. The patched libraries (`scripts/build-libs.sh`, now
`darwin_arm64_b11998k6`) ask for what a current SDK gets, 4.0 on macOS 26,
however kinfer was linked. On the box, Qwen3.5-0.8B BF16 now gives the same
reply from either build. The release set that `scripts/fetch-libs.sh` takes is
not patched.

## v0.1.10 — 2026-11-04

### Added — `kinfer launch`: Claude Code or Codex on a local model

`kinfer launch claude` and `kinfer launch codex` start the agent talking to
kinfer: to the daemon `kinfer run` uses, started if nothing answers, or with
`-host` to a kinfer on another machine. With no agent or no `-model`, launch
asks. The model is loaded before the agent starts. The agent is told how long a
conversation can grow, so it compacts its history before kinfer would have to
drop turns. Claude Code is given the model under every name it asks for one, so
no background request asks for a claude-haiku the server does not have. Codex is
given a provider on its command line. Nothing is written to either agent's
configuration. From the Mac mini, against ornith-1.5:35b on the box, each wrote a
word counter and its tests and ran them: Claude Code in 37 s, Codex in 15 s.

`/api/ps` says the context a loaded model gives each conversation
(`context_length`, as Ollama's does) and the most one can have (`context_max`).

### Added — what Claude Code and Codex ask for besides a reply

- **`POST /v1/messages/count_tokens`**: Anthropic's count of a request's tokens,
  rendered the way `/v1/messages` renders it — system prompt, history and tools,
  with thinking on or off. On ornith-1.5:35b it gave 288 and 284 for a request
  whose `usage.input_tokens` came back 288 and 284.
- **`GET /v1/models/{id}`**: one model, in OpenAI's fields and Anthropic's.
- **`HEAD /api/hello`**: Claude Code's first request to a server. Every session
  used to count it as a rejected request in `/metrics`.

A Claude Code session and a Codex session through `kinfer launch` now make no
request kinfer refuses: 18 answered, none rejected.

`GET /v1/models` now lists the models borrowed from Ollama's store as well as
kinfer's own, as `/api/tags` does. On the box it listed five of the ten that a
request could name.

## v0.1.9 — 2026-11-04

### Fixed — a model with experts in host memory lost its prefix pool on a busy machine

A prefix pool needs llama.cpp's KV cache unified, and a pass over a unified
cache is sized for all of it: on the box, Flash-Next's four slots of 32768 took a
pass of 5.8 GiB, where the conversations' own caches took 2.2. Placement had
given that memory to the expert cache, so the pool was measured out and every
turn of a conversation was read whole again — with the box's other memory in
use, 12.6–13.1 s for a turn of 7,800 tokens, about 70 s for one of kinclaw's
45,000. The expert cache placement chose now gives up as much as half of itself
before the pool goes: there it gave up 2.5–3.1 GiB, kept one entry, and the same
turns took 1.0–1.2 s. Decode is slower with the smaller cache, 21.3–21.8 tok/s
on a 113-token reply against 24.2–25.1, and the three-turn conversation took 26 s
instead of 47. A cache set with `-moe-cache` is left as given, and nothing changes
where the pool already fit: the laptop's Flash-Next and Ornith-397B load as they
did.

## v0.1.8 — 2026-11-04

### Added — drafting with the model's MTP head

A model trained with multi-token prediction now drafts its next tokens with its
own MTP head — up to four a step — and reads them in the pass that reads its
next token: each it would have written itself is kept. Every token is one the
model chose — Qwen3.8-27B's replies at temperature 0 came out identical with
drafts and without, fifteen of fifteen — and they come sooner. On
the box, Qwen3.8-27B wrote one conversation's reply at 37 to 69 tok/s instead of
31.5, least on prose and most on code; on llama-server's own three prompts it
accepted exactly the drafts llama-server's MTP drafting accepts (180 of 297, 189
of 264, 178 of 303) at the same speed, 51, 58 and 50 tok/s. On the decode prompts
of the comparison with Ollama, which drafts with the same head, it wrote 43 tok/s
where Ollama wrote 42 and kinfer had written 31.5.

The pass that checks drafts costs more the more tokens it carries, and every
conversation in it pays, so drafts are shared out: four each for one or two
conversations writing, two for three, one for four to eight, none past eight.
Conversations writing at once wrote more together at every count measured up to
eight: two 51 tok/s instead of 38, four 89 instead of 61, eight 111 instead of
103. Ten wrote 86 instead of 92: the snapshots below are written every pass.

A mixture of experts drafts one token a step: ornith-1.5-35b wrote 118–131 tok/s
at one instead of 108, accepting two drafts in three or more, and 78–85 at four,
accepting one in five; two at once 166 together instead of 149, four 229 instead
of 201. At one draft its replies were identical too; at two or more, three of
five took another word somewhere the model was torn, because a pass of three
tokens multiplies some weights with a Metal kernel that sums in another order.
`-draft -1` turns drafting off. It costs memory, which sizing counts: the head's own cache, and four
snapshots of every sequence's recurrent state for a hybrid model like Qwen3.8 —
with the box service's four slots of 65536, 24.7 GiB of context instead of 19.9;
with twelve slots of 4096, 19.0 instead of 8.6.
`/metrics` counts drafts and the ones accepted.

### Fixed — a conversation's every turn was read again when memory was tight

A turn reuses the one before it through the prefix pool, and nothing else. Slots
sized the way Ollama sizes one request's context can take more than half of the
memory left, and the pool was held to half: it got nothing, even sharing the
slots' cells, where it costs only a hybrid model's recurrent state per entry. On
the box, beside Ollama's ornith, qwen3.8's four slots of 32768 left the pool no
entries, and a conversation's second turn read all 7,851 tokens again: 26.2 s,
where Ollama took 0.3. The pool is now held to what the slots leave of the memory
they were sized against; the same conditions give it two entries sharing the
slots' cells, 0.3 GiB more, and the second turn takes 0.49 s.

A bare model name — `qwen3.8` — is Ollama's `qwen3.8:latest` before any model the
name begins, as Ollama's clients mean it; on the box it was refused as ambiguous
beside Qwen3.8-Flash-Next and qwen3.8-27b.

### Added — pictures, for a model with a projector

A vision model now reads pictures: Ollama's `images`, OpenAI's `image_url`
content parts, Anthropic's `image` blocks and the Responses API's `input_image`,
sent as base64 or a `data:` URL — a link is refused, not fetched. The projector
it reads them through loads with it: Ollama's own layer for a model borrowed from
its store (ornith-1.5:35b and qwen3.8 on the box), or `NAME.mmproj.gguf` beside
one of kinfer's, which `kinfer pull` now fetches with a vision model (F16
first) and `kinfer rm` removes with it. Its memory is counted with the model's
when the context is sized.

A prompt with pictures is read whole when its turn comes, alone — the other
conversations wait the second or so it takes — and the prefix pool, which knows
a prompt by its tokens, neither lends to it nor keeps it. A picture for a model
without a projector is refused (400), as Ollama refuses it: it used to be said
in words, and told that a picture was there, Qwen's 4B described one it had not
seen. OpenAI content
sent as an array of parts was refused before, text parts too; it reads now.

On the box, ornith-1.5:35b from Ollama's store said a picture split red and blue
was "red on the left half and blue on the right half", as Ollama's own copy of
it did, in both dialects; two pictures in one message were named in order; and
a 97-token prompt with its picture was read in 0.22 s.

### Added — embeddings, beside the chat model

An embedding model — nomic-embed-text, bge-m3, EmbeddingGemma, Qwen3-Embedding:
a GGUF whose header names a pooling, or an encoder like BERT's — now makes
vectors over Ollama's `/api/embed` (unit length; `truncate`, `dimensions`),
Ollama's older `/api/embeddings` (the model's own vector) and OpenAI's
`/v1/embeddings` (texts or tokens; float or base64). LocalKin's RAG and
kinhome's text lookups ask Ollama for these today.

It loads in a place of its own beside the chat model, with its own idle clock:
an agent looking up what to put in its next prompt does not unload the model it
is about to send it to — Flash-Next takes half a minute to load again. Asked to
chat, an embedding model is refused (400) before the chat model is let go;
`/api/show` says it embeds, `/api/ps` lists it, and `ollama stop` unloads it. A
text is read up to 8192 tokens, or as far as the model was trained, and a longer
one is cut there unless `truncate` is false. Pictures and sound in `input` — what
Ollama 0.40 embeds with EmbeddingGemma 2 — are refused for now.

On the box, nomic-embed-text read from Ollama's own store: kinfer's vectors and
Ollama's agree to a cosine of 1.000000 on each of four texts, Chinese among
them, from the same 69 tokens; batched or one at a time, the same; it loaded in
0.45 s, and a chat model loaded before it answered next in 6 ms, still loaded.

## v0.1.7 — 2026-11-03

### Fixed — Claude Code's every turn was read from the start

Claude Code opens its system prompt with a billing line whose `cch=` is new on
every request. The prefix pool adopts a prompt only up to its first difference,
so with that line first, each turn of a Claude Code session read its system
prompt and tools whole — tens of thousands of tokens — where it should have
read only what was new. The stamp means nothing to a model; kinfer now pins it
to `cch=00000;`, as llama.cpp's server does (#21793), and two turns reach the
model with the same system prompt.

### Added — `-kv-cache q8_0`: the attention cache at 8 bits

`serve -kv-cache q8_0` keeps K and V at q8_0, half f16's memory, and sizes for
it: the same memory holds twice the context — Flash-Next unasked on the box
goes from 8 slots × 32768 tokens to 8 × 65536. It costs decode speed at long
contexts, so f16 stays the default: with 20,000-token prompts on the box,
qwen3.8-27b went from 28.8 to 28.4 tok/s, ornith-1.5-35b from 97.4 to 92.3, and
Qwen2.5-0.5B from 259 to 222; prompts read as fast. The DeepSeek family, and a
model whose heads q8_0's blocks of 32 do not divide, stay f16. `kinfer plan
-kv-cache q8_0` shows the sizing without loading.

### Fixed — a pull that stopped started over, over one connection

`kinfer pull` fetched a file in one request over one connection, and deleted
what had arrived whenever the request failed: a dropped network anywhere in a
50 GB model meant fetching all 50 GB again, and a download still running after
six hours was cut off. A file now arrives in 32 MB pieces, eight at a time over
connections of their own (`-connections n`; HTTP/2 would carry them all over
one), written in place in `name.part` while `name.part.json` records how far
each piece has got. A pull that stops — Ctrl-C, a dropped network, a full disk
— leaves both, and the same pull fetches only what is missing. A connection
that delivers nothing for a minute is dropped and its piece asked for again,
and the CDN's address, signed for an hour, is renewed when it expires. A
network or a Hugging Face that is down is waited out for ten minutes — a Mac's
sleep not counted — and the progress line says why it waits.

Each file is hashed as it arrives and takes its name only if it matches the
sha256 Hugging Face publishes for it, and only once it is on disk whole, so
neither a resume gone wrong nor a power cut can leave something that passes
for a model. A split model now arrives with every piece; pull fetched only the
first, which llama.cpp cannot open without the rest. A link copied from the
website works too, folder and `?download=true` included.

A pull checks first that the disk can take what it still needs, and stops
before the disk is down to 2 GB: a full disk fails everything on the machine,
swap among it, and what a stopped pull fetched stays there. `kinfer list`
shows what a stopped pull left, and how far it got; `kinfer rm` deletes it. A
split model with a piece missing is shown there too, no longer as a model that
loads. Off a terminal — a log, ssh without one — the progress line comes every
half minute rather than redrawn in place.

Qwen2.5-0.5B-Instruct Q4_K_M (469 MB) on the Mac mini's Wi-Fi: one connection
averaged 22 MB/s, eight reached 35 MB/s. Stopped with Ctrl-C at 152 MB and run
again, the pull went on from 152 MB, and the file matched its sha256.

## v0.1.6 — 2026-11-02

### Fixed — a model was placed as if the GPU were its own

Metal reports only what this process has left of the GPU's budget, so kinfer
placed and sized a model as if the apps around it held nothing. On the MacBook
Pro (40 GiB for the GPU), with 8 GiB held by a browser, Teams and the rest,
Flash-Next was placed into 37.7 GiB and its first reply failed with
`Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)` — on every
try, since the reload was placed the same way.

Free GPU memory is now what the whole GPU has left, for every command: the
reading v0.1.5 gave `kinfer worker` (patch 0004), which counts every process's
allocations. The same laptop with the same apps places Flash-Next into the
32 GiB actually free, 40 of its 48 layers' experts on the SSD and a 12.9 GiB
cache, and decodes a 384-token reply at 29.9 tokens a second (30.9 placed into
all 40 GiB, the day that worked). `GGML_METAL_FREE_DEVICE=0` goes back to the
old reading. The memory line says which it is: `(the whole GPU)`.

### Fixed — a reply the backend failed under goes on

When an app takes GPU memory after the load anyway, the GPU fails under the
model and llama.cpp gives up the whole context. Queued requests were already run
again on a reloaded model; the ones mid-reply were failed, because running them
again would repeat what the client had. Now the reload is placed to fit what is
free by then, and each reply goes on from where it stopped: the replacement
prefills the reply so far after the prompt and sends only what follows. A
request reloads at most twice.

Flash-Next on the MacBook Pro with a failure injected 40 tokens into its
thinking: the same thinking and answer, word for word, after a 15-second pause.

## v0.1.5 — 2026-11-01

### Faster — the MoE cache keeps the GPU waiting less

The MoE cache stops the GPU before every layer whose experts it holds, to read
which experts the layer selected and upload the ones it lacks. DeepSeek V4.1
IQ2_XXS alone on the box holds 26 such layers, and Metal's own timestamps put
a 68 ms token at about 32 ms of computing, 19 of waiting for the SSD and 11 of
idling around the stops. Patch 0005 shortens the stops:

- Metal marks the end of each graph with an event, and the host watches for
  it rather than waiting to be woken once Metal has handled the completion:
  the host now carries on about 40 µs after the GPU's last instruction.
- The cache reads a layer's selection and writes its slot map in the memory
  the GPU shares, where each was a copy queued on the GPU, two more command
  buffers a layer to start and wait for.
- A miss nobody read ahead comes from the SSD straight into the stage, past
  the page cache.

V4.1 alone on the box, the same five requests, replies identical: decode from
14.7 to 16.1 tokens a second as placed by default, a 3,614-token prompt from
176 to 184 a second. With shorter stops more layers behind the cache pay:
`-host-experts 32 -moe-cache 47542` decodes at 17.7 and reads the prompt at
197, and 36 or all 40 layers do as well, with short prompts faster still.

Elsewhere, back to back: Flash-Next alone on the MacBook Pro, 28 layers'
experts on its SSD, from 27.1 to 30.9 a second, prompts much the same; V4.1
from the box with the MacBook Pro's GPU taking ten layers, from 18.0 to 21.5,
most of it the event (without it 18.4, with the old copies 20.2). A model
split with every expert on a GPU is unchanged: Flash-Next from the MacBook Pro
borrowing the box, 43.4 and 44.3. Placement still sizes the cache at a third
of the experts in host memory: on the MacBook Pro 32 layers instead of 28
gain 5% and 36 lose 8%, while V4.1 on the box gains 10% from 32 to 40.

Tried and left out: queueing the next layer's graph before the stop, held on
the GPU at an event until the slot map is written (the GPU took about as long
to resume from the event as to start a new command buffer: 1.5%, within the
noise), and keeping the GPU warm through the stops (2%, and the MacBook Pro
once lost most of its speed to it).

### Fixed — a worker offered GPU memory its machine's apps held

Metal reports a process's own allocations only, so a worker offered the whole
GPU budget and placement filled it: DeepSeek V4.1 IQ2_XXS served from the box,
12 layers on the MacBook Pro's 40 GiB, while its own apps held 6.5 GB more. Its
GPU paged those layers in and out every token: 2.3 tokens a second. A worker
now reports what the whole GPU has left (patch 0004, `GGML_METAL_FREE_DEVICE`,
read from IOKit): 33.9 of 40 GiB there, 10 layers, and 17.4-18.4 tokens a
second, against 13.3-16.2 for the box alone; a 3,614-token prompt at 229 a
second, against 176. The same reading keeps a laptop borrowing the box from
being given the memory a model loaded by the box's own service holds.

### Fixed — a worker turning around no longer aborts a load

ggml-rpc connects afresh whenever nothing holds its last connection, several
times a load, and aborted the process on any refusal. A worker serves one
connection at a time and resets one opened right after the last closed (6 of
300 over the cable). Refused and reset connections are tried again for two
seconds (patch 0003, and kinfer's own hello), and a connection that fails for
good says why in the log. A "no route to host" on a Mac now says it may be
Local Network privacy: on the box, a managed Mac, a kinfer started over ssh
lost the network the cable is on once that ssh session ended, while programs
in a live session still reached the worker.

The patched libraries are `darwin_arm64_b11998k5`.

## v0.1.4 — 2026-10-31

### Fixed — the REPL edits Chinese, and survives Ctrl-C

`kinfer run`'s interactive prompt left line editing to the terminal, which
erases a byte per backspace: backspacing over Chinese, three bytes a
character, left half a character on the screen and broken UTF-8 in the
request. On a terminal the REPL now edits its lines itself, in raw mode
(`cmd/kinfer/lineedit.go`): a backspace erases a character, two columns for a
wide one; the cursor keys, Home/End, Ctrl-U/K/W and history (up/down) work; a
long line wraps and redraws; a bracketed paste is one message, its newlines
kept. Piped, it reads lines as before. Driven through a real terminal with
expect: 你好世界, two backspaces, 吗 sent "你好吗".

Ctrl-C during a reply stopped it for good: the REPL shared run's one context,
cancelled once, and every turn after failed. Each reply now has its own, and
Ctrl-C stops just that one ("(stopped)"); at the prompt it clears the line, or
leaves on an empty one, as Ctrl-D does.

## v0.1.3 — 2026-10-31

### `kinfer run` borrows a GPU by itself

Lending a GPU took looking up the Thunderbolt Bridge's link-local address on
one Mac and typing it into `-peers` on the other, and again whenever the cable
was plugged in again. Now:

- `kinfer worker` listens on its end of the Thunderbolt Bridge by default (it
  listened on loopback, of no use to a worker), and answers a question broadcast
  on the bridge alone (`internal/link`). A Thunderbolt Bridge passes neither the
  limited broadcast nor multicast from a bound socket — those datagrams never
  left it — but the link-local network's own broadcast address has a route
  scoped to it.
- `kinfer run` starts its daemon with `-peers auto`. For a model that does not
  fit this GPU, the daemon asks over the cable (0.3 s), borrows the workers that
  answer, and gives them only the layers that do not fit here. A model that fits
  never connects. With no answer, or a worker lent to another machine, the model
  stays here with its experts in host memory. On the MacBook Pro, Flash-Next
  found the box's worker and decoded at 44.4 tok/s; with the worker busy it
  stayed on the laptop at 26.5.
- `-peers none`, `-peers host:port[,...]` and `KINFER_PEERS` say otherwise;
  `serve -peers auto` does what run does, and `serve` alone still borrows
  nothing. `/health` lists a daemon's peers, and a run that asks for peers
  refuses a daemon running with others and says how to stop it.
- A worker is asked for a hello, within 2 s, before llama.cpp connects
  (`llama.RPCHello`): llama.cpp waited forever on a worker serving another
  machine, and aborted the process on another protocol version.
- The patch adds TCP keepalive to RPC connections
  (`patches/llama.cpp/b11998/0003-rpc-keepalive.patch`, libraries
  `darwin_arm64_b11998k4`). A borrower that died with its connection open left
  the worker serving it, the only connection it takes, until TCP's two-hour
  keepalive; probed after 10 s of silence it freed in 15 s.

### `kinfer worker -on`, and `kinfer stop`

`kinfer worker -on user` starts a worker on the Mac at the other end of the
Thunderbolt cable, over ssh, from the Mac that will borrow it: that Mac is found
by the IPv6 all-nodes address on the bridge, which every Mac answers with
nothing of kinfer's running, its host key taken the first time
(`StrictHostKeyChecking=accept-new`), and the worker left running there in the
background. `-on user@host` names any Mac ssh reaches, `-stop` stops it, and a
release that differs from this one is said at once. It needs Remote Login there
and, for no password, this Mac's key in that one's authorized_keys.

`kinfer stop` stops the daemon `kinfer run` started — it took `pkill -f 'serve
-addr 127.0.0.1:11500'`. `/health` now says the daemon's pid; one from an older
release is found by the port it listens on, if that process is kinfer. The serve
`kinfer install` keeps running is refused, since launchd would start it again.

### `kinfer run` shows the thinking and the speed

After each reply `run` prints, on stderr, how fast the prompt was read and the
reply made — `[prompt 18 tokens in 0.25 s (71 tok/s) · reply 112 tokens at 44.1
tok/s]`, as `ollama run --verbose` and play.sh do; `-stats=false` drops it. A
reasoning model's thinking, which `run` never showed — a long think looked like
a hang, and a reply cut short by `-n` was blank — is shown in grey on stderr as
it comes, when stderr is a terminal. Piped, stdout carries the reply alone.

## v0.1.2 — 2026-10-31

### Fixed — no keep-warm beside experts in host memory

v0.1.1 kept a split's GPUs warm whatever else was loaded. With experts in host
memory that is a trap: the MoE cache waits on the GPU several times a layer, and
on the MacBook Pro a GPU kept busy by anything else made each wait long.
Flash-Next with 27 layers' experts on the host (the laptop alone, with
`GGML_METAL_WARM_MS=1000`) decoded at 1.0-1.2 tok/s instead of 29.5-31.3; the
same kernel from another process did the same, and the box lost nothing to it
with 16 layers' experts on the host. v0.1.1 never turned the window on for one
machine, so it showed only for a split that also kept experts in host memory —
the way DeepSeek-V4.1 was to run across the two. Now a model with host experts
keeps the window off for as long as it is loaded (`internal/engine/warm.go`), and
says so at load.

### A split decodes as fast as one machine: the GPUs kept warm

With a Thunderbolt 5 cable in place of the USB 2.0 one, Flash-Next split between
the box and the MacBook Pro loaded in 15 s instead of minutes and read a 7,016-token
prompt at 850-1,400 tok/s instead of 254-285, but decoded at 25-29 tok/s, about
what it did over USB. A token waits on the cable only twice (about 0.3 ms each,
the TCP stack's, on either cable). The rest was the GPUs: each waits while the
other computes its layers, and macOS lowers an idle GPU's clock within
milliseconds, so the laptop's 8 layers took 6 ms at the start of a reply and
11-14 ms twenty tokens in.

The patch (`patches/llama.cpp/b11998/0002-metal-keep-warm.patch`, libraries
`darwin_arm64_b11998k3`) adds `ggml_backend_metal_set_warm`: for a window after
each graph, a background thread runs a tiny kernel back to back on a queue of
its own (one SIMD group, about 50 µs each, one at a time), and sleeps once the
window passes without another graph. kinfer turns it on, for a second, in
`kinfer worker` and in `serve -peers`; `GGML_METAL_WARM_MS` overrides it.

Back to back on Thunderbolt 5, v0.1.0 against the patched build: the box with 8
layers on the laptop, 25.1 → 42.3 tok/s; the laptop with 17 GB of layers on the
box, 28.9 → 43.6 tok/s. The box alone decodes at 44.1. Prompts are unchanged
(850; 1,403 and 1,274). On the box alone the window is worth 4% (43.8 → 45.5),
and a single machine leaves it off.

### Where the way with experts on the SSD comes from

The README now says it: Strata, whose own engine runs a mixture of experts from
the SSD on a Mac, did first what kinfer's patch does in llama.cpp's MoE cache:
the misses read by a pool of threads, the next layer's experts guessed from its
router and read ahead, a profile of the experts generation uses filling the
cache at load. kinfer's own are the streamed prompt passes, the per-layer table
read in parallel, and the placement on one machine or several. The README
gives each step's share of the 7,753-token prompt on the MacBook Pro: 62 tok/s
with the release, 159 with Strata's three parts, 306 reading with pread, then
kinfer's own 391, 445 and 701. Decode is mostly Strata's way: 9, 23, 31.

### Measured against Ollama on the MacBook Pro

Ollama 0.40.13 runs Flash-Next IQ3_S on the laptop. As it sets itself up,
llama.cpp's own fit with 13.2 GiB of experts for the CPU, it reads the
7,753-token prompt at 17 tok/s and decodes at 7-8. Its `llama-server` given
kinfer's hand-tuned MoE settings (24 layers' experts on the host, an 8 GiB
cache, passes of 2048) reads it at 81 and still decodes at 6-8. kinfer with no
flags: 716 and 28. The same reply from all three; the tables are in the README
and docs/big-models.md.

## v0.1.0 — 2026-10-30

The first release: a binary for Apple Silicon with the patched llama.cpp
inside, installed by `scripts/install.sh`, and `kinfer version` to say which.

Built and measured on a 96 GB Mac Studio (M3 Ultra) and, for the entries on
experts in host memory and on the SSD, a 48 GB MacBook Pro (M5 Max): the numbers
are theirs, which is why they are specific.

### Across machines, the peers take only what does not fit here

With `-peers` and no `-split`, llama.cpp split a model by free memory while
sizing measured only this machine. On the box with the MacBook Pro as its peer,
that put 16 of Flash-Next's layers on the laptop with their share of a context
sized from the box's 88 GiB — 15.5 GiB of weights, 7.7 of KV cache and a 35.5
GiB compute buffer on a 40 GiB GPU; the first prompt ran it out of memory
(`kIOGPUCommandBufferCallbackErrorOutOfMemory`), the worker aborted, and the
server with it. Placement now makes the split:

- `engine.PlaceDevices`: this machine keeps what it can hold with the room
  placement keeps; each peer in turn takes the next of the first layers, only
  while the rest would not fit here, keeping 2 GiB and a pass's activations of
  its own free; what still does not fit stays in this machine's host memory —
  the routed experts of its own last layers with a GPU cache, never the layers a
  peer computes. A model that fits here gives the peers nothing, since a layer
  behind the cable costs every token a round trip.
- The split is a `tensor_split` whose boundaries fall half a layer before each
  device's first layer: llama.cpp puts layer il where il/(n+1) falls, the output
  counting as a layer, and an exact boundary could round either way. Host
  experts can start at any layer (`hostExpertOverrides(first, n)`), and the
  header now totals each layer's tensors (`LayerBytes`).
- A model split this way sizes its context and pass to the room kept, as on one
  machine. `-split` chooses by hand; `-place=false` is llama.cpp's split again.

Over the USB 2.0 cable, Flash-Next IQ3_S, the 7,753-token prompt and three
200-token replies, back to back: the box with the laptop as its peer placed
nothing on the laptop, 729 tok/s and 43 tok/s (the box alone: 748 and 43);
`-place=false` ran the laptop out of memory as above. The laptop with the box as
its peer placed layers 0-17 on the box, 258 tok/s and 23 tok/s; the laptop
alone, its experts from the SSD, 720 and 28. On this cable the SSD beats the
other machine's GPU, and kinfer still uses the peers it is given; a Thunderbolt
5 link is the next measurement.

Connecting to a worker waits half a second after kinfer's own reachability
check. The worker serves one connection at a time with a queue of one, and
ggml-rpc connecting at once was refused or reset over the cable from the laptop
to the box (`Connection reset by peer` in the hello, ggml then aborting the
server) in nearly every load in a row; with the pause, and in a plain RPC hello
sent 0.5 s after a check, it connected every time.

### A model the GPU cannot hold, placed by itself

Flash-Next on the MacBook Pro ran fastest with `-host-experts 24 -moe-cache 8192
-batch 2048 -slots 1 -ctx 16384`, numbers found by loading the model again and
again. `kinfer serve` with none of them now runs it as fast:

- The header is read for what the tensors hold: every shard's tensors, each the
  distance to the next, totalled as `WeightBytes`, the routed experts of each
  layer as `ExpertBytes`, and the vocabulary as `Vocab`. A split model with a
  shard missing counts nothing rather than part of itself.
- `engine.Place`: a mixture of experts the GPU cannot hold with room kept — for
  one conversation of 32768 tokens or what `-slots` and `-ctx` ask, a prompt
  pool entry, a pass of 2048 (its logits and activations) and the 2 GiB serve
  warns below — leaves the experts of as few of its first layers on the host as
  free what is missing plus a GPU cache of a third of them, the ratio of the
  hand-tuned configuration; the cache gets everything freed beyond what was
  missing. A dense model that does not fit, or a mixture whose shared part alone
  does not, loads as before, with a line saying why it will not fit.
- Sizing counts the MoE cache, which is allocated with each context: what the
  context may cost is what the weights left less the cache. With `-moe-cache`
  and the sizing left to serve, the cache's memory had been counted free for the
  KV cache as well. A placed model's context is sized to the room placement kept,
  and when the measured context costs more, the pool and the context give way
  before the pass, which with experts in host memory goes last.
- `-place=false` turns it off, `-host-experts` and `-moe-cache` choose by hand as
  before, and with `-peers` it stays out of the way for now. `kinfer plan` shows
  the placement, and what is left for the context after it.

Same laptop, back to back, the 7,753-token prompt and three 200-token replies:
no flags placed 28 of 48 layers' experts on the host with a 9.0 GiB cache and
read the prompt at 706 and 726 tok/s, decoding at 27-29; the hand-tuned flags
read it at 712 and decoded at 29. The same reply from both; the placed model
keeps 32768 tokens of context and a pool entry where the flags kept 16384, and
2.3 GiB free where they left 0.7. On the box nothing changes: Flash-Next (51.0
GiB of weights against 88 GiB), ornith-1.5-35b and qwen3.8-27b all fit.

### Prompts from the SSD at speed: reads with pread, a stream for large batches

The patched cache made generation fast and left prompts slow. On the MacBook Pro,
Flash-Next with 24 of 48 layers' experts on the host read a 7,753-token prompt
(the top of the README as it was then) at 73 tok/s in passes of 512 tokens, 117 in passes of
1024 and 162 in passes of 2048: every pass uses most experts of every layer, so
the host experts are read again for each, and the reads ran at 1.1-1.5 GB/s from
an SSD that does 13.2 given requests of a few MB from a few threads. The patch
(`patches/llama.cpp/b11998`, libraries `darwin_arm64_b11998k2`) now:

- reads what stays mapped from the model file with `pread` from a pool of
  threads, in requests of up to 2 MB, past the page cache for bulk reads, instead
  of copying it out of the mapping one page fault at a time — the file and offset
  are found from the mapping itself (`mach_vm_region`, `proc_regionfilename`) and
  checked against it;
- streams the experts of large batches (`llama_moe_stream`): each weight is read
  into one of three slots of host memory, the next two while the GPU computes the
  current one, in the order the copies came last time (seeded from the layers) and
  only the experts the previous pass used, then copied into the GPU's copy of the
  weight by every thread. The experts the MoE cache holds are copied from its
  banks the same way: a GPU copy for each run of them, three command buffers each,
  had cost 6.4 s of host time per prompt while the GPU waited;
- reads the rows of a per-layer embedding table the same way before each pass.
  Flash-Next's is a 27 GiB n-gram table, 16 random rows of 90 bytes per token, and
  `llama_prefetch_rows` only `madvise`d them: the CPU's `get_rows` then paged them
  in one fault at a time, 4 s for a pass of 4096 tokens, against 0.3 s now.
  DeepSeek-V4.1's table is read through the same call.

A device behind the network (`-peers`) keeps the single-threaded copies, since
its socket takes one request at a time; `LLAMA_MOE_STREAM_SLOTS` sets the number
of slots, 0 turns the stream off. Where k1 only added to llama.cpp, k2 also
changes two of its functions, `sched_copy_experts` and `llama_prefetch_rows`
(1,380 lines added, 15 replaced).

kinfer's own part: a pass may be as large as `-batch` asks, up to 8192 (the step
was capped at 2048 before), and with `-host-experts` the automatic micro-batch
starts at 2048, stepped down while the context leaves less than 2 GiB free.

Same laptop, same model, the builds run back to back on the 7,753-token prompt
in passes of 2048 with the 8 GiB cache: the release libraries 62 tok/s, last
night's k1 159, k2 705; in passes of 4096 with a 6 GiB cache k2 reads it at 740.
The 32-token reply is the same, byte for byte, from every build at every pass.
Decode is 32.3 / 31.0 / 31.3 tok/s with the 8 GiB cache (23.1 / 22.7 / 23.1 with
k1) and 28 with 6 GiB. Single runs on this laptop wander by a quarter as it
heats and swaps, so only runs side by side are compared. Three slots and not
four: the laptop had 5-6 GB in swap, and the fourth slot's 450 MB made the copies
ten times slower through the memory compressor. A model without experts is
unchanged (qwen2.5-0.5b, 172.0 / 171.4 against 166.4 / 173.7 tok/s, the same
text).

### Experts from the SSD at speed: a patched MoE cache

On a MacBook Pro (48 GB, 40 GiB for the GPU) Flash-Next IQ3_S does not fit, so
what is not on the GPU comes from the SSD. Experts placed on the host by whole
layers, 16 of 48, decoded at 4.3 tok/s, reading 130 MB a token at about
0.6 GB/s: llama.cpp copies them out of the mapped file one page fault at a time.
With a GPU cache in front (24 layers on the host, 8 GiB cache, 94% hits) the
release build managed 9.1.

`scripts/build-libs.sh` builds llama.cpp b11998 with a patch to its MoE cache
(`patches/llama.cpp/b11998/`, 675 added lines, nothing upstream changed):
misses are asked of the kernel at once and paged in by sixteen threads; the next
cached layer's experts are guessed from its router applied to this layer's input
and asked for ahead; a prompt's large batches page in what they copy the same
way; and two new calls, `llama_moe_cache_save_profile` and
`llama_moe_cache_load_profile`, keep counts of the experts generation selects.
kinfer saves them to `~/.kinfer/profiles/<model>.moeprof` when a context is
freed and fills a loading model's cache from them. The libraries go to
`darwin_arm64_b11998k1`; kinfer binds the profile calls only if they are there,
so the release set still works, without a profile.

Same model, same laptop, same 24 layers and 8 GiB: decode 19.3 / 23.2 / 22.9
tok/s over three replies (9.1 before), the prompt 103 tok/s (38), the first
reply 23.1 once a profile exists. The reply is identical word for word, and a
model without experts gives the same reply at the same speed (qwen2.5-0.5b,
176.6 and 180.1 tok/s). A read-ahead as wide as the router's choice won: half as
wide again reached 21.7, twice as wide 19.6, because the unused reads crowd the
SSD. Filling the cache from the profile was 28 s while it read all 8 GiB before
copying any — the first experts were evicted again before they were copied — and
is 5 s in 512 MiB batches. A context rebuilt for a longer prompt saves its counts
but does not refill.

### Experts in host memory, and llama.cpp's MoE cache measured on Metal

`serve -host-experts N` leaves the routed experts of a mixture of experts' first
N layers (`-1`: every layer) in host memory, mapped from the file, as llama.cpp's
`--n-cpu-moe` does; `-moe-cache MiB` gives llama.cpp b11998's MoE cache that much
GPU memory for the ones in use. A model with no experts ignores the cache rather
than failing every context on llama.cpp's "MoE cache requires a MoE model".

The first run cached 9 of Flash-Next's 48 layers: llama.cpp had repacked every
IQ4_NL down projection for the CPU (CPU_REPACK), a repacked buffer is not plain
host memory, and the cache only takes layers whose experts all are. Placing
experts on the host now turns repacking off (`--no-repack`).

Measured on the box with Flash-Next IQ3_S (48 GB of experts), decode the warm
third of three replies, a 2,116-token prompt: everything on the GPU 44.2 tok/s
and 770; 12 of 48 layers' experts on the host 29.6 and 566, with a 4 GiB cache
32.7 and 511 (89% hits); every layer's on the host 14.6 and 235, with an 8 GiB
cache 16.8 (81%), 16 GiB 19.9 (94%), and a cache holding every expert 22.2 and
60 (98%). The last says what the cache costs on Apple Silicon: half the speed of
the same weights on the GPU, at almost no misses — a host round trip at every
layer to learn the router's choice, and a prompt's large batches copying experts
out of it. It was built for VRAM behind PCIe; on unified memory host memory is
the GPU's. A model that fits keeps its experts on the GPU; one that does not puts
the fewest layers' on the host.

### Two machines, one model: `kinfer worker` and `serve -peers`

`kinfer worker -addr host:port` lends a machine's GPU over llama.cpp's RPC
backend; `kinfer serve -peers host:port -split a,b` puts the first layers of
each model there, in llama.cpp's own order (peers first, this machine's GPU
last, with the output). The worker is kinfer, not llama.cpp's rpc-server, so
both ends are always the same build; the protocol has no authentication, so it
defaults to loopback and belongs on the cable between two machines. It keeps the
tensors it is sent, by hash, in `~/.kinfer/rpc-cache`. kinfer hands llama.cpp
the device list itself rather than registering the remote devices globally,
which leaves every local memory reading as it was. ggml aborts the whole process
when it cannot connect (`rpc_dispatcher::start`), which took the first test
server down; kinfer now dials each worker itself first, so a worker that is
unreachable — a laptop asleep — is a failed load rather than a dead server. One
that accepts and then fails the handshake still aborts, in ggml.

Measured with Qwen3.8-Flash-Next IQ3_S between the box and a MacBook Pro M5 Max
over a USB-C cable that negotiated USB 2.0 (1.2 ms round trip, 343 Mb/s):

- 8 layers on the laptop: 42.6 GiB stays on the box (50.7 alone); the first
  load took 248 s, nearly all of it 8 GB crossing the cable; the second, from
  the worker's cache, 21.9 s. Decode 23.7 tok/s, a 7,016-token prompt at
  283 tok/s — against 44.5 and 666 on the box alone.
- 2 layers: decode 26.3 tok/s, the same prompt at 284. So the cost is the link,
  about 16 ms of round trips a token, not the laptop's share of the work (about
  0.6 ms a layer); prompt reading is the cable's bandwidth. Thunderbolt is the
  next measurement.

The laptop's firewall held the worker's first connections, unanswered, until
the binary appeared as permitted in its list.

Also: `llama.DeviceType` skipped ggml's IGPU, so BLAS (3) read as unknown and an
integrated GPU (2) as an accelerator. Metal reports GPU, which is why nothing
showed it.

### Sizing counts what stays in memory, not what is on disk

llama.cpp leaves one table of three architectures on disk and reads it a few rows
a token: `per_layer_token_embd`, which is DeepSeek-V4.1's 111 GB n-gram table
(the architecture new in b11998), Qwen3.8-Flash-Next's (already in b11175), and
Gemma 4's per-layer embeddings — marked `TENSOR_READ_LAZY` by their loaders and
honoured over 4 GiB. `serve` was never fooled: it sizes from what the load
measures. Everything that estimates beforehand judged a model by its file size,
and counted the table as weights.

The header now says how big the table is. `store.ReadShape` reads the tensor
offsets of every shard (a split model's table usually sits in a later one, and
in ISTA's V4.1 layout that file is a link to a shared one) and reports
`LazyBytes`; the rest is what the model holds.

- `kinfer plan` subtracts it, and says how much is read from disk.
- `kinfer fit` judges installed models by what they hold.
- `kinfer can-run`, for these three architectures, reads each quantisation's
  headers with ranged requests — growing from 256 KB, never the weights — and
  drops a group that is only a shared table published on its own, such as
  ISTA's Engram file, which it used to offer as a model to pull.
- `/api/ps` reports the loaded model's size without the table, as Ollama's does.

On the box, the ISTA Flash-Next IQ3_S (54.8 + 28.8 GB, the second file its
n-gram table) is the model the server has run since the evening it was
downloaded: 42 loads from October 11 to 28, each 51.0 GiB of model by the load's
own measurement. All that time `plan` said it did not fit, by 117.8 MB. It now
counts 51.1 GiB of weights and 26.8 read from disk. Loaded with b11998 it takes
50.7 GiB of the 77.8 GiB budget, leaving 27 GiB for 8 slots + 4 prefix x 32768
and 5.8 free; it decodes at 44.5 tok/s and reads a 7,024-token prompt at
666 tok/s.

An earlier version of this entry, and commit ea62147, said the file had sat
unused until now. It had not; the server's own log says otherwise.

### An exact prefix pool copies whole sequences, and says so

The same load went down on its first pooled prompt: b11998 gave Flash-Next an
indexed hybrid memory whose `seq_cp` aborts on anything but a whole sequence —
`GGML_ASSERT(p0 <= 0 && p1 < 0 && "partial seq_cp is not supported")` — and the
pool asked for `[0, n)`. For a hybrid model the pool only ever copies whole
sequences already (it adopts whole entries and publishes a slot at the instant
it holds exactly the checkpoint), so it now asks for `(0, -1)`, which means the
same thing to every memory. A pool for a plain transformer still copies ranges.
On the box after the fix, an agent's second turn reused its 3,338-token prompt
in 0.12 s on Flash-Next (4.76 s cold) and 0.05 s on ornith-1.5:35b (1.54 s).

### DeepSeek-V4.1 prompts, rendered by kinfer

Loading V4.1 was not serving it. llama.cpp has no V4 chat family, and matched
against the ones it has, V4.1's template passes for DeepSeek-V3's: no system
token, no think markers, no DSML. Close enough to answer, wrong in every turn.

So a model whose GGUF says `deepseek41` now gets a renderer ported from the
encoder DeepSeek ships with the model (`encoding/encoding.py` in
deepseek-ai/DeepSeek-V4.1-Flash-1017). It is checked byte for byte against
four of the five conversations DeepSeek publishes with that encoder, now in
`internal/chat/testdata/deepseek-v4.1`; the fifth hangs its tools on a system
message mid-conversation, which kinfer's API cannot say. The assertions of
DeepSeek's encoder tests are ported as well. Where its server library,
deepseek-recipe, differs from the model's own encoder, the encoder wins; the
only difference that reaches kinfer is whether a running tool loop keeps its
reasoning when no functions are declared (the encoder keeps it).

- **Tool calls in DSML**, V4.1's markup: declared in DeepSeek's words, written
  back into history as the model wrote them, parsed out of replies. The parser
  takes V4's unspaced tags too, as DeepSeek's server does, and keeps arguments
  valid JSON even when the model flags a non-JSON value as JSON. Schemas and
  non-string arguments are printed as Python's `json.dumps` prints them —
  `", "`, key order kept, `1e-05`, non-ASCII as is — because the encoder is
  Python and that is the spelling the model read in training.
- **The think block the prompt opens.** V4.1's prompt ends in
  `<｜Assistant｜><think>`, so the model writes only `</think>`. The engine
  reports the opening tag in front of the reply, and every splitter downstream
  works unchanged. Checked on a real model with the template forced: thinking
  on, the reply begins `<think>`; off, it does not.
- **Reasoning effort.** `"think": "low"|"high"|"max"` and OpenAI's
  `reasoning_effort` (where `"none"` is off) reach the engine; V4.1 states the
  budget as 50, 75 or 100, or a number from 1 to 100. Other templates ignore it.
- **Earlier thinking.** An assistant turn's `thinking`, or DeepSeek's
  `reasoning_content`, is kept and rendered where V4.1 expects it.
- `llama.ModelMeta` reads GGUF metadata, which is how the architecture is known.

### llama.cpp b11175 → b11998: DeepSeek-V4.1 and GLM-5.3-Flash load

b11175 did not know `deepseek41` (DeepSeek-V4.1-Flash, merged upstream on
October 25 as PR #31086) or `glm5-next` (GLM-5.3-Flash). b11998 knows both, and
reads V4.1's 111 GB Engram table lazily from disk: llama.cpp's default lazy
mode, which kinfer leaves alone, reads any marked tensor over 4 GiB a row at a
time instead of loading it. ISTA's GSQ-RCO sets moved to this layout on October
27 (b11870 or newer): shard 1 is metadata only and the shared Engram file is
linked in as the last shard.

`llama_context_params` gained `moe_cache_size` after `type_v`, so every field
behind it moved by 8 bytes and the struct grew from 160 to 168. kinfer's mirror
gained the field and `bind`'s size check its new number. The rest of llama.h
only gained functions, two counters at the end of the perf struct kinfer does
not bind, and a vocabulary type at the end of its enum; nothing kinfer calls in
ggml-backend.h changed. The RPC protocol went from 7 to 8, so an RPC server
and its client must come from the same build.

On the box, ornith-1.5:35b, one slot of 16384, temperature 0, four rounds each,
old and new binaries back to back with nothing else loaded: decode 105.6 →
108.2 tok/s (median), an 8,438-token prompt read at about 1,950 tok/s by both.
The libraries grew from 8.8 to 12.4 MB, the binary to 24 MB.

Loading is not serving. llama.cpp's built-in templates include no DeepSeek V4
family; V4.1 needs a renderer of its own, which the entry above adds.

### llama.cpp b10901 → b11175, and Ollama's micro-batch

Measured against Ollama on the same ornith-1.5-35b file, kinfer decoded one
request at 87 tok/s to Ollama's 103. The cause was not the slots (one slot ran
the same), nor the CPU threads (Ollama's llama-server gave 103 with 4 or 20),
nor kinfer's code: `KINFER_DEBUG_SCHED` now splits a step, and of 11.5 ms per
token 10.06 was the GPU, 1.32 queueing it, 0.12 sampling, 0 the scheduler. The
official llama-server of kinfer's embedded build, b10901 — the same library
byte for byte — ran at 83.8. The official b11175 ran at 106–108. llama.h between
the two only gained functions; nothing kinfer binds changed shape.

Now embedded: b11175, fetched by `scripts/fetch-libs.sh`, which puts the eight
libraries (8.8 MB) where they are embedded and removes any other version. And
the micro-batch is Ollama's rule (automaticGenerationBatch
in its server/sched.go): 2048 tokens a pass for a context over 32768, 1024 over
4096, stepped down while the model and its cache are more than 60% (or 75%) of
the GPU budget or the extra memory the bigger pass takes is not there; `-batch`
overrides. A measured load that leaves too little steps the batch down before
anything a conversation would notice.

On the box, each measurement taken while ComfyUI was idle:

- **ornith-1.5-35b, 2048 a pass:** one request 106.6 tok/s (Ollama 103.3); four
  at once 166 together (was 145; Ollama 102); an 11,479-token prompt read in
  5.85 s (was 7.7; Ollama 6.17). Ollama is still 4% faster on a 103,641-token
  prompt.
- **Flash-Next, 512 a pass (76.2 of 77.8 GiB):** 41.9 tok/s alone (was 33.6),
  66 together at four (was 53), the same 4 x 16384 and 1.6 GiB left.

### `-slots` without `-ctx`: each model's context sized the way Ollama sizes one

The box ran `-ctx 16384 -slots 4`, numbers chosen for Flash-Next, whose weights
leave 4.3 GiB. They applied to every model the server loaded. ornith-1.5-35b,
whose weights leave 58 GiB, got the same four slots of 16384, while Ollama on
the same machine gave its one request 262144.

Ollama's rule is in its server/routes.go: 262144 tokens with at least 47 GiB of
GPU memory, 32768 with 23, 4096 below, capped at the trained context. It is per
machine, not per model. Given `-slots` and no `-ctx`, with flex on, kinfer now
takes that number as the total the slots share. It halves it until the cache
fits with 1 GiB to spare and splits it equally at home; flex gives one long
prompt the whole of it. A cache sized this way is shrunk after loading only if
it leaves under 1 GiB, not the 2 GiB warning line: Flash-Next at 4 x 16384
leaves 1.6. `kinfer plan -slots N` prints the result without loading anything.

Measured on the box, started with `-slots 4` and no other size given:

- **Flash-Next:** 4 x 16384, 65536 in all, 2.7 GiB of cache and 1.6 GiB left —
  what the old flags said.
- **ornith-1.5-35b:** 4 x 65536, 262144 in all, 5.9 GiB of cache and 52.1 GiB
  left. Four short requests at once came to 145 tok/s together, the same as at
  4 x 16384. A 103,641-token prompt got 2 x 131072 in 422 ms and answered whole:
  150 s to read it, then 55 tok/s.

Against Ollama on the same file this was, at the time, 18% slower one request at
a time and 42% faster several at once; the entry above is what closed the first
gap. The table is in docs/performance.md.

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
