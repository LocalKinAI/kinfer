# Serving

How `kinfer run` and `kinfer serve` load and size a model, keep it up, and say what it costs — part of [kinfer](../README.md).

## `kinfer run` keeps the model warm

`run` never loads a model itself. It talks to a background daemon, starting one
the first time, so the weights are loaded once instead of once per command —
the same reason `ollama run` feels instant.

```
$ kinfer run qwen "say hi in three words"    # 0.50s — daemon starts, model loads
Hi there!
[prompt 13 tokens in 0.01 s (1120 tok/s) · reply 3 tokens at 248.5 tok/s]
$ kinfer run qwen "name three colors"        # 0.41s — warm
$ kinfer run qwen                            # no prompt: interactive
kinfer · qwen
  /bye, /exit    leave (the daemon keeps the model warm)
  /clear         forget this conversation
> 

$ kinfer ps
NAME                                               SIZE  UNTIL
qwen2.5-0.5b-instruct-q4_k_m                   468.6 MB  5m0s

$ kinfer stop                                # the daemon, and the model it holds
stopped the kinfer daemon on 127.0.0.1:11500 (pid 46715)
```

The prompt edits by character — Chinese included, two columns a character —
with the cursor keys, Home/End, Ctrl-U/K/W and history; a paste is one message.
Ctrl-C stops a reply and leaves the conversation as it was; at an empty prompt
it leaves, as Ctrl-D does.

After each reply `run` says how fast the prompt was read and the reply made, on
stderr, so a reply piped from stdout is the reply alone; `-stats=false` drops
it. A reasoning model's thinking is shown in grey as it comes, when stderr is a
terminal. A model that does not fit this GPU borrows a kinfer worker on a
Thunderbolt cable, if one answers ([One model on two machines](big-models.md#one-model-on-two-machines)).

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
pool's own cells, then — for a model with experts in host memory — as much as
half of the expert cache placement chose, then pool entries, and the context
only after that. A pool
without cells of its own keeps its prompts in whatever the slots leave free and
gives them back when a slot needs the room, which on a machine with room for one
conversation still carries an agent's turns forward:

```
sizing: 1 slots + 1 prefix x 32768 tokens measured 2.6 GiB, leaving 1.7 GiB — the prefix pool shares the slots' cells instead, keeping 32768 tokens each
memory: MTL0   model 73.4 GiB + context 1.7 GiB -> 2.6 GiB free of 77.8 GiB (the whole GPU)
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
slots: 4 x 16384 -> 1 x 65536 for a 44555-token prompt (it queued 10.9s; the cache was rebuilt in 224ms, leaving 10.1 GiB free (the whole GPU))
slots: 1 x 65536 -> 4 x 16384 — nothing running needs the longer slots (it queued 1m56.3s; the cache was rebuilt in 223ms, leaving 9.8 GiB free (the whole GPU))
slots: 4 x 16384 -> 2 x 32768 for a 27174-token prompt (it queued 2m4s; the cache was rebuilt in 246ms, leaving 10.0 GiB free (the whole GPU))
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

## `-kv-cache q8_0` buys context, not speed

The attention cache is f16 unless `-kv-cache q8_0` says otherwise: K and V at 8
bits, half the memory (34 bytes for every 32 values, where f16 takes 64), and
sizing counts it at that. The same memory then holds twice the context — on the
box, Flash-Next unasked goes from 8 slots × 32768 tokens to 8 × 65536, which a
45,000-token agent prompt fits whole. llama.cpp turns flash attention on for it,
which a quantized V cache needs.

It is not the default because the memory is all it saves. On the box, with
20,000-token prompts:

| model | decode, f16 → q8_0 | prompt, f16 → q8_0 | context, f16 → q8_0 |
|---|---:|---:|---:|
| qwen3.8-27b Q4_K_M | 28.8 → 28.4 tok/s | 295 → 296 tok/s | 6.5 → 4.6 GiB |
| ornith-1.5-35b (MoE) | 97.4 → 92.3 tok/s | 1665 → 1700 tok/s | 1280 → 680 MiB of cache |
| Qwen2.5-0.5B | 259 → 222 tok/s | 7720 → 7630 tok/s | 576 → 306 MiB of cache |

(ornith's context as a whole grew: sizing spent what the cache saved on a
micro-batch of 2048 instead of 1024, which read the prompt 2% faster.) The DeepSeek
family, whose latent and compressed caches are not the per-head K and V this
counts, and a model whose heads 32 does not divide, stay at f16 when asked;
the log says which. `kinfer plan -kv-cache q8_0` shows the sizing first.

## Drafting with the model's MTP head

A model trained with multi-token prediction carries a small head beside it — one
more layer, on Qwen3.8-27B — that reads the hidden state the model left at a
token and guesses the tokens after it. kinfer has that head draft the next few
tokens of every conversation that is writing, and the model reads them in the
pass that reads its next token: it keeps each draft it would have written
itself, and where it would not, what it did write there. Every token is one the
model chose — at temperature 0, Qwen3.8-27B's replies came out identical with
drafts and without, fifteen of fifteen — and it comes sooner, because a dense
model reads every weight once whether a pass carries one token or five.

Qwen3.8-27B Q4_K_M on the box, one conversation, temperature 0, 256 tokens:

| reply | without drafts | with four a step |
|---|---:|---:|
| a Python function | 31.4 tok/s | 69.4 tok/s |
| a French translation | 31.3 | 58.8 |
| the planets, one fact each | 31.4 | 53.3 |
| how a hash map works, in 150 words | 31.5 | 37.3 |
| TCP's handshake, in Chinese | 31.3 | 36.8 |

How much it buys is how many drafts the model accepts: most of them in code,
which says what comes next, about a third in prose. On llama-server's own three
prompts kinfer accepted exactly the drafts llama-server's MTP drafting accepts
from the same file — 180 of 297, 189 of 264, 178 of 303 — at the same speed, 51,
58 and 50 tok/s. Ollama drafts with the same head: on the decode prompts of the
comparison with it, kinfer wrote 43 tok/s and Ollama 42, where kinfer had
written 31.5.

A pass costs more the more tokens it reads — about 7 ms a token past the first
few, draft or not — and every conversation in it pays. So drafts are shared out
over the conversations writing: four each for one or two, two for three, one
for four to eight, none past eight, where the head also stops reading what they
write. The same model, conversations writing at once, tok/s together:

| at once | without drafts | with |
|---:|---:|---:|
| 1 | 30.4 | 38.0 |
| 2 | 38.0 | 51.3 |
| 3 | 48.0 | 62.2 |
| 4 | 60.8 | 89.3 |
| 6 | 83.3 | 90.3 |
| 8 | 102.7 | 110.7 |

(Four drafts each for four conversations had made it 42.6.) Past eight it costs
a little: no drafts are made, but each pass still writes the snapshots below —
ten at once wrote 85.6 tok/s together, against 92.1 without. A fleet that keeps
more than eight conversations writing is better off with `-draft -1`.

It costs memory, which sizing counts: the head's own cache, as long as the
model's, and for a hybrid model like this one a snapshot of every sequence's
recurrent state for each draft, to go back to when one is rejected — 150 MiB a
snapshot here. With the box service's four slots of 65536 and a pool of two,
the context took 24.7 GiB instead of 19.9; with twelve slots of 4096 and a pool
of four, 19.0 instead of 8.6 — the snapshots are per sequence. Reading a prompt
went from 315 tok/s to 300: the head reads every token the model does, and the
larger cache left room for passes of 1024 tokens rather than 2048.

A mixture of experts drafts one token a step. Every token of a pass fetches its
own experts, so each draft costs nearly a token's reading, and the head's
guesses past the first are seldom right. ornith-1.5-35b (35B, 3B active) on the
box, llama-server's three prompts, 256 tokens:

| drafts a step | tok/s | drafts accepted |
|---:|---|---|
| none | 107.4, 108.7, 109.4 | |
| 1 | 130.7, 130.9, 118.3 | 77%, 85%, 66% |
| 2 | 107.0, 119.8, 101.3 | 43%, 53%, 40% |
| 4 | 80.2, 84.6, 77.6 | 22%, 27%, 20% |

With one a step two at once wrote 166 tok/s together instead of 149, four 229
instead of 201; the head's layer added 0.7 GiB to the model and the snapshots
3.5 GiB to the context. One is also the most at which a pass reads exactly what
reading the tokens one at a time would: past two tokens a pass, Metal multiplies
some weights with a kernel that sums in another order, and the expert router
can turn that last-bit difference into another word where the model was torn.
At one draft ornith's five replies were identical to its replies without; at
two or more, three of five parted somewhere — "insert a key" became "insert a
key-value pair", "swinging from" "swinging dramatically from", 三次完成后
完成三次握手后 — at the same three places at two, three and four. That is the same kind of difference the prefix pool makes, and like it
not a mistake: each word is one the model chose. A mixture of experts whose
experts are kept in host memory does not draft unless asked, since the reads a
draft adds come from there.

`-draft N` sets how many tokens a step one conversation drafts (up to 8), and
`-draft -1` turns it off and gives the memory back. A conversation with pictures
is not drafted for, nor is a model split across machines. `kinfer plan -draft 4`
shows the sizing with it; `/metrics` counts the drafts and the ones accepted,
and the log says how many were when the model unloads.

On MLX the same head drafts, for a Qwen3.5 or Qwen3.8 model that keeps it —
Ollama's NVFP4 models and Hugging Face's releases do, mlx-community's conversions
drop it — five tokens a step, while a reply writes alone; a second request puts
the step back to one token a reply, and the one that was drafting goes on in the
batch. A step waits on the GPU once: the pass that checks the drafts is built
while they are worked out. A model under 4 GB of weights does not draft unless
`-draft N` says so: on the M3 Ultra, Qwen3.5-0.8B wrote 221 tok/s alone without
drafts and 141 with three. qwen3.8:27b-mlx on a MacBook Pro (M5 Max) went from
30 tok/s to 40–107, 72.5 on average over six replies — Ollama 0.40.15 averaged 78
on the same six.

## Embeddings, beside the chat model

An embedding model makes vectors instead of text: Ollama's `/api/embed`
(`input` a text or texts; `truncate`, `dimensions`, `keep_alive`), its older
`/api/embeddings` (`prompt`, one vector), and OpenAI's `/v1/embeddings` (`input`
a text, texts, or tokens; `encoding_format` float or base64). `/api/embed` and
`/v1/embeddings` answer with vectors of unit length, so a dot product is the
cosine; `/api/embeddings` gives the model's own, as Ollama's does. `dimensions`
keeps that many of the first values and normalizes them again — the sizes a
Matryoshka-trained model such as EmbeddingGemma is still good at (512, 256, 128).

A model embeds if its header names a pooling (`<arch>.pooling_type`: mean, the
first token, the last), or if it is an encoder — BERT, nomic-bert, jina-bert,
modern-bert, EmbeddingGemma — which embeds by the mean of its tokens when its
header names none. Ollama's own copies are found the way its chat models are,
so `nomic-embed-text` pulled with `ollama pull` answers without a download.
EmbeddingGemma 2 (`embeddinggemma-2`), which Ollama 0.40 keeps for its MLX
engine as safetensors in NVFP4 rather than a GGUF, runs on MLX on a Mac
(internal/mlxembed), up to 8,192 tokens, the same vectors as Ollama's to a
cosine of 0.9998 or better. It reads pictures and sound too, as Ollama takes
them: an input `{"image": base64}` or `{"audio": base64}`, with an optional
`"text"` before them. A picture Go does not decode (WebP, HEIC, TIFF) goes
through macOS's `sips`, and a sound that is not a WAV through `afconvert`.

It loads in a place of its own, beside whatever chat model is loaded, with its
own idle clock. That is the point: an agent asks for vectors between its turns
— which documents to put in the prompt it is about to send — and unloading the
chat model for them would cost the next turn a reload, half a minute for
Flash-Next. One embedding model is loaded at a time; asking for another swaps
it, not the chat model. It shows in `kinfer ps` and `/api/ps`, `keep_alive`
governs it as it does a chat model, and `ollama stop` unloads it. A chat request
naming an embedding model is refused (400) before anything is unloaded, and
`/api/show` says `"capabilities": ["embedding"]`.

Texts are read in passes of up to 16, as many as fit 8192 tokens — or the
length the model was trained for, if shorter — and a text never spans two
passes: an encoder reads a text both ways at once. A longer text is cut to fit,
keeping the marker that closes it (BERT's `[SEP]`), unless `truncate` is false,
which refuses it. Pictures and sound in `input` — Ollama 0.40 embeds them with
EmbeddingGemma 2 — are refused for now.

## Pictures, for a model with a projector

A vision model reads pictures through its projector — the "mmproj" file a
Hugging Face repo ships beside it, a layer of its own in Ollama's store. kinfer
loads it with the model when there is one: Ollama's layer for a model borrowed
from its store, or `NAME.mmproj.gguf` beside `NAME.gguf` in kinfer's own, which
`kinfer pull` fetches with a vision model (F16 where the repo has it) and
`kinfer rm` removes with it. It loads before memory is measured, its encoder
warmed up, so the context is sized around it.

Every dialect carries pictures its own way, and each is read: Ollama's `images`
(base64, which go before the message's text), OpenAI's `image_url` content parts,
Anthropic's `image` blocks and the Responses API's `input_image` — base64 or a
`data:` URL. A link is refused: kinfer does not fetch a picture on a caller's
behalf. A picture llama.cpp cannot decode (jpg, png, bmp and gif are what it
reads) is a 400.

A prompt with pictures is read whole when its turn comes, by llama.cpp's libmtmd:
each picture encoded, then read into its conversation's cells, positions laid out
in two dimensions where the model wants them so (Qwen-VL). It reads alone — the
other conversations wait the second or so that takes — and the prefix pool, which
knows a prompt by its tokens, neither lends to it nor keeps it. Nor is an older
turn dropped to make it fit: a conversation of pictures longer than its slot is
refused (413), as a text one too long by itself is. A picture for a model
without a projector is refused (400), as Ollama refuses it. It used to be said in
words where it was; told so, a 4B model described a picture it had not seen.

## Speech to text, for a model that hears

A speech recognition model hears through a projector too: Qwen3-ASR's GGUF
ships an audio encoder as its mmproj, and libmtmd takes a recording where it
would take a picture. So `POST /v1/audio/transcriptions`, as OpenAI has it,
goes down the pictures' path — the recording in the user's turn, the reply the
transcript — and nothing on the chat path changed for it.

```bash
kinfer pull ggml-org/Qwen3-ASR-1.7B-GGUF:Q8_0          # 2.2 GB, and its projector
kinfer serve -addr :11591 -slots 1 -ctx 16384 -prefix off
curl -F file=@meeting.wav -F model=Qwen3-ASR-1.7B-Q8_0 localhost:11591/v1/audio/transcriptions
{"text":"今天下午在二号会议室开项目进度会，请大家提前准备好本周的数据。"}
```

- `file`: wav, mp3 or flac, what llama.cpp decodes; on MLX also m4a, aiff and
  whatever else macOS's afconvert reads. Anything else is a 400.
  Up to 64 MiB, and as long as the slot holds: a second of speech is about 13
  tokens of sound and 3 of transcript, so 16384 tokens hold some fifteen
  minutes. A longer one is a 413.
- `language`: an ISO 639-1 code (`zh`, `en`, `yue`…) or the English name of
  one of Qwen3-ASR's 30 languages. The model's reply is started with it, as
  Qwen3-ASR is told a language; without it the model says which it heard. One
  it cannot name is a 400.
- `prompt`: what is known about the recording — names, terms — which Qwen3-ASR
  takes as its system message.
- `response_format`: `json` (the default), `text` or `verbose_json`, which adds
  the language heard and, for a WAV, the duration. `temperature` defaults to 0.
- `stream=true` sends the transcript as it is written, as OpenAI streams one:
  server-sent events of `{"type":"transcript.text.delta","delta":…}`, then
  `{"type":"transcript.text.done","text":…,"usage":…}`. The "language
  Chinese<asr_text>" the model opens with is held back, not sent. json and text
  only.
- Not yet: `srt` and `vtt`, timestamps.
- A model whose projector hears nothing is a 400.

Both pulls were run on the box into an empty store (11-09): the GGUF and its
projector, kept as `Qwen3-ASR-1.7B-Q8_0.mmproj.gguf`, 2.6 GB in 134 s, and the
MLX conversion, 2.3 GB in 117 s, each with its sha256 checked against Hugging
Face's, then served and transcribing.

`POST /v1/audio/translations`, as OpenAI has it, gives the recording's words in
English. Qwen3-ASR does not translate — told to in its prompt, or started with
English, it still writes Chinese down in Chinese — so the transcript is
translated by a chat model on a chat server, since the speech server is a
process of its own: start it with `KINFER_TRANSLATE_URL` (that server's address,
such as `http://127.0.0.1:11590`) and `KINFER_TRANSLATE_MODEL`. Without them a
translation is a 400 saying so. A recording heard in English goes back as
heard, and a translation is not streamed. On the box, with ornith-1.5:35b
translating: "There will be a project progress meeting in Conference Room No. 2
this afternoon. Please prepare this week's data in advance."

`kinfer serve` answers it at its own address (see [One address for chat and
speech](#one-address-for-chat-and-speech)): the speech model runs in a process
of its own, so the chat model stays loaded and a recording that crashed libmtmd
takes only that process down, sized as a speech model on llama.cpp wants — one
slot of 16384 tokens and no prefix pool, 2.6 GB, where `serve` left to itself
gave the 1.7B model four slots of 32768 and a pool, 22.5 GB. The pool never
helps here, since a prompt with sound is read whole. A server of its own for
speech alone is `kinfer serve -speech off -slots 1 -ctx 16384 -prefix off`.

### On MLX, on a Mac

On Apple silicon the same model runs on MLX too, from mlx-community's 8-bit
conversion, which mlx-audio — the box's Python service — runs as well. A
directory whose config.json names `qwen3_asr` is served by `internal/mlxasr`,
a port of mlx-audio's Qwen3-ASR to Go on the MLX library kinfer's MLX engine
uses: Whisper's log-mel features (the FFT on the GPU), the convolutional front
end and 24 encoder layers attending within windows of eight seconds, and the
Qwen3 decoder reading the prompt with the sound in it. Checked against
mlx-audio step by step on the same recordings: the features within 1.4e-4, the
encoder's output within 1.3e-6, the prompt token for token, the transcripts word
for word.

```bash
kinfer pull mlx-community/Qwen3-ASR-1.7B-8bit          # 2.5 GB
kinfer serve -addr :11591
curl -F file=@meeting.wav -F model=Qwen3-ASR-1.7B-8bit localhost:11591/v1/audio/transcriptions
```

It takes the same fields. A 16 kHz WAV is read in Go; any other recording —
another rate, mp3, m4a, flac — goes through macOS's `afconvert` first. One
recording is transcribed at a time, and the MLX buffers it used are let go
after each.

Two things had to differ from mlx-audio for the memory to. Its GELU is compiled
into one kernel; here its five ops each make an intermediate as large as the
convolution's output, so the convolutions take sixteen seconds of sound at a
time. And an MLX array lives as long as a handle to it does, so each encoder
layer runs in a scope of its own: with one scope for the encoder, 163 s of sound
peaked at 21.7 GB, against 4.4 GB now, the 2.5 GB of weights included.

### The emotion and the sounds: SenseVoice

SenseVoice Small, FunAudioLLM's, hears more than words: the first positions of
its output name the language, the emotion (happy, sad, angry, neutral,
fearful, disgusted, surprised) and the sound besides speech (BGM, laughter,
applause…). It runs on MLX here, ported from mlx-audio as Qwen3-ASR is
(`internal/mlxasr/sensevoice.go`): Kaldi's filterbank, seven frames stacked
every six, the model's am.mvn, 70 SANM layers and a CTC head, read whole —
nothing is generated a token at a time, so a recording takes 23–33 ms, 76 ms for
54 s. Checked against mlx-audio on nine recordings in five languages: the
features within 0.0008, the head's log-probabilities within 5e-4, the most
likely token the same at every one of 1,035 positions.

```bash
kinfer pull mlx-community/SenseVoiceSmall               # 893 MB, with am.mvn and its vocabulary
KINFER_EMOTION_MODEL=SenseVoiceSmall kinfer serve -addr :11591
curl -F file=@angry.wav -F model=Qwen3-ASR-1.7B-8bit -F response_format=verbose_json localhost:11591/v1/audio/transcriptions
{"emotion":"angry","language":"chinese","task":"transcribe","text":"你到底有没有在听我说话？…"}
```

- Beside Qwen3-ASR, named by `KINFER_EMOTION_MODEL` (a model in the same
  store, or a path), it adds `emotion` and `audio_events` to the transcript,
  as the Python service's `--emotion sensevoice:small` does: in json and
  verbose_json, and the streamed transcript's last event, each only when there
  is one — an emotion it cannot tell, and speech alone, add nothing.
- On its own (`model=SenseVoiceSmall`) it writes the transcript too, in zh, en,
  yue, ja or ko, with the same fields.

Two things differ from the two Python implementations on purpose. mlx-audio
does not fetch am.mvn with the MLX conversion and so runs without the model's
normalization (CMVN); this applies it, as FunASR does. And FunASR dithers the
samples (noise of σ 1 on the 16-bit scale, seeded), which this does not, so its
words can differ from the Python service's: "开放时间早上9点" where FunASR writes
"开饭时间", "持っていけない" where it writes "持っていきない", but also a Korean
sentence spaced worse and an mp3's "July 21, 1969" read as "July 1". On six
recordings spoken happily, sadly and angrily in Chinese and English (made with
Qwen3-TTS VoiceDesign), it told five, the English angry one not — and neither
did FunASR: the emotion was the same as FunASR's on all six. MLX only; on Linux
and Windows there is no SenseVoice yet.

### Measured

On the M3 Ultra (11-09), against the Python service the box runs (mlx-audio,
the 8-bit MLX conversion), each a fresh process, the median of six runs in
rotating order. kinfer on llama.cpp ran the GGUF at Q8_0 with `-slots 1 -ctx
16384 -prefix off`:

| recording | kinfer, llama.cpp | kinfer, MLX | Python, MLX |
|---|---|---|---|
| 5 s English | 0.18 s | 0.18 s | 0.17 s |
| 7 s Chinese | 0.18 s | 0.19 s | 0.18 s |
| 44 s English | 1.09 s | 1.00 s | 0.93 s |
| 53 s English, 24 kHz | 1.25 s | 1.19 s | 1.21 s |
| 54 s Chinese | 1.07 s | 0.96 s | 0.91 s |
| 163 s Chinese | 3.21 s | 2.88 s | 2.64 s |
| four 54 s recordings at once | 4.22 s | 3.73 s | 3.52 s |
| memory (whole process) | 2.4 GB | 2.6 GB | 3.4 GB |

The words were the same in six recordings of seven, and none wrong in the five
whose text is known. The seventh, an mp3 of a newspaper read aloud, came out
three ways: "July twenty one, one thousand, nine hundred and sixty nine"
(llama.cpp), "July twenty first, nineteen sixty nine" (kinfer on MLX), "July
21st, 1969" (Python). kinfer's MLX path computes what mlx-audio does from the
same samples, so the difference there is the mp3's decoding and resampling,
afconvert's against Python's.

Where the time goes, for the 163 s: llama.cpp spends 0.91 s on the encoder and
the prompt and 2.15 s decoding at 185–189 tokens a second; mlx-audio 0.70 s and
1.92 s at 209; kinfer on MLX 0.63 s and 2.01 s at 201, in all 2.64 s against
mlx-audio's 2.62 (the table above was measured before the next fix). It had
been 2.19 s at 185: each decoding step asked MLX for the KV cache's arrays as
well as the token, where mlx-lm asks for the token alone and lets the cache be
computed as the token needs it. A short reply now decodes as fast as
mlx-audio's, 219–223 tokens a second against 222; a long one is 4% slower. llama.cpp's own server is as fast as kinfer on it, so kinfer
adds nothing there: the difference between the two engines is their kernels.

## One address for chat and speech

`kinfer serve` answers chat and speech at one address. The chat model is
loaded here as ever; a request to `/v1/audio/transcriptions`,
`/v1/audio/translations`, `/v1/audio/speech` or `/v1/audio/voices` goes to a
speech model in a process of its own — a runner, as Ollama runs each model: this
binary's `serve -speech off` on the loopback, started on the first request for
its model and stopped once idle as long as `-keepalive`. So the chat model stays
loaded, the three can be used at once, and a speech model that crashes takes
only its runner down, to be started again by the next request. A runner reads
the server's stdin and ends with it, however the server ends.

```bash
kinfer serve                                        # chat, hearing and speaking at :11500
curl localhost:11500/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"一个地址，什么都能做。"}' -o one.wav
curl -F file=@one.wav -F model=whisper-1 localhost:11500/v1/audio/transcriptions
{"text":"一个地址，什么都能做。"}
```

- The model is the one a request names, if that is a speech model of the right
  kind here (an MLX Qwen3-ASR, SenseVoice or Qwen3-TTS, or a GGUF whose
  projector hears); any other name — OpenAI's `whisper-1` and `tts-1`, the
  Python service's `tts` — is this machine's default of the kind: Qwen3-ASR
  1.7B and Qwen3-TTS 0.6B CustomVoice on MLX on a Mac, Qwen3-ASR's GGUF
  elsewhere (no text to speech there: 501). A default this store has of
  another form is taken before one fetched: Qwen3-ASR's GGUF on a Mac without
  the MLX one.
- A default that is not here is fetched the first time it is asked for
  (`-auto-pull`, on unless set false): the request is answered 503 at once,
  with how far the fetch has got and a Retry-After, and asked again. On the box,
  from an empty store: 503 with 21%, 46%, 65%, 84%, the Retry-After falling 58 →
  11 s, and at 78 s the speech. `-auto-pull=false` answers 404 saying what to
  pull; `kinfer pull speech` fetches the defaults beforehand.
- `-speech keep` starts the default speech models with the server and keeps
  them, loaded: the first answer then takes what any other does (0.13 s for a
  transcript). `-speech off` serves speech in this process, one model at a time
  with the rest, as before; a runner is that.
- `/api/ps` lists the speech models beside the chat model.
- Translations go to this server's own chat model address
  (`KINFER_TRANSLATE_URL` unset), the model still `KINFER_TRANSLATE_MODEL`.
  `KINFER_EMOTION_MODEL` reaches the runners as it is.

Measured on the box (11-09): a runner up in 105 ms; the first speech 2.1 s
with the model's loading, then 0.7 s; a transcript 0.14 s; chat, speech and a
transcript at once 0.6 s; streamed speech's first audio 0.15 s through the
front server; a runner killed with -9, started again by the next request in
1.2 s; idle runners stopped after `-keepalive`; the server's SIGTERM and its
SIGKILL each ending its runners within 4 s. Not yet: the speech models are not
counted in the memory a chat model is sized against when they start after it,
and chat models are still loaded in the server itself.

## Every model in a process of its own

`kinfer serve` loads no model itself. A chat or embedding request goes, by the
model it names, to a runner of that model — this binary's `serve -runners=false`
on a port of the loopback, with the sizing flags this server was given — as
speech and pictures do. So:

- A model that crashes takes only its runner down. A reply it was writing ends
  with an error in its own format (`{"error": …, "done": true}` in Ollama's
  stream, `data: {"error": …}` in OpenAI's), and the next request starts the
  model again.
- A model unloaded is a process ended: all its memory goes back, and none of a
  model's state outlives it into the next load.
- Before a runner starts, this server makes room on the GPU for its model,
  from what the whole GPU has free (other programs' allocations included):
  about the model's weights plus what it needs beside them. While that is
  short it stops the runner unused longest that nothing is using — the default
  speech models kept by `-speech keep` stay — and once none is left, waits up to
  five minutes for one in use to finish. What it cannot free, the model's own
  sizing makes do with. Two models that fit stay side by side; a chat model
  sizes its context and prefix pool to the memory it finds, so a second one
  usually stops the first, as one model at a time did before.
- A request's `keep_alive` is kept here; 0 stops the runner once the request is
  done, and a request with nothing in it and `keep_alive` 0 stops it at once.
- `/api/ps` lists each runner's model as it reports itself, its context too;
  `/metrics` gathers the runners', each line labelled with its model.

`-runners=false` serves chat in this process, one model at a time, as v0.6.0
did. Measured on the box: decode through a runner 94.6–104.1 tok/s against
95.7–99.1 in-process (ornith-1.5, 200 tokens, three each); a runner killed
with -9 mid-reply, the next request answered 2.3 s later; ten unloads and
reloads, every reply right and no runner left behind. Making room, with the
budget pretended at 45 GB (`KINFER_RUNNER_BUDGET`, for testing on a machine
with room): ornith loaded; qwen3.8-27B stopped it; ornith asked for while
qwen3.8 wrote a long reply waited, then stopped it; a picture stopped ornith.

## Text to speech, for a model that speaks

`POST /v1/audio/speech`, as OpenAI has it, answered on a Mac by Qwen3-TTS
CustomVoice: nine voices by name, ten languages, a tone told in words. A
directory whose config.json names `qwen3_tts` is served by `internal/mlxtts`, a
port of mlx-audio's Qwen3-TTS — what the box's Python service runs — to Go on
the MLX library. Speech is made 12.5 frames a second: the talker, a Qwen3 model
reading the text and the voice, writes each frame's first code, the code
predictor its other fifteen, and the speech tokenizer's decoder turns the codes
into 24 kHz audio, 1,920 samples a frame.

```bash
kinfer pull mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit   # 2 GB, with its speech_tokenizer/
kinfer serve -addr :11592
curl localhost:11592/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit","input":"明天上午九点记得去复查。","voice":"serena"}' -o reminder.wav
```

- `input`: up to 4,096 characters. `voice`: aiden, dylan (Beijing), eric
  (Sichuan), ono_anna, ryan, serena, sohee, uncle_fu, vivian. As the Python
  service does, a Kokoro voice is spoken in the nearest (`zf_…` vivian, `af_…`
  serena, …) and any other name, OpenAI's alloy among them, in the default:
  vivian, or `KINFER_TTS_VOICE`. The answer's `X-Kinfer-Voice` says which.
- `language` (kinfer's): a code (`zh`, `en`, `ja`…) or the model's name for it;
  without one it is told from the text's script as the Python service tells it
  — kana Japanese, hangul Korean, Han Chinese, the rest English.
  `X-Kinfer-Language` says which.
- `instructions`: the tone, "用特别开心的语气说", passed to the model as mlx-audio
  passes it.
- `response_format`: `wav` (the default, 16-bit, as the Python service answers;
  OpenAI's is mp3) or `pcm` (16-bit little-endian, 24 kHz, no header). Not yet:
  mp3, opus, aac, flac.
- `speed`: 0.5 to 2, by time-stretching the speech with its pitch kept (WSOLA,
  the Python service's), since Qwen3-TTS takes no speed of its own. Not with a
  stream.
- `stream_format`: `audio` sends the speech as it is made — pcm does by default,
  as OpenAI's does; a WAV starts with a header of unknown length — and `sse`
  sends OpenAI's `{"type":"speech.audio.delta","audio":…}` events, then
  `speech.audio.done` with the usage. The first piece is 4 frames (0.32 s), the
  rest 7, as the Python service's 0.6 s.
- `temperature` (kinfer's): 0.9 by default, mlx-audio's; top-k 50 and a
  repetition penalty of 1.05 on the first code, as there.
- `GET /v1/audio/voices?model=` lists the voices and the languages.

The pull was run on the box into an empty store (11-09): twelve files, 1.8 GB,
in 94 s, the weights' sha256 checked against Hugging Face's, `speech_tokenizer/`
kept as a folder beside the talker, then served and speaking.

`kinfer serve` answers it at its own address, the model in a process of its
own as a speech recognition model is (see [One address for chat and
speech](#one-address-for-chat-and-speech)): one kinfer process holds one model
besides its embedders, and a home assistant hearing and answering in turn would
otherwise swap the two at every turn.

An answer is bounded by its text: 2 s and three times what it takes read at an
ordinary pace (a quarter second a Han character, kana or hangul, 0.08 s
anything else). Qwen3-TTS 0.6B now and then talks on past its text, mlx-audio's
as much as kinfer's — on "Hello there, this is kinfer speaking.", normally 3–5
s, one seed in six on either side ran to the 12 s a test allowed, pausing,
saying it twice or babbling — and would have gone on for minutes before its
4,096 frames ran out. An answer that reaches the bound is said again, once,
with another seed; a streamed one, already on its way, is cut there. Over 30
seeds of that sentence, every answer now came back in 3.1–9.7 s, three of them
said again; over 30 of a Chinese one, 7.0–10.2 s, none.

`seed` (kinfer's) makes an answer the same each time it is asked for. The
draws follow it as mlx-audio's follow `mx.random.seed` — the key split before
each, its second half drawn with — from a key of the answer's own rather than
MLX's global one, which MLX keeps for each thread while Go moves the work
between threads: before, the same seed gave different speech run to run.

How close it is to mlx-audio's: the prompt it builds, the talker's first
logits and the code predictor's are the same bits; the decoder on mlx-audio's
codes within 1.5e-6. After that the two drift by rounding: kinfer's MLX is not
the Python service's (0.32.5-0f4740d against 0.32.2), and fed mlx-audio's own
codes frame after frame, the talker's logits stay within cos 0.99992 (no first
code drawn otherwise) and the code predictor's within cos 0.9997 — one
bfloat16 step on a logit now and then, which turns a near tie the other way in
about one code in thirty. So the same seed does not give the same speech, and
the speech is as good: four sentences six times each from both, heard back by
Qwen3-ASR, every one word for word, the durations alike.

The decoder streams exactly: it is causal, and carries from piece to piece
each convolution's last inputs, each upsampling block's overlap and the
transformer's keys and values, so speech made a piece at a time is the speech
made at once (within 2e-6). mlx-audio's streaming adds the transposed
convolutions' bias twice where pieces meet; this does not. A whole answer is
decoded 50 frames at a time the same way, where mlx-audio decodes 300 at a time
and reads each after only 25 frames of the last, so beyond 24 seconds the two
differ there too, kinfer's being the one pass.

Measured on the M3 Ultra (11-09), each in its own process, two rounds of the
same four sentences three times each, alternating:

| | kinfer | Python (mlx-audio) |
|---|---|---|
| speed | 5.15–5.21× real time | 5.03–5.14× |
| first audio, streamed | 0.09 s | 0.12–0.17 s |
| memory (whole process) | 1.5 GB | 2.4 GB |

Three things made it so. Frames are pipelined: while the GPU works one out, the
next one's graph is built on the codes it will have drawn — the repetition
penalty's recent codes stay on the GPU for that — so building costs no waiting.
The rotation of queries and keys and the SwiGLU are each one kernel, rounding
as mlx-audio's compiled ones do, and q, k, v and gate, up are each one product,
which leaves the same bits; a frame is some 1,750 kernels, and handing them to
MLX is most of a frame's 14 ms. And the talker's text embedding, 151,936 rows
of 2,048 — 622 MB — of which a sentence reads a few dozen, is read a row at a
time from its file rather than held. The 1.7B models, VoiceDesign and Base's
voice cloning are not served yet.

### A voice told in words: VoiceDesign

Qwen3-TTS's VoiceDesign models have no voices by name: the voice is the one
`instructions` describe, a new one each time.

```bash
curl localhost:11500/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit","input":"孩子们，过来坐下。","instructions":"一位声音低沉的老爷爷，说话慢悠悠的"}' -o grandpa.wav
```

Against mlx-audio on the box: the prompt the same bits, the decoder within
1e-6, the talker's logits cos ≥ 0.99996 fed the same codes and one near tie in
73 frames drawn otherwise; 4.3–4.6× real time against mlx-audio's 4.6×.

## The Python audio service's addresses

Before kinfer heard and spoke, the box's speech was localkin-service-audio,
one Python process a model at a port of its own, with an API of its own beside
OpenAI's. `-speech-addr` has kinfer answer at those ports in its place:

```bash
kinfer serve -speech keep -speech-addr asr=:8100,tts=:8101,design=:8102
```

Each address answers one kind — `asr` speech recognition, `tts` Qwen3-TTS's
voices by name, `design` VoiceDesign — in the model this server chooses for
it, whatever model a request names, as the Python service did:

- `POST /transcribe`: the recording in the form's `file`, `language` and
  `response_format` (`json` or `text`) as query parameters or fields; the
  answer `{"text", "language", "duration"}`, with `emotion` and
  `audio_events` when there are any.
- `POST /synthesize`: JSON with `text`, `speaker`, `language`, `speed` and
  `instruct`; the answer WAV.
- `GET /voices`: each voice's id, name, language and whose it is.
- `GET /health`, `GET /`, and OpenAI's `/v1/audio/*` too.

On the box, after the Python services stopped: Pixelle's `/transcribe` and
`/synthesize`, OpenMontage's `/v1/audio/transcriptions` and `/synthesize` in
both voices, and kinhome's OpenAI calls all answered; a 104-character
narration in a designed voice came back heard word for word.

## Pictures from words

`POST /v1/images/generations`, as OpenAI has it, answered on a Mac by
Z-Image-Turbo (Tongyi-MAI, Apache-2.0): a port of mflux's to Go on the MLX
library (`internal/mlximage`), reading mflux's saved weights. Behind the chat
address it runs in a runner of its own, as speech does, started on the first
request and stopped once idle; the default is fetched on first use.

```bash
kinfer pull mflux-community/z-image-turbo-mflux-q8      # 11.0 GB, 8-bit
curl localhost:11500/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"黄昏时的江南水乡，小船，灯笼，水墨画","size":"768x1024","seed":3}' \
  | python3 -c 'import sys,json,base64; open("pic.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

- `size` is 256 to 2048 a side in multiples of 16 (1024x1024 by default),
  `n` up to 4, each with the next seed; the answer is PNG, `b64_json`, or a
  `data:` URL when `response_format` is `url`. kinfer's `seed` gives the same
  picture each time, and `steps` (9) the denoising steps.
- The model is the one a request names if it paints, else the default —
  OpenAI's `dall-e-3` and `gpt-image-1` get it.

The pipeline is mflux's, op for op: the prompt in Qwen3's chat template, the
Qwen3-4B text encoder's second-to-last hidden state, the 6B single-stream
transformer (two refiner blocks for the picture's patches and for the caption,
then thirty over both, each modulated by the timestep), nine Euler steps on a
shifted linear schedule, and Flux's VAE decoder.

Measured on the box (M3 Ultra) against mflux 0.22 run eagerly: the tokens,
the caption's features (cos 1.0000000), the schedule and the noise the same;
the first step's velocity cos 0.9997; the decoder on mflux's latents cos
0.99999; the pictures at 40.5 dB (512²) and 40.8 dB (1024²) from mflux's own.
1024×1024 in 30.9 s, 29.7 s of it denoising, as mflux's 29.7 s; 1024×768 in
22.5 s; 512×512 in 7.5 s. The runner holds 10.8 GB and peaks at 18.4 GB for a
1024² picture.

One thing a port in Go has to do that Python does by itself: each stage's
intermediates are let go before it is evaluated. A handle still held keeps its
buffer to the end of the evaluation, and a 1024² picture's attention held so,
step after step, took a runner to 84 GB and the box into swap; let go, MLX
frees each as soon as the graph is done with it.

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

**A request survives a backend failure.** llama.cpp reports a fatal decode for
the whole context, so kinfer retires the model — and used to fail every request
it held. Measured: 32 concurrent requests against a dying backend, 0 answered
before, 28 after. Those 28 were still queued and simply run on the replacement.
The four that had already streamed text now go on too: the replacement prefills
the reply so far after the prompt and continues from its end, so nothing is sent
twice and the client sees a pause. Flash-Next on the MacBook Pro, with a failure
injected 40 tokens into its thinking: the same thinking and answer, word for
word, 15 seconds later. A request reloads at most twice.

On a Mac the usual failure is the GPU running out of memory under the model,
because an app took some after the load. Free GPU memory is what the whole GPU
has left — other processes' allocations taken off, read by kinfer's build of
llama.cpp (patch 0004, `GGML_METAL_FREE_DEVICE`, which `=0` turns off) — so a
replacement is placed to fit what is free when it loads, and so is a model in
the first place. Metal alone reports only what this process has left of its
budget: on the MacBook Pro, with 8 GiB held by a browser, Teams and the rest,
Flash-Next was placed into 37.7 of the GPU's 40 GiB and its first reply failed
with `Insufficient Memory` every time. Placed against the 32 GiB actually free —
40 layers' experts on the SSD and a 12.9 GiB cache — it decoded a 384-token
reply at 29.9 tokens a second; placed into all 40 GiB, on a day that happened
to work, it did 30.9.

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
