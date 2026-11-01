# Changelog

## Unreleased

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
