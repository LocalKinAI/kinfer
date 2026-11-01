# Models bigger than the machine

A mixture of experts the GPU cannot hold, from host memory, from the SSD, and across two machines — part of [kinfer](../README.md).

## One model on two machines

A model can take a second machine's GPU for some of its layers, through
llama.cpp's RPC backend. The other machine runs a worker; this one names it:

```bash
# on the MacBook Pro: its end of the cable between the two
kinfer worker -addr 169.254.85.249:50052

# on the box: the layers that do not fit here go there, the rest and the output stay
kinfer serve -peers 169.254.85.249:50052
```

The worker is kinfer itself, not llama.cpp's `rpc-server`, because both ends
must be the same llama.cpp build — the protocol went from 7 to 8 between b11175
and b11998 — and one binary on both machines cannot disagree with itself. The
protocol has no authentication, so a worker belongs on the cable between the two
machines (a Thunderbolt bridge, a USB link), never on a LAN address; it defaults
to loopback. The worker keeps every tensor it is sent in `~/.kinfer/rpc-cache`,
by hash, so a model's second load sends hashes instead of weights.

Measured with Qwen3.8-Flash-Next IQ3_S between the box (M3 Ultra) and a MacBook
Pro (M5 Max), joined by a USB-C cable that negotiated USB 2.0 — 1.2 ms round
trip, 343 Mb/s:

| | box alone | 2 layers on the laptop | 8 layers on the laptop |
|---|---:|---:|---:|
| model on the box | 50.7 GiB | 48.8 GiB | 42.6 GiB |
| first load | — | — | 248 s (8 GB over USB) |
| next load, from the worker's cache | — | 7.1 s | 21.9 s |
| decode | 44.5 tok/s | 26.3 tok/s | 23.7 tok/s |
| a 7,016-token prompt | 666 tok/s | 284 tok/s | 283 tok/s |

Two layers cost almost as much as eight, and reading a prompt is the cable's
bandwidth, the same however the layers are split.

### Over Thunderbolt 5, and the GPU that waits

A Thunderbolt 5 cable between the same two machines (Thunderbolt Bridge, 80
Gb/s) moves 57–68 Gb/s each way over TCP, two hundred times the USB link. A
round trip still takes about 0.3 ms, as it does over USB: that is the TCP stack
at both ends, not the wire. The same model, v0.1.0 in the first two columns, the
patched libraries in the third:

| | USB 2.0 | Thunderbolt 5 | Thunderbolt 5, GPUs kept warm |
|---|---:|---:|---:|
| box, 8 layers (8.3 GB) on the laptop: load | 25.6 s (from the worker's cache) | 15.5 s (no cache) | 34.7 s (no cache) |
| decode | 22.7 tok/s | 25.1 tok/s | 42.3 tok/s |
| a 7,016-token prompt | 285 tok/s | 850 tok/s | 850 tok/s |
| laptop, placed by itself (17 GB on the box): load | 475 s\* | 54 s (no cache) | 56 s (no cache) |
| decode | 25.6 tok/s\* | 28.9 tok/s | 43.6 tok/s |
| a 7,016-token prompt | 254 tok/s\* | 1,403 tok/s | 1,274 tok/s |

\* The day before, with an earlier build and `-split 0.32,0.68` (15.8 GB on the
box); over Thunderbolt that split decoded as fast as placement's.

The box alone decodes at 44.1 tok/s and reads the prompt at 777.

The bandwidth fixed loading and prompts, and barely moved decode. Decode was
never the cable's: a token crosses it in two waits (the asynchronous commands
before them cost about 0.1 ms each), and the rest of the time went to the GPUs.
Each waits while the other computes its layers, and macOS lowers an idle GPU's
clock within milliseconds — the laptop's 8 layers took 6 ms a token at the
start of a reply and 11–14 ms twenty tokens in. The patched libraries
(`scripts/build-libs.sh`) keep a GPU busy with a tiny kernel, one SIMD group for
about 50 µs at a time on a queue of its own, for a second after each graph it
computes; a worker and a `serve -peers` turn it on, and `GGML_METAL_WARM_MS`
overrides the window (0 is off). With the release libraries a split decodes as
in the middle column. On the box alone it is worth 4% (45.5 against 43.8 tok/s),
so a single machine leaves it off.

A model with experts in host memory is never kept warm. The MoE cache waits on
the GPU several times a layer, and on the MacBook Pro a GPU kept busy by anything
else — the keep-warm kernel, or the same kernel from another process — made
those waits long: Flash-Next with 27 layers' experts on the host decoded at 1
tok/s instead of 30, and its reads from the SSD waited 34.5 s in all instead of
4.2. On the box host experts lost nothing to it (25.2 against 25.1 tok/s), but
the rule is the same on both. A worker keeps its GPU warm regardless, so a
machine that runs a worker should not also serve a model with host experts.

llama.cpp's RPC can also use RDMA over Thunderbolt, negotiated at connect, which
would shorten the two waits. It is off on both machines here (`rdma_ctl status`);
turning it on is done from macOS Recovery, and llama.cpp's RPC README says the
address must be on the Thunderbolt interface itself, not a bridge.

A Mac with its firewall on can hold a new worker's incoming connections until the
program is allowed — on the MacBook Pro they sat unanswered until `kinfer` showed
up as permitted in the firewall's list (System Settings › Network › Firewall ›
Options).

### Split by itself

Given `-peers` and no `-split`, kinfer splits the model itself, from the header
and every device's free memory: this machine keeps what it can hold with the room
placement keeps (see [Placed by itself](#placed-by-itself)); each peer in turn
takes the next of the first layers only while the rest would not fit here,
keeping 2 GiB and a pass's activations of its own free; what still does not fit
stays in this machine's host memory — the routed experts of its own last layers,
with a GPU cache. A model that fits here leaves the peers nothing: a layer behind
the cable costs every token a round trip.

llama.cpp left to itself splits by free memory, and sizing measured only this
machine. On the box with the laptop as its peer, that put 16 of Flash-Next's
layers (15.5 GiB) on the laptop with their share of a context sized from the
box's memory — 7.7 GiB of KV cache and a 35.5 GiB compute buffer, 59 GiB on a
40 GiB GPU. The laptop ran out of memory on the first prompt, its worker
aborted, and the server with it. `-place=false` still does that; `-split` still
chooses by hand.

Over the same USB 2.0 cable, Flash-Next IQ3_S, the 7,753-token prompt and three
200-token replies, the same reply from every row that answered:

| this machine | peer | placed | prompt | decode |
|---|---|---|---:|---:|
| box | laptop | nothing on the laptop | 729 tok/s | 43 tok/s |
| box | laptop | llama.cpp's split (`-place=false`): 16 layers on the laptop | — | — (the laptop ran out of memory) |
| box | — | — | 748 tok/s | 43 tok/s |
| laptop | box | layers 0-17 on the box | 258 tok/s | 23 tok/s |
| laptop | — | 28 layers' experts in host memory | 720 tok/s | 28 tok/s |

On this cable the laptop's own SSD beats the box's GPU. kinfer still uses the
peers it is given, so leave them out where the link is this slow; whether a
Thunderbolt 5 link turns that round is the next measurement.

## A mixture of experts that does not fit: experts in host memory

A token uses a few experts per layer — 10 of Flash-Next's 512 — so a model too
big for the GPU can leave some layers' routed experts in host memory, mapped
from the file, and the CPU computes them; llama.cpp b11998 can also keep the ones
in use in a GPU cache (`llama-moe-cache.cpp`), so only a miss is copied:

```bash
kinfer serve -host-experts 12                  # the first 12 layers' experts stay on the host
kinfer serve -host-experts 12 -moe-cache 4096  # and 4 GiB of GPU memory caches the ones in use
kinfer serve -host-experts -1 -moe-cache 16384 # every layer's
```

Measured on the box with Qwen3.8-Flash-Next IQ3_S (48 GB of experts), one slot:
decode is the third of three 200-token replies, by when a cache is warm; the
prompt is 2,116 tokens.

| experts on the host | GPU cache | decode | prompt | decode hit rate | GPU memory used |
|---|---:|---:|---:|---:|---:|
| none | — | 44.2 tok/s | 770 tok/s | — | 52.3 GiB |
| 12 of 48 layers (11 GB) | — | 29.6 | 566 | — | 41.6 GiB |
| 12 of 48 layers | 4 GiB | 32.7 | 511 | 89% | 45.6 GiB |
| all | — | 14.6 | 235 | — | 5.5 GiB |
| all | 8 GiB (17%) | 16.8 | 233 | 81% | 13.5 GiB |
| all | 16 GiB (34%) | 19.9 | 211 | 94% | 21.5 GiB |
| all | 48 GiB (every expert) | 22.2 | 60 | 98% | 52.4 GiB |

The last row is the one to read. With every expert in the cache and almost every
lookup a hit, decode is half of what the same weights do on the GPU directly: the
cache stops at every layer to ask the host which experts the router chose, and a
prompt's large batches copy experts out of it wholesale. It was built for a card
with its own VRAM behind PCIe, where host memory is far away. On Apple Silicon
host memory *is* the GPU's memory, and the cache only adds the copying.

So on a Mac: a model that fits keeps every expert on the GPU. One that does not
puts the fewest layers' experts on the host (`-host-experts N`), where the CPU
computes them from the page cache; a small cache on top buys about a tenth more
decode for a tenth less prompt speed. The host experts are read from the file as
they are touched, so they need RAM outside the GPU's budget to stay fast — past
that, from the SSD.

Placing experts on the host also turns off llama.cpp's CPU repacking
(`--no-repack`): a repacked buffer is not plain host memory, and the cache passed
over every layer that had one — 39 of Flash-Next's 48.

### When the experts do not fit in memory either

All of that holds while the host experts stay in the page cache. On a MacBook Pro
(48 GB, 40 GiB of it the GPU's) Flash-Next IQ3_S does not fit at all, and what is
not on the GPU comes from the SSD. There the cache earns its place: the GPU keeps
the experts in use and the SSD serves the rest. Only, llama.cpp copies a miss out
of the mapped file one page fault at a time, about 0.6 GB/s from an SSD that reads
13 GB/s, so it spends most of its time waiting. kinfer's build of llama.cpp
(`scripts/build-libs.sh`, the patch in `patches/llama.cpp/`) adds what Strata
does on a Mac:

- what stays mapped from the model file is read with `pread` by a pool of
  sixteen threads, in requests of up to 2 MB, instead of copied out of the
  mapping one page fault at a time; the file and offset come from the mapping
  itself, checked against it;
- a layer's misses are read that way before they are copied, and the next cached
  layer's experts are guessed by running its router on this layer's input and
  read into the page cache while the GPU computes the layers in between;
- the experts generation selects are counted into
  `~/.kinfer/profiles/<model>.moeprof` whenever a context is freed (older counts
  halved), and a model loads with its cache already holding the ones counted
  most;
- a prompt's large batches use most experts of every layer, so every pass reads
  the host experts again: each weight is read into one of three slots of host
  memory past the page cache, the next two while the GPU computes, and copied
  into the GPU's copy of it by every thread, the experts the cache holds from the
  cache;
- the rows a batch needs of a per-layer embedding table (Flash-Next's 27 GiB
  n-gram table, DeepSeek-V4.1's) are read the same way before the pass, where
  the CPU would page in one random row at a time.

Measured on the MacBook Pro, one slot: decode is three 200-token replies in
order, the prompt 2,116 tokens. The reply is the same, word for word, in every
row.

| experts on the host | GPU cache | build | decode | prompt | read from the SSD per token |
|---|---:|---|---|---:|---:|
| 16 of 48 layers | — | release | 4.2 / 4.2 / 4.4 | 47 | 130 MB |
| 24 of 48 layers | 8 GiB | release | 6.2 / 8.8 / 9.1 | 38 | 39 MB |
| 24 of 48 layers | 8 GiB | kinfer k1, no profile yet | 19.3 / 23.2 / 22.9 | 103 | 57 MB |
| 24 of 48 layers | 8 GiB | kinfer k1, with a profile | 23.1 / 22.7 / 23.1 | 104 | 58 MB |
| 24 of 48 layers | 8 GiB | kinfer k2, with a profile | 32.3 / 31.0 / 31.3 | 415 | 53 MB |

A longer prompt — the first 7,753 tokens of kinfer's README at the time — run back to back with
each build, the same 32-token reply from all of them:

| build | GPU cache | pass | prompt |
|---|---:|---:|---:|
| release | 8 GiB | 2048 | 62 tok/s |
| kinfer k1 | 8 GiB | 2048 | 159 tok/s |
| kinfer k2 | 8 GiB | 2048 | 705 tok/s |
| kinfer k2 | 6 GiB | 4096 | 740 tok/s |

The 2,116-token prompt above is read in passes of 512: with an 8 GiB cache a
pass of 2048 leaves the laptop's GPU less than the 2 GiB kinfer keeps free, so
the automatic micro-batch steps down and says so. `-batch 2048` takes the room
from that headroom, a smaller `-moe-cache` makes it. A pass of 4096 makes the
GPU's own work faster too (more tokens for each expert), and costs 2.2 GiB more
of it — here the cache's: with 6 GiB decode is 28 tok/s instead of 31.

Most of what k2 gains on a prompt came from two places that were not the experts.
The per-layer table: 16 random rows of 90 bytes per token, which llama.cpp only
`madvise`s before the CPU looks them up — 4 s of page faults for a pass of 4096,
0.3 s read by the pool. And the copies from the cache: llama.cpp copies each run
of cached experts on the GPU with three command buffers, 6.4 s of host time per
prompt while the GPU waited. The SSD reads 13.2 GB/s in requests of a few MB;
the mapping had read it at 1.1-1.5.

An 8 GiB cache over 22.7 GiB of experts hits 94% of the time in generation; the
other 6% is all the SSD reads, and the read-ahead guesses a third of them. The
guess is as wide as the router's choice (10 experts): half as wide again caught
44% of the misses instead of 34%, but read twice as much and decoded at 21.7, and
twice as wide at 19.6 — reads that go unused slow the ones that are needed; with
`pread` half as wide again catches 46% for no gain. Three slots and not four:
the laptop had 5-6 GB in swap, and a fourth slot of 450 MB made the copies ten
times slower through the memory compressor. Filling the cache from the profile
reads 8 GiB in about a second; a context rebuilt for a longer prompt does not
refill it, since warming up by itself costs less than that.

For comparison, the same model with that third on the box's GPU over USB 2.0
instead (`-peers`) decodes at 25.6 tok/s and reads a prompt at 254, slower than
the SSD; over Thunderbolt 5 with both GPUs kept warm it decodes at 43.6 and reads
the prompt at 1,274 (see [Over Thunderbolt 5](#over-thunderbolt-5-and-the-gpu-that-waits)).

### Placed by itself

None of those flags is needed any more. A mixture of experts the GPU cannot
hold, given neither `-host-experts` nor `-moe-cache`, is placed as it loads:

```
placement: Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf has 51.0 GiB of weights; with 6.2 GiB
kept free for its context, a pass of 2048 and the headroom, it needs 17.3 GiB more than the GPU's 40.0 GiB.
The routed experts of its first 28 of 48 layers, 26.2 GiB, stay in host memory, and a 9.0 GiB GPU cache
keeps the ones in use; host memory beside the GPU's part is about 8.0 GiB, so the ones the cache misses
are read from the SSD.
```

The header says how much each layer's experts take, every shard's tensors
counted; the GPU says what it has free. kinfer keeps room for one conversation
of 32768 tokens (or what `-slots` and `-ctx` ask for) with a prompt pool entry,
a pass of 2048 and the 2 GiB it warns below, then leaves the experts of as few
layers on the host as free what is missing plus a GPU cache of a third of them
— the ratio of the best configuration found by hand above. The context is sized
to the room kept, not to half of what is left; and when the measured context
costs more, the pool and the context give way before the pass does, since every
pass reads the host experts again.

Measured back to back on the MacBook Pro, the 7,753-token prompt and three
200-token replies, the same reply from both:

| flags | experts on the host | GPU cache | prompt | decode | context | free after load |
|---|---|---:|---:|---:|---|---:|
| none (two runs) | 28 of 48 layers | 9.0 GiB | 706, 726 tok/s | 27-29 tok/s | 32768, and a pool entry | 2.3 GiB |
| `-host-experts 24 -moe-cache 8192 -batch 2048 -slots 1 -ctx 16384 -prefix off` | 24 of 48 | 8.0 GiB | 712 tok/s | 29 tok/s | 16384 | 0.7 GiB |

`kinfer plan <model>` shows the placement without loading anything.
`-host-experts` and `-moe-cache` still choose by hand, and `-place=false` loads
the model whole. A model that fits is untouched: on the box, Flash-Next, ornith
and qwen3.8-27b load as before. A dense model that does not fit, or a mixture
whose shared part alone does not, is loaded as it is, with a line saying why it
will not fit. The rule was tuned where the host experts come from the SSD; where
they would fit in host memory instead, it has not been measured against the
fewest layers and no cache. With `-peers`, placement decides the split as well:
see [Split by itself](#split-by-itself).

### Against Ollama on the same laptop

Ollama 0.40.13 runs Flash-Next IQ3_S on the MacBook Pro too, imported from the
same two files: `FROM` the directory that holds them, since `FROM` the first
file alone fails ("has 1 shards, expected 2"). It runs llama.cpp's own
`llama-server`, which fits the model by itself: 37.8 GiB on the GPU, 13.2 GiB of
experts in host memory for the CPU to compute, the 27 GiB table mapped, passes
of 512. Ollama passes it no MoE settings, but its `llama-server` takes the ones
kinfer was tuned with by hand, so it was also run directly with them:
`-ngl 99 --fit off --n-cpu-moe 24 --moe-cache-mib 8192 -b 2048 -ub 2048`.

Back to back, the 7,753-token prompt and three 200-token replies, thinking off,
the same reply from all three:

| | prompt | decode |
|---|---:|---:|
| Ollama 0.40.13, as it sets itself up | 17 tok/s (459 s) | 6.9 / 7.0 / 8.4 tok/s |
| Ollama's `llama-server` with kinfer's hand-tuned settings | 81 tok/s (96 s) | 6.3 / 6.7 / 8.2 tok/s |
| kinfer v0.1.0, no flags | 716 tok/s (11 s) | 28.4 / 27.9 / 28.3 tok/s |

The settings lift Ollama's prompt fivefold and leave its decode where it was:
with the experts in host memory, upstream llama.cpp reads a miss from the SSD
one page fault at a time, about 0.6 GB/s. The rest of the difference is
kinfer's patch, described above.

