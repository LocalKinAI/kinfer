# For agents

What kinfer does for agents that send the same long prompts, call tools and think out loud — part of [kinfer](../README.md).

## Prefix pool

An agent's system prompt is identical byte for byte on every turn it ever takes,
and a soul runs to thousands of tokens. Without help, each turn prefills all of
it again, and a hundred agents sharing a soul pay for it a hundred times.

Recently seen prompt prefixes stay resident in the KV cache under their own
sequence ids. A request that begins the same way adopts those cells rather than
recomputing them — and nothing is copied: `llama_memory_seq_cp` tags cells that
already exist with a second sequence id, so taking over a 4000-token prefix
costs a pointer walk. It needs llama.cpp's unified KV buffer, which is what
allows a cell to belong to more than one sequence.

Measured on the M3 Ultra with a 3,100-word system prompt, time to first token:

| | no pool | prefix pool |
|---|---|---|
| same soul, repeated | 266 ms | **17 ms** |
| multi-turn conversation, turns 2–5 | 283 ms | **22 ms** |
| **8 agents sharing a soul, concurrent** | 2.38 s wall, 1767 ms median | **0.16 s wall, 68 ms median** |
| six distinct souls through four pool slots | 91 ms | 94 ms |

The last row is the control that matters: prefixes that do not repeat get no
benefit, and pay no penalty for the attempt.

```
$ kinfer serve -slots 8 -prefix 4
  8 slots × 4096 tokens — conversations share one forward pass
  4 prefix slots — a repeated system prompt is prefilled once
```

Each pooled prefix costs a sequence's worth of KV cache, the same as a slot, so
this is memory traded for latency — unless memory is short and nothing was
asked for, in which case the pool shares the slots' cells (see sizing above). A match must leave at least one token to
decode — logits exist only for tokens that went through a forward pass, and a
prompt adopted whole would have nothing to sample the first reply token from.

**Reuse is correct but not bit-exact, and that is worth knowing before turning
it on.** How much of a prompt gets adopted depends on what the pool happens to
hold: the first request prefills its question in one batch of a dozen tokens,
and a repeat decodes a single token instead. Different batch shapes mean
different floating-point reduction orders, so a logit that was a near-tie can
land the other way.

Measured: with the pool off, the same question produced one output in ten runs;
with it on, two. Across twenty varied prompts nineteen were byte-identical and
one differed — an arithmetic question the model was genuinely torn on, where the
pooled answer happened to be the better one. Nothing is misaligned: a prompt
reused in full still lets the model quote the last sentence of its own system
prompt, which a position error would destroy. What is lost is the guarantee that
a fixed seed reproduces a previous run exactly, so `-prefix off` turns the pool
off for callers who need it.

`-prefix` takes `auto` (the default), `off`, or a number of entries. A number is
taken as given. `auto` gets only the memory the conversations leave: with
`-slots` and `-ctx` both set, the pool takes cells of its own if they fit, shares
the slots' cells if only that fits, and is left out otherwise — `-prefix 0`
used to mean "sized with the rest", and on a 96 GB box running a 73.4 GiB model
that sized four entries of 16384 into 0.0 GiB. `0` now means none, like `off`.

## Chat templates come from the model

An instruct model answers properly only when the prompt is wrapped the way it
was fine-tuned to see. kinfer used to guess that from the filename and knew
three families, so anything else silently got ChatML — a format it had never
seen. The template it was actually trained on is right there in the GGUF's
metadata, so kinfer reads it.

`llama_chat_apply_template` does not run Jinja. It matches the stored template
string against the families llama.cpp implements in C++ and applies that one, so
binding it buys **54 families** without shipping a template engine.

The difference, on Gemma 3 — which has no system role at all:

```
from the GGUF   <start_of_turn>user\nYou are terse.\n\nHi.<end_of_turn>\n<start_of_turn>model\n
guessed         <|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\n…
```

The guess does not merely look wrong, it shows: asked the same question through
each, the model's own template stops cleanly while the guessed one leaks
`<|im_end` into the reply, because ChatML's marker is not what this model was
taught to stop at.

Generation stops on an end-of-generation token — `llama_vocab_is_eog` reports
whichever terminator the model uses — so a template read from metadata needs no
stop strings of its own. The three hand-written families remain as a fallback
for a file that carries no template, or one llama.cpp does not implement, and
`-template` still forces one.

## Tool calls

A model can ask for a function to be run, in both dialects:

```
POST /api/chat   {"tools": [{"type":"function","function":{"name":"get_current_weather", …}}]}

{"message": {"role":"assistant", "content":"",
             "tool_calls":[{"function":{"name":"get_current_weather",
                                        "arguments":{"city":"Berlin","unit":"celsius"}}}]},
 "done": true, "done_reason": "tool_calls"}
```

Send the result back as a `tool` message and the model answers from it — *"The
current weather in Berlin is light rain with a temperature of 7 degrees
Celsius."*

`llama_chat_apply_template` takes only `{role, content}` — it has no tools
parameter — so the tool branches of a model's Jinja template are unreachable
through the C API. What is reachable is the convention those branches encode,
and kinfer reproduces it exactly as Qwen's own template writes it: the
signatures join the end of the system prompt, an assistant's calls become
`<tool_call>` blocks in its text, and a result becomes a user turn wrapped in
`<tool_response>`, because no instruct model has a `tool` role of its own. The
template then has nothing unusual left to render.

**There is no single convention**, so kinfer reads which one a model knows out
of its own chat template and both declares and parses in that one. Three are
implemented:

| | declaration | a call looks like |
|---|---|---|
| **hermes** | "You may call one or more functions…" | `<tool_call>{"name":…,"arguments":{…}}</tool_call>` |
| **function-xml** | "You have access to the following functions:" | `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` |
| **dsml** | "## Tools … You can invoke tools by writing a…" | `<｜DSML｜ calls><｜DSML｜ invoke name="f"><｜DSML｜ parameter name="k" string="true">v</｜DSML｜ parameter>…</｜DSML｜ calls>` |

DSML is DeepSeek V4.1's, and it comes with a prompt format llama.cpp has no
family for: matched against its built-in templates, V4.1's would pass for
DeepSeek-V3's, without the system token, the think markers or DSML. So for a
`deepseek41` model kinfer renders the whole prompt itself, porting the encoder
DeepSeek ships with the model (`encoding/encoding.py`), and is tested byte for
byte against the conversations DeepSeek publishes beside it
(`internal/chat/testdata/deepseek-v4.1`). The prompt opens the reply's think
block, so kinfer reports the `<think>` the model never writes, and the reply
reads like any other reasoning model's. Earlier turns' thinking, sent back as
`thinking` or as DeepSeek's `reasoning_content`, is rendered where V4.1 expects
it: in every turn when functions are declared, in the current one otherwise.

Getting this wrong is not a near miss. Told to answer in JSON when it had been
taught the XML form, `ornith-1.5:35b` produced `{"name":_get_weather"` —
deterministically, four runs out of four. It was trying to obey an instruction
that contradicted its training. Reading the convention from the template instead
makes the same model answer correctly, with `days: 3` arriving as a number
rather than the string the XML form carries.

Both conventions wrap their calls in `<tool_call>`, so only the inner form tells
them apart — checking the outer tag first would misread every XML model as
Hermes.

A model whose template describes neither is refused rather than answered
uselessly, since it would reply in prose however the functions are declared:

```
$ curl … -d '{"model":"gemma","tools":[…]}'
{"error":"this model's chat template does not use the <tool_call> convention,
          so it cannot answer with tool calls"}
```

A reply carrying functions is not streamed. A tool call is JSON spread over many
tokens, and streaming it would hand the client fragments of syntax; with
functions on the table the reply is buffered and delivered once, parsed. Without
them, streaming is unchanged.

## Reasoning models

A reasoning model writes its working out before it answers, wrapped in `<think>`
tags. That text is not the reply: a client showing replies verbatim shows the
model talking to itself, and — worse for a fleet — a tool call *decided on*
inside the thinking block is not a tool call, so a caller waiting for one sees
prose. kinfer separates the two, as Ollama does.

```
POST /api/chat  {"think": true, …}

{"message": {"role":"assistant", "content":"4",
             "thinking":"The user asks \"What is 2+2?\" and wants just the number…"}}
```

Without `think` the model thinks, and the working out comes back in `thinking`
as it does from Ollama; `content` is the answer and nothing else either way.
`"think": false` keeps the model out of its think block altogether. Dropping the
thinking when the field was absent — the old behaviour — returned an empty reply
with nothing to show for it whenever a reasoning model ran out of time.

A level — `"think": "low"`, `"high"` or `"max"`, or OpenAI's
`"reasoning_effort"`, where `"none"` is off — is passed to models that take a
reasoning budget. DeepSeek V4.1 does, as a line at the top of its prompt
("Reasoning Effort: 50", 75 or 100, or any number from 1 to 100); models
without one ignore it.

A reasoning model can spend an entire token budget thinking and return an empty
answer, so a reply cut short by the budget reports `done_reason: "length"` —
`finish_reason` in the OpenAI dialect — rather than `"stop"`. Told a truncated
reply completed normally, a caller has no way to tell it from a model that had
nothing to say. Streaming separates them as they arrive rather than
buffering the reply. The OpenAI dialect returns it as both `reasoning`, the
field Ollama's own `/v1` writes, and `reasoning_content`, the one DeepSeek
introduced and most clients now look for.

Thinking is split before tool calls are parsed, so a `<tool_call>` the model
merely considered while reasoning is not mistaken for one it made. All three
arrive separately:

```
tool_calls  [{"function":{"name":"get_weather","arguments":{"city":"Berlin"}}}]
content     ""
thinking    "The user is asking about the weather in Berlin. I have a tool available…"
```
