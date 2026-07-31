# Changelog

## [Unreleased] - 2026-07-24 (第五场) — `kinfer fit`:让运行时替你选模型

选本地模型基本靠猜:量化名不含尺寸、「7B」不告诉你要多少内存,而且**选错不会
干净地失败** —— macOS 会一路 swap 到整机卡死。这个答案运行时自己知道,那就该由它说。

```
$ kinfer fit
Machine
  darwin/arm64 · 10 cores · 16 GB RAM · Apple GPU (Metal, unified memory)

Recommended
  7B at Q4_K_M  (~4.5 GB, comfortable)
  kinfer pull Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M

Backend
  metal — offload every layer with -ngl 99
```

### 设计要点

- **尺寸估算按真实文件校准**,不用名义的 4.83 bits/param:Llama-3-8B Q4_K_M ≈ 4.9 GB
  (0.61 GB/B)、Qwen2.5-7B ≈ 4.7 GB(0.67 GB/B),取 **0.65 GB/B**。小模型因
  embedding 占比大而偏重(实测 0.5B = 0.47 GB,合 0.94 GB/B),但小模型从来不是
  瓶颈,所以系数按「真会把机器压垮的尺寸」调。
- **预留 40% 内存**。系统、浏览器、编辑器动辄几个 GB,而 Apple Silicon 上 **GPU 和
  CPU 抢同一份内存** —— 超了不会报错,只会 swap 到机器爬不动。
- **后端建议来自实测,不是想当然**:M 系列上 Qwen2.5-0.5B **CPU 123.7 tok/s vs
  Metal 114.5 tok/s** —— 权重太小时传输开销盖过算力收益。所以 ≤1.5B 推荐 CPU,
  更大的推荐 Metal。这条数据是 Phase 0 量出来的,现在变成了产品建议。
- **已装模型按同一预算判定**(comfortable / tight / too large),不用自己算。

8 个单测,含「估算必须贴近真实文件」「推荐必须真的装得下预算」「0.5B 该选 CPU」
「读不到内存时优雅降级而不是瞎推荐」。

### 零新依赖

读物理内存本来要 `golang.org/x/sys`(标准库只有 `SysctlUint32`,>4GB 会溢出)。
为了不给「单文件零依赖」这个卖点开第一个口子,改用 `sysctl -n hw.memsize`
—— macOS 自带,而且 `fit` 一次只跑一遍。Linux 走 `/proc/meminfo`。

## [Unreleased] - 2026-07-24 (第四场) — Phase 1:CLI + HTTP 服务成型

`kinfer` 从「一个验证程序」变成**能用的命令行工具**:

```
kinfer pull <repo>[:quant]    从 Hugging Face 拉模型
kinfer list / rm              管理本地模型
kinfer run <model> [prompt]   命令行生成
kinfer serve                  HTTP 服务(Ollama + OpenAI 双方言)
```

### 新增

- **`internal/engine`** —— 一个模型的完整生命周期。CLI 和 HTTP 服务共用它,
  生成逻辑只有一处、不会两边漂移。**每次对话前 `Memory_clear`** 清 KV cache ——
  漏掉这步,第 N+1 轮会静默继承第 N 轮的状态,模型回答一个没人问过的问题。
  生成串行化(一个 llama context 的 KV cache 是可变状态,两个对话共用必然互相污染)。
- **`internal/store`** —— 模型目录。**存成人能读的文件名**,不是 Ollama 那种 blob
  哈希:凌晨两点出事时你要能 `ls` 看出有什么、能 `scp` 拷到另一台机器。
  `Resolve` 支持唯一前缀(`kinfer run qwen`),**歧义时报错而不是挑一个** ——
  静默加载错的 7B 模型比报错更糟。9 个单测,含「`rm` 拒绝删除目录外的文件」。
- **`internal/hub`** —— 从 HF 拉 GGUF。**中间没有 registry**:repo id 就是模型名。
  下载先落 `.part` 再 rename,中断的 pull 永远不会留下看起来能用的半个模型。
- **`internal/server`** —— **Ollama + OpenAI 双方言**。Ollama 那套是关键:
  LocalKin 的 ~300 个 soul 已经写着 `provider: "ollama"`,**切过来只改一个 endpoint,
  切回去也是同一个改动**。四个端点全部 curl 验证通过,两种流式(NDJSON / SSE)都对。

### ⛔ 上游第五个缺陷:context params 结构体整体错位

新版 llama.cpp 从 `llama_context_params` 里**删掉了 `seed` 字段**(种子归采样器管),
但 gollama 的 Go 镜像第一个字段仍是 `Seed` —— **后面每个字段都错位一格**。
赋给 `NCtx` 的值实际写进了 `n_batch`,`NBatch` 写进 `n_ubatch`,以此类推。
**不报错,参数就是不生效。**

直接证明:

```go
cp.Seed = 8192; cp.NCtx = 111   →   llama_context: n_ctx = 8192
```

后果不是学术问题:服务 LocalKin 的 soul 时报「prompt is 8267 tokens but the
context holds 4096」,无论 `-ctx` 传什么都一样,因为真实 context 一直是
llama.cpp 的 512 默认值。

绕过 = `internal/engine/ctxparams.go`(把尺寸写进真正到达 `n_ctx` 的字段),
并**绑定 `llama_n_ctx` 做自检** —— 上游一旦修好,这个 workaround 会反过来变成
写 seed、悄悄恢复原 bug,所以 `Open()` 会核对实际分配的 context 并在不符时报错。

### ⚠️ 未解决:LocalKin → kinfer 端到端

**kinfer 侧全部正常**,已逐项验证:

- 四个 API 端点 curl 全通,含 LocalKin 的确切请求格式(`stream:true` + `think` + `tools:[]`)
- 同样长度的 soul 当 system prompt,kinfer 自己答得很好:
  「施舍是信仰的试金石。你可以通宵祷告、禁食整年、读遍圣经——但如果你看见穷人而不怜悯,
  你就是假基督徒。」
- LocalKin 能连上:`Checking ollama connection... OK` + `Model "..." is available`
- **kinfer 确实收到了 LocalKin 的请求**(日志 `→ /api/chat from [::1]:49419`)

但 LocalKin 拿到的 content 是空的。已排除:tool calling(去掉 skills 仍然空)、
技能数量(只留 5 个仍然空)、context 大小(16384 仍然空)、prompt 长度(短 soul 仍然空)。

**下次从这里查**:在 kinfer 里 dump 收到的请求体与发出的响应体,对比 LocalKin
`pkg/brain/ollama.go` 的流式解析期望。问题被压缩到「kinfer 收到请求之后、
LocalKin 解析响应之前」这一段。⚠️ **修 kinfer 不改 localkin** —— 今天全程遵守,
测试用的是 soul 副本 + 临时 agent(:9999),localkin 仓一行未动。

## [Unreleased] - 2026-07-24 (第三场) — 自绑原生符号,三个上游缺陷一次解决

不再等上游。`internal/llama` 用 purego 直接绑定 gollama 缺失或做错的五个 llama.cpp
入口 —— **仍然零 CGO**,交叉编译能力不受影响。

```
✅ vocabulary  151,936 tokens   ← gollama 硬编码报 32
✅ 输出连贯,无重复循环
✅ 99 tokens (hit stop marker)  ← 模型自己说完就停
```

同一个问题、同一个 soul,前后对比:

```
之前  财是人的命根，是人的命脉，是人的命关。财是人的命根…（打转到 token 耗尽）
现在  钱财是上帝的礼物，不是用来炫耀或占有…
      在教会中，我们教导人们要谨慎地使用我们的金钱…（说完自然停止）
```

### 新增 `internal/llama`

| 绑定 | 解决 |
|---|---|
| `llama_vocab_n_tokens` | 拿到**真实词表 151,936**,采样器终于能接上 |
| `llama_token_to_piece` | **真** detokenizer,不再需要 byte-level BPE 后处理 |
| `llama_vocab_is_eog` | 真正的结束判定,取代「在解码文本里找停止串」的猜测 |
| `llama_model_get_vocab` / `llama_vocab_eos` | 上面三个的前置 |

`RegisterLibFunc` 缺符号会 panic(遇到不兼容的 llama.cpp 构建就整个程序崩),
已包成 error 返回。`Dlopen` 用 `RTLD_GLOBAL` —— 若加载出第二份 libllama 副本,
两份各有各的状态,句柄互相看不懂。

### 现在的状态

四个上游缺陷,**三个已用正解绕过**(自己绑定,不是打补丁):
detokenize ✅ · 词表大小 ✅ · 采样器 ✅ · `Config.LibraryPath` 仍靠 chdir(见下)。

### ⚠️ 代价:吞吐 130 → 40.5 tok/s

每步要对 **151,936** 个候选做全排序(之前只有 32 个,快是因为它错)。
`internal/sampling` 的 `sort.Slice` 是明显瓶颈 —— top-k=40 根本不需要全排,
用 partial selection(`container/heap` 或 quickselect)可以把 O(n log n) 降到 O(n)。
**正确性先行,性能是下一步。**

### `internal/bpe` 现在冗余

有了真 detokenizer,byte-level 解码不再需要。**保留不删**:它是 fallback,
也是那个 bug 的现场记录(3 个单测仍然通过)。

## [Unreleased] - 2026-07-24 (下半场) — chat template 通了,采样器撞上上游硬编码

### 新增

- **`internal/chat` —— chat template。** 这是「续写」与「对话」的分界:裸文本喂给
  instruct 模型,它只会把你的话往下写;套上它被训练的模板,它才作为 assistant 回答、
  **并且听 system prompt**。对 kinfer 这是命门 —— **LocalKin 的 soul 就是 system prompt**,
  没有模板,300+ 个人格一个都用不了。内置 chatml / llama3 / mistral 三族(mistral 无 system
  角色,按官方做法折进首个 user turn),按文件名 `Detect()`,默认 chatml(最常见也最宽容)。
  9 个单测,含「prompt 必须停在 assistant 开头」「多轮顺序」「停止标记取最早的那个」。
- **`internal/sampling` —— 自研采样器**(temperature / top-k / top-p / repeat penalty,
  llama.cpp 的标准顺序)。10 个单测,含**「repeat penalty 能打破循环」**、负 logit 要
  **乘**而不是除(除会让它变大 = 奖励重复)、softmax 数值稳定性、同 seed 可复现。
  ⚠️ **写完了、测过了,但接不上** —— 见下。

### 验证

- **soul 生效了。** 同一个问题,加上金口约翰的 system prompt 后,prompt 从 13 → **100 tokens**,
  模型开口即「我是一个讲道家」。**chat template 这条链路是通的**,这是 kinfer 能接 LocalKin
  的前提条件。

### ⛔ 上游第四个缺陷:词表大小被硬编码成 32

```go
// gollama.cpp v0.2.2, Token_data_array_from_logits:
// Use hardcoded vocabulary size for now to avoid corruption issues
nVocab := int32(32)
```

Qwen2.5 的词表是 **151,936**。这个函数只返回**前 32 个候选**,
`Token_data_array_init` 则是另一个硬编码的 256。于是:

- **自研采样器无法接入** —— 在 32 个候选上跑 top-k=40 + repeat penalty,好 token 会被
  迅速惩罚光,输出退化成 `"1.4,25,38"(17-26,40)` 无限重复,**比 greedy 还糟**。
  已回退到 gollama 的 greedy,代码与注释保留在 `cmd/probe`,等词表问题解决再接。
- **greedy 的重复问题因此无解** —— 实测 0.5B 上「财是人的命根，是人的命脉，是人的命关。」
  会一直重复到 token 预算耗尽。这不是模型的锅,是**只有 greedy 可用**。

**出路三条,尚未决定:**
1. **kinfer 自己绑 `llama_vocab_n_tokens`** —— 它已经知道库路径(`nativelib`),
   用 purego 补几个函数即可。最干净,也最符合「不受上游缺陷限制」。
2. **换 `tcpipuk/llama-go`**(CGO 版,活跃维护)—— 但会失去 purego 带来的交叉编译能力。
3. 给上游提 PR —— 最慢,但连同前三个 bug 一起提是像样的贡献。

**这个发现改变了对 gollama.cpp 的判断**:它能跑通推理(Phase 0 成立),但**上层能力残缺**,
kinfer 不能把它当成完整地基,得准备好自己补绑定层。

## [Unreleased] - 2026-07-24 — Phase 0：可行性验证通过,单文件推理成立

**起因**:2026-07-23 `kimi-k2.5:cloud` 在 Ollama 云端**挂起**(不是报错,是不返回),
LocalKin 130 个 fleet agent 集体失声半天。真正的教训不是"选错了模型",而是
**只有一条腿**——所有 agent 的推理路径都压在同一个外部服务上。kinfer 是第二条腿。

Phase 0 只回答一个问题:**这条路走不走得通**。答案是通,而且比预期便宜一个量级。

### 验证结论

- **纯 Go(零 CGO)推理跑通** —— 走 `dianlight/gollama.cpp`(purego + libffi),
  `GOOS=linux go build` 从 Mac 交叉编译的能力得以保留,binary 不依赖 C 工具链。
- **Metal 真实生效** —— 首次运行日志里是 llama.cpp 在编译 Metal kernel;缓存后模型
  加载从 5.46s 降到 **0.14s**。
- **单 binary 成立(铁证)** —— 10.6M 一个文件,在「gollama 缓存移走 + kinfer 缓存清空 +
  项目 libs 目录移走 + binary 拷到项目外」的干净环境下,照样 89.8 tok/s 跑通。
- **中文正确** —— 多字节字符跨 token 的流式解码验证通过,无乱码无半字符。

### 实测(M 系列,Qwen2.5-0.5B Q4_K_M)

| 后端 | 吞吐 | 模型加载 |
|---|---|---|
| CPU (`-ngl 0`) | **123.7 tok/s** | 0.13s |
| Metal 冷启 | 75.8 tok/s | 5.46s(含一次性 kernel 编译) |
| Metal 热启 | 114.5 tok/s | **0.14s** |

**0.5B 模型上 CPU 比 Metal 快** —— 小模型的 GPU 传输开销盖过了计算收益。这不是缺陷,
是应该由运行时自动决定、而不是让用户去配的事。**自动选后端**因此进了路线图,
上面这组数字就是理由。

### 新增

- **`internal/nativelib` —— 单文件分发的核心。** `//go:embed all:libs` 把 4.6M × 8 个
  dylib 打进 binary,首次运行解压到 `~/Library/Caches/kinfer/lib/<platform>/`,
  写入走「临时文件 + rename」以防两个进程同时启动时读到半个 dylib,
  内容比对用 sha256 而非 mtime(被 kill 的进程会留下坏文件)。
- **`internal/bpe` —— byte-level BPE 解码器。** GPT-2 系词表的 256 字节双射逆表,
  3 个单测覆盖:空格还原(`ĠMerkle` → ` Merkle`)、CJK 跨 token 拼接、字节表双射性。
- **`cmd/probe` —— 可行性验证程序。** 一个程序同时测加载/推理/Metal/吞吐,
  流式解码累积原始 piece、只输出已构成合法 UTF-8 的部分。

### 上游两个 bug(已绕过,值得提 PR)

**① `Token_to_piece` 名不副实。** 它调的是 `llama_vocab_get_text`(词表原始条目)
而不是 `llama_token_to_piece`(会做 byte-level 解码)。于是输出是 `ĠMerkleĠtree`
而不是 ` Merkle tree`。绕过 = `internal/bpe`。

**② `Config.LibraryPath` 设了等于没设。** `ApplyConfig` 里只 `UnloadLibrary()`,
**从来没把新路径存进 loader**,下次加载仍走默认搜索。
进程内 `os.Setenv("DYLD_LIBRARY_PATH")` 同样无效——动态链接器在 exec 时就读完了。
绕过 = `Backend_init` 期间 **chdir 进库目录**(loader 用裸名字 dlopen,会搜 CWD),
完事立刻恢复。⚠️ chdir 是进程级的,注释里写明了并发约束;上游修好后这个 hack 应删。

### 操作上的坑(踩过,记下来省下次)

- **`go get @latest` 拿到的是 v0.1.0。** 最新的 `v0.2.2-llamacpp.b6862` 带 `-` 后缀,
  被 Go 当成 prerelease 而跳过,**必须显式 pin**。
- **包内的 `//go:embed libs/**` 对使用者无效。** module cache 只读,下游项目永远填不进去。
  把 libs 放项目根目录再 build,**binary 大小一个字节都不变**——这就是 `nativelib` 存在的理由。
- **首次运行不会自动下载库。** 文档说会,实测不会,要先跑 `gollama-download -download`。
- **`-copy-libs` 必须配合 `-download`**,单独用只会打印 usage。
- **Metal kernel 编译日志刷屏**,目前只能 `2>/dev/null`;gollama 的 `Config.EnableLogging`
  是 TODO,`llama_log_set` 尚未绑定。

### 尚未做

HTTP 服务、模型管理 CLI、从 HF 拉模型、日志静音,以及这个项目真正的理由——
**健康探测 / 自动降级 / 熔断 / 按 agent 路由**。今天只证明了地基能承重。
