# Native libraries live here

This directory holds the llama.cpp shared libraries that get embedded into the
kinfer binary. It ships empty — the libraries are ~4.6 MB of platform-specific
binaries that change with every llama.cpp release, so they are fetched rather
than committed:

```bash
go run github.com/dianlight/gollama.cpp/cmd/gollama-download \
    -download -copy-libs -libs-dir internal/nativelib/libs
```

That produces a per-platform subdirectory, e.g.:

```
libs/darwin_arm64_b6862/
    libllama.dylib      1.8M
    libggml-metal.dylib  696K
    libggml-cpu.dylib    696K
    …                          8 files, 4.6 MB total
```

`internal/nativelib` then embeds the whole tree with `//go:embed all:libs` and
unpacks it to `~/Library/Caches/kinfer/lib/<platform>/` on first run.

**This file is not decoration.** `//go:embed all:libs` fails at compile time if
the directory does not exist, and git does not track empty directories — so
without a committed file here, a fresh clone would not build.

To cross-compile, fetch every platform first:

```bash
go run github.com/dianlight/gollama.cpp/cmd/gollama-download \
    -download-all -copy-libs -libs-dir internal/nativelib/libs
```
