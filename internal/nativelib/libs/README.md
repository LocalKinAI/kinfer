# Native libraries live here

This directory holds the llama.cpp shared libraries that get embedded into the
kinfer binary. It ships empty — they are ~8.2 MB of platform-specific binaries
that change with every llama.cpp release, so they are fetched rather than
committed:

```bash
V=b11175
curl -sfL "https://github.com/ggml-org/llama.cpp/releases/download/$V/llama-$V-bin-macos-arm64.tar.gz" \
    | tar xz -C /tmp
mkdir -p internal/nativelib/libs/darwin_arm64_$V
for f in libllama libggml libggml-base libggml-cpu libggml-blas libggml-metal libggml-rpc libmtmd; do
    cp -L /tmp/llama-$V/$f.0.dylib internal/nativelib/libs/darwin_arm64_$V/
done
```

**Copy the versioned sonames, not the plain names.** llama.cpp's macOS builds
reference `@rpath/libggml.0.dylib`, so a directory containing `libggml.dylib`
fails to load with "Library not loaded". The release tarball ships the
unversioned names as symlinks, and `//go:embed` cannot carry symlinks, so `cp -L`
is what puts real files under the names the loader will ask for.

The result is one subdirectory per platform:

```
libs/darwin_arm64_b11175/
    libllama.0.dylib        3.2M
    libggml-metal.0.dylib   2.2M
    libmtmd.0.dylib         1.4M
    …                             8 files, 8.4 MB total
```

`internal/nativelib` embeds the whole tree with `//go:embed all:libs` and
unpacks it to `~/Library/Caches/kinfer/lib/<platform>/` on first run.
`internal/llama` then dlopens `libllama.0.dylib` by absolute path; libllama's
`LC_RPATH` is `@loader_path`, so its siblings resolve from the same directory.

**Keep exactly one directory per platform.** `platformDir` matches on the
`<goos>_<goarch>_` prefix and takes the first hit, so leaving an old build
beside a new one makes which version loads depend on directory order.

**Upgrading llama.cpp means checking the structs.** `internal/llama` mirrors
`llama_model_params`, `llama_context_params`, `llama_batch` and
`llama_sampler_chain_params` field for field. Between b6862 and b10901 both
param structs changed shape. `bind` asserts each size at startup and `Open`
verifies `n_ctx` after every load, so a mismatch is a clear error rather than
silent corruption — but the fix is still to re-transcribe from the new
`include/llama.h`.

**This file is not decoration.** `//go:embed all:libs` fails at compile time if
the directory does not exist, and git does not track empty directories — so
without a committed file here, a fresh clone would not build.
