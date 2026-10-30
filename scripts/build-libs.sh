#!/usr/bin/env bash
# build-libs.sh — build llama.cpp with kinfer's patches and put its libraries
# where kinfer embeds them.
#
#   scripts/build-libs.sh                   the release kinfer is tested against, patched
#   scripts/build-libs.sh llama-b11998.tgz  that release's source, already downloaded
#
# The release libraries (fetch-libs.sh) do everything kinfer needs except serve
# a mixture of experts whose experts do not all fit on the GPU well. For that,
# kinfer patches llama.cpp (patches/llama.cpp/$VERSION): what stays mapped from
# the model file is read with pread by a pool of threads instead of one page
# fault at a time — the MoE cache's misses, the experts a prompt's large
# batches upload, which are read ahead while the GPU computes, and the rows of
# a per-layer embedding table — and a profile of the experts generation used
# fills a fresh cache. The libraries land in
# internal/nativelib/libs/darwin_arm64_${VERSION}k2, which replaces the release
# set the way one release replaces another (see fetch-libs.sh).
#
# Needs cmake, ninja and Xcode's command line tools; about two minutes on an M4.
# Without a source tarball it downloads one from GitHub, about 39 MB.

set -euo pipefail

VERSION=b11998 # the release the patches are made against
LEVEL=k2       # bumped when the patches change, so a cached copy is not reused
SRC=""
for arg in "$@"; do
  case "$arg" in
  -h | --help)
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *) SRC=$arg ;;
  esac
done

case "$(uname -s)/$(uname -m)" in
Darwin/arm64) PLATFORM=darwin_arm64 ;;
*)
  echo "build-libs: only Apple Silicon macOS is set up here; this is $(uname -s)/$(uname -m)" >&2
  exit 1
  ;;
esac

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PATCHES=$ROOT/patches/llama.cpp/$VERSION
LIBS=$ROOT/internal/nativelib/libs
DEST=$LIBS/${PLATFORM}_$VERSION$LEVEL
NAMES=(libllama libggml libggml-base libggml-cpu libggml-blas libggml-metal libggml-rpc libmtmd)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if [[ -z $SRC ]]; then
  SRC=$TMP/llama-$VERSION.tar.gz
  URL=https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/$VERSION
  echo "build-libs: downloading $URL"
  curl -fL --retry 3 -o "$SRC" "$URL"
fi
echo "build-libs: sha256 $(shasum -a 256 "$SRC" | cut -d' ' -f1)"
mkdir "$TMP/src"
tar xzf "$SRC" -C "$TMP/src" --strip-components 1

for p in "$PATCHES"/*.patch; do
  echo "build-libs: applying $(basename "$p")"
  patch -d "$TMP/src" -p1 --forward --quiet <"$p"
done

# The release's own macOS build, minus the server and examples kinfer does not
# embed (.github/workflows/release.yml, macos-cpu).
cmake -S "$TMP/src" -B "$TMP/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3 \
  -DCMAKE_INSTALL_RPATH='@loader_path' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DGGML_RPC=ON >"$TMP/configure.log" ||
  { cat "$TMP/configure.log" >&2; exit 1; }
cmake --build "$TMP/build" --config Release -j "$(sysctl -n hw.logicalcpu)" \
  --target llama ggml ggml-base ggml-cpu ggml-blas ggml-metal ggml-rpc mtmd

# The .0.dylib names, copied through their symlinks; see fetch-libs.sh.
rm -rf "$DEST.partial"
mkdir -p "$DEST.partial"
for n in "${NAMES[@]}"; do
  cp -L "$TMP/build/bin/$n.0.dylib" "$DEST.partial/"
done
rm -rf "$DEST"
mv "$DEST.partial" "$DEST"

# One version per platform; see fetch-libs.sh.
for old in "$LIBS/${PLATFORM}"_*; do
  [[ -d $old && $old != "$DEST" ]] || continue
  echo "build-libs: removing $(basename "$old")"
  rm -rf "$old"
done

du -sh "$DEST" | awk -v v="$VERSION$LEVEL" '{print "build-libs: " v " ready, " $1 " in " $2}'
echo "build-libs: now build: go build -o kinfer ./cmd/kinfer"
