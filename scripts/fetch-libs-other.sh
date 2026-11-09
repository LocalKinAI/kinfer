#!/usr/bin/env bash
# fetch-libs-other.sh — llama.cpp's CPU libraries for Linux or Windows, where
# kinfer embeds them.
#
#   scripts/fetch-libs-other.sh linux amd64 [libgomp.so.1]
#   scripts/fetch-libs-other.sh windows arm64
#
# These are upstream's release builds, unpatched: the Metal and SSD-cache patches
# in scripts/build-libs.sh do not carry over. They land in
# internal/nativelib/libs/<os>_<arch>_<build>, and every directory there is
# embedded, so a release build should have only its own platform's.
#
# Linux wants libgomp.so.1, which a minimal system does not have. Pass a copy to
# carry it in the binary, taken from Ubuntu 22.04: it needs glibc 2.34, as
# upstream's x86-64 libraries do, and anything newer would raise that floor.
#
# Windows wants the Visual C++ runtime (msvcp140, vcruntime140) on the machine.

set -euo pipefail

VERSION=b11998 # the build internal/llama was checked against
OS=${1:-}
ARCH=${2:-}
GOMP=${3:-}

case "$OS/$ARCH" in
linux/amd64) ASSET=ubuntu-x64 EXT=tar.gz SHA=d31bfcd5f85c380ab44f0a07a0ebf6d41bad7a63f84d15ba214de5e5ef1bdae6 ;;
linux/arm64) ASSET=ubuntu-arm64 EXT=tar.gz SHA=66f9fa2544d6ee80f2eae69dac0ed37ecd73fd54022825b4984e184eab5c91af ;;
windows/amd64) ASSET=win-cpu-x64 EXT=zip SHA=2cdb7ff52769dff9bf19c4cf7d6b6e136e7b46249679d5d92ecd92631b8f7fb2 ;;
windows/arm64) ASSET=win-cpu-arm64 EXT=zip SHA=4ced3bd7ea611b4f7731da7d314cc36953cff7e50643369fc4ae9a14b53443e4 ;;
*)
  echo "usage: fetch-libs-other.sh <linux|windows> <amd64|arm64> [libgomp.so.1]" >&2
  exit 2
  ;;
esac

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=$ROOT/internal/nativelib/libs/${OS}_${ARCH}_${VERSION}
NAME=llama-$VERSION-bin-$ASSET.$EXT
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl -fsSL -o "$TMP/$NAME" "https://github.com/ggml-org/llama.cpp/releases/download/$VERSION/$NAME"
if command -v sha256sum >/dev/null; then
  GOT=$(sha256sum "$TMP/$NAME" | cut -d' ' -f1)
else
  GOT=$(shasum -a 256 "$TMP/$NAME" | cut -d' ' -f1)
fi
if [ "$GOT" != "$SHA" ]; then
  echo "fetch-libs-other: $NAME has sha256 $GOT, want $SHA" >&2
  exit 1
fi

mkdir "$TMP/x"
rm -rf "$DEST"
mkdir -p "$DEST"
case "$OS" in
linux)
  tar -xzf "$TMP/$NAME" -C "$TMP/x"
  X=$TMP/x/llama-$VERSION
  # These four are linked against by soname; the backends are found by a scan
  # for libggml-*.so and carry no version.
  cp -L "$X/libllama.so.0" "$X/libggml.so.0" "$X/libggml-base.so.0" "$X/libmtmd.so.0" "$DEST/"
  cp "$X"/libggml-rpc.so "$X"/libggml-cpu-*.so "$DEST/"
  if [ -n "$GOMP" ]; then
    cp "$GOMP" "$DEST/libgomp.so.1"
  else
    echo "fetch-libs-other: no libgomp.so.1 given; the binary will need it on the machine" >&2
  fi
  ;;
windows)
  unzip -q "$TMP/$NAME" -d "$TMP/x"
  X=$TMP/x
  cp "$X/llama.dll" "$X/ggml.dll" "$X/ggml-base.dll" "$X/mtmd.dll" "$X/ggml-rpc.dll" "$X"/ggml-cpu*.dll "$X/libomp.dll" "$DEST/"
  ;;
esac
chmod 755 "$DEST"/*

echo "fetch-libs-other: $(ls "$DEST" | wc -l | tr -d ' ') libraries in $DEST"
