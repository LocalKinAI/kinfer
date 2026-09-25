#!/usr/bin/env bash
# fetch-libs.sh — put llama.cpp's libraries where kinfer embeds them.
#
#   scripts/fetch-libs.sh            the build kinfer is tested against
#   scripts/fetch-libs.sh b11200     another llama.cpp release
#   scripts/fetch-libs.sh -f         fetch again even if it is already there
#
# kinfer binds llama.cpp directly and embeds its eight dynamic libraries
# (about 8.8 MB) in the binary, so nothing needs installing where it runs. They
# are not committed: every llama.cpp upgrade is a new set, and git would keep
# each one forever. This is the one command that replaces committing them.
#
# Only one version may sit in internal/nativelib/libs at a time. kinfer finds
# its libraries by the platform prefix (darwin_arm64_), and with two of them it
# would take whichever the directory listing returned first. So a new version
# replaces the old one.
#
# Before moving to another build, diff llama.h against the embedded one:
# kinfer's structs are transcribed from it (internal/llama), and a field that
# moved would pass the build and fail at the first call. See the upgrade to
# b11175 in CHANGELOG.md for how that was checked.

set -euo pipefail

VERSION=b11175 # the build internal/llama was checked against
FORCE=0
for arg in "$@"; do
  case "$arg" in
  -f | --force) FORCE=1 ;;
  -h | --help)
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  b[0-9]*) VERSION=$arg ;;
  *)
    echo "fetch-libs: unknown argument $arg (want a build like b11175, or -f)" >&2
    exit 2
    ;;
  esac
done

case "$(uname -s)/$(uname -m)" in
Darwin/arm64) PLATFORM=darwin_arm64 ASSET=macos-arm64 ;;
*)
  echo "fetch-libs: only Apple Silicon macOS is set up here; this is $(uname -s)/$(uname -m)" >&2
  exit 1
  ;;
esac

ROOT=$(cd "$(dirname "$0")/.." && pwd)
LIBS=$ROOT/internal/nativelib/libs
DEST=$LIBS/${PLATFORM}_$VERSION
NAMES=(libllama libggml libggml-base libggml-cpu libggml-blas libggml-metal libggml-rpc libmtmd)

have_all() {
  for n in "${NAMES[@]}"; do
    [[ -s $DEST/$n.0.dylib ]] || return 1
  done
}

if [[ $FORCE == 0 ]] && have_all; then
  echo "fetch-libs: $VERSION is already in place ($DEST)"
else
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  URL=https://github.com/ggml-org/llama.cpp/releases/download/$VERSION/llama-$VERSION-bin-$ASSET.tar.gz
  echo "fetch-libs: downloading $URL"
  curl -fL --retry 3 -o "$TMP/llama.tar.gz" "$URL"
  echo "fetch-libs: sha256 $(shasum -a 256 "$TMP/llama.tar.gz" | cut -d' ' -f1)"
  tar xzf "$TMP/llama.tar.gz" -C "$TMP"

  # The release ships each library as a versioned file behind two symlinks.
  # Copy the .0.dylib names with -L: llama.cpp's macOS builds reference
  # @rpath/libggml.0.dylib, and //go:embed cannot carry symlinks.
  rm -rf "$DEST.partial"
  mkdir -p "$DEST.partial"
  for n in "${NAMES[@]}"; do
    src=$TMP/llama-$VERSION/$n.0.dylib
    if [[ ! -e $src ]]; then
      echo "fetch-libs: $n.0.dylib is not in the $VERSION release" >&2
      exit 1
    fi
    cp -L "$src" "$DEST.partial/"
  done
  rm -rf "$DEST"
  mv "$DEST.partial" "$DEST"
fi

# One version per platform; see the top of this file.
for old in "$LIBS/${PLATFORM}"_*; do
  [[ -d $old && $old != "$DEST" ]] || continue
  echo "fetch-libs: removing $(basename "$old")"
  rm -rf "$old"
done

du -sh "$DEST" | awk -v v="$VERSION" '{print "fetch-libs: " v " ready, " $1 " in " $2}'
echo "fetch-libs: now build: go build -o kinfer ./cmd/kinfer"
