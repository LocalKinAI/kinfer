#!/bin/bash
# release.sh — build a release of kinfer for Apple Silicon: llama.cpp with
# kinfer's patch (build-libs.sh) inside, the version stamped in, packed as
# dist/kinfer-VERSION-darwin-arm64.tar.gz with its .sha256.
#
#   scripts/release.sh v0.1.0                     the patched llama.cpp from GitHub's source tarball
#   scripts/release.sh v0.1.0 llama-b11998.tgz    that tarball, already downloaded
#
# It builds from a copy of the commit checked out (git archive), so a release
# is a commit, and the checkout's own libraries are left as they are.
set -euo pipefail

VERSION=${1:?usage: scripts/release.sh VERSION [llama.cpp source tarball]}
SRC=${2:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "release.sh: the checkout has uncommitted changes, and a release is built from a commit" >&2
  exit 1
fi
if [ -n "$SRC" ]; then
  SRC=$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git -C "$ROOT" archive HEAD | tar -x -C "$WORK"
"$WORK/scripts/build-libs.sh" ${SRC:+"$SRC"}

NAME=kinfer-$VERSION-darwin-arm64
mkdir -p "$WORK/$NAME" "$ROOT/dist"
(cd "$WORK" && GOOS=darwin GOARCH=arm64 go build -trimpath \
  -ldflags "-s -w -X main.version=$VERSION" -o "$NAME/kinfer" ./cmd/kinfer)
cp "$WORK/LICENSE" "$WORK/README.md" "$WORK/$NAME/"
tar -C "$WORK" -czf "$ROOT/dist/$NAME.tar.gz" "$NAME"
(cd "$ROOT/dist" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256")

"$WORK/$NAME/kinfer" version
ls -l "$ROOT/dist/$NAME.tar.gz"
