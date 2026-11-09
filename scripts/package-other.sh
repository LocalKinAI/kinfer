#!/usr/bin/env bash
# package-other.sh — one Linux or Windows package of a kinfer release, into dist/
# with its .sha256:
#
#   scripts/package-other.sh v0.3.0 linux amd64    kinfer-v0.3.0-linux-amd64.tar.gz
#   scripts/package-other.sh v0.3.0 windows arm64  kinfer-v0.3.0-windows-arm64.zip
#
# Like release.sh it builds from a copy of the commit checked out, so a package
# is a commit and carries only its own platform's libraries. These are upstream's
# CPU builds; the GPU patches are Metal's.
set -euo pipefail

VERSION=${1:?usage: scripts/package-other.sh VERSION <linux|windows> <amd64|arm64>}
OS=${2:?usage: scripts/package-other.sh VERSION <linux|windows> <amd64|arm64>}
ARCH=${3:?usage: scripts/package-other.sh VERSION <linux|windows> <amd64|arm64>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "package-other: the checkout has uncommitted changes, and a package is built from a commit" >&2
  exit 1
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git -C "$ROOT" archive HEAD | tar -x -C "$WORK"
rm -rf "$WORK"/internal/nativelib/libs/*_*
bash "$WORK/scripts/fetch-libs-other.sh" "$OS" "$ARCH"

NAME=kinfer-$VERSION-$OS-$ARCH
EXE=
[ "$OS" = windows ] && EXE=.exe
mkdir -p "$WORK/$NAME" "$ROOT/dist"
(cd "$WORK" && GOOS=$OS GOARCH=$ARCH CGO_ENABLED=0 go build -trimpath \
  -ldflags "-s -w -X main.version=$VERSION" -o "$NAME/kinfer$EXE" ./cmd/kinfer)
cp "$WORK/LICENSE" "$WORK/README.md" "$WORK/THIRD_PARTY.md" "$WORK/$NAME/"

if [ "$OS" = windows ]; then
  ARCHIVE=$NAME.zip
  PY=$(command -v python3 || command -v python)
  (cd "$WORK" && "$PY" -m zipfile -c "$NAME.zip" "$NAME/")
  mv "$WORK/$NAME.zip" "$ROOT/dist/$ARCHIVE"
else
  ARCHIVE=$NAME.tar.gz
  tar -C "$WORK" -czf "$ROOT/dist/$ARCHIVE" "$NAME"
fi

cd "$ROOT/dist"
if command -v sha256sum >/dev/null; then
  sha256sum "$ARCHIVE" >"$ARCHIVE.sha256"
else
  shasum -a 256 "$ARCHIVE" >"$ARCHIVE.sha256"
fi
ls -l "$ARCHIVE"
