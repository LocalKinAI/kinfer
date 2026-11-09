#!/bin/sh
# install.sh — install kinfer on an Apple Silicon Mac or on Linux: the latest
# release (or KINFER_VERSION), checked against its sha256, into ~/.local/bin (or
# KINFER_BIN). On Windows, scripts/install.ps1 does the same.
#
#   curl -fsSL https://raw.githubusercontent.com/LocalKinAI/kinfer/main/scripts/install.sh | sh
#
# The binary is not signed. Downloaded with curl it carries no quarantine, so
# macOS runs it; a firewall that is on still asks once before serve or worker
# takes connections from another machine.
set -eu

REPO=LocalKinAI/kinfer
fail() {
  echo "$*" >&2
  exit 1
}

GLIBC=
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) PLATFORM=darwin-arm64 ;;
  Linux/x86_64) PLATFORM=linux-amd64 GLIBC=2.34 ;;
  Linux/aarch64 | Linux/arm64) PLATFORM=linux-arm64 GLIBC=2.38 ;;
  *) fail "kinfer has no release for $(uname -s) on $(uname -m); to build it: https://github.com/$REPO#build" ;;
esac

if [ -n "$GLIBC" ]; then
  have=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true)
  [ -n "$have" ] || fail "kinfer's Linux build needs glibc, and this system uses another C library"
  oldest=$(printf '%s\n%s\n' "$GLIBC" "$have" | sort -V | head -n 1)
  [ "$oldest" = "$GLIBC" ] || fail "kinfer's Linux build needs glibc $GLIBC or newer; this system has $have"
fi

VERSION=${KINFER_VERSION:-$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" |
  sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1)}
if [ -z "$VERSION" ]; then
  fail "no kinfer release found at https://github.com/$REPO/releases"
fi

NAME=kinfer-$VERSION-$PLATFORM
URL=${KINFER_BASE_URL:-https://github.com/$REPO/releases/download/$VERSION}/$NAME.tar.gz
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "downloading kinfer $VERSION..."
curl -fsSL "$URL" -o "$TMP/$NAME.tar.gz"
curl -fsSL "$URL.sha256" -o "$TMP/$NAME.tar.gz.sha256"
if command -v sha256sum >/dev/null 2>&1; then
  CHECK="sha256sum -c"
else
  CHECK="shasum -a 256 -c"
fi
if ! (cd "$TMP" && $CHECK "$NAME.tar.gz.sha256" >/dev/null); then
  fail "the download does not match its sha256"
fi
tar -xzf "$TMP/$NAME.tar.gz" -C "$TMP"

BIN=${KINFER_BIN:-$HOME/.local/bin}
mkdir -p "$BIN"
# A new file renamed over the old one: macOS kills a binary rewritten in place
# once it has run.
cp "$TMP/$NAME/kinfer" "$BIN/kinfer.new"
chmod 755 "$BIN/kinfer.new"
mv -f "$BIN/kinfer.new" "$BIN/kinfer"

echo "installed $("$BIN/kinfer" version) in $BIN"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) echo "add it to your PATH:  export PATH=\"$BIN:\$PATH\"" ;;
esac
[ -z "$GLIBC" ] || echo "kinfer runs on the CPU on Linux for now."
echo "then:  kinfer fit    # what this machine can run"
