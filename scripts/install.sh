#!/bin/sh
# install.sh — install kinfer on an Apple Silicon Mac: the latest release (or
# KINFER_VERSION), checked against its sha256, into ~/.local/bin (or KINFER_BIN).
#
#   curl -fsSL https://raw.githubusercontent.com/LocalKinAI/kinfer/main/scripts/install.sh | sh
#
# The binary is not signed. Downloaded with curl it carries no quarantine, so
# macOS runs it; a firewall that is on still asks once before serve or worker
# takes connections from another machine.
set -eu

REPO=LocalKinAI/kinfer
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "kinfer releases are built for Apple Silicon Macs; elsewhere, build it: https://github.com/$REPO#build" >&2
  exit 1
fi

VERSION=${KINFER_VERSION:-$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" |
  sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1)}
if [ -z "$VERSION" ]; then
  echo "no kinfer release found at https://github.com/$REPO/releases" >&2
  exit 1
fi

NAME=kinfer-$VERSION-darwin-arm64
URL=https://github.com/$REPO/releases/download/$VERSION/$NAME.tar.gz
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "downloading kinfer $VERSION..."
curl -fsSL "$URL" -o "$TMP/$NAME.tar.gz"
curl -fsSL "$URL.sha256" -o "$TMP/$NAME.tar.gz.sha256"
if ! (cd "$TMP" && shasum -a 256 -c "$NAME.tar.gz.sha256" >/dev/null); then
  echo "the download does not match its sha256" >&2
  exit 1
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
echo "then:  kinfer fit    # what this Mac can run"
