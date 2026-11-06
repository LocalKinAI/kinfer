#!/bin/bash
# pypi-publish.sh — the wheel of a kinfer release for PyPI (pip install kinfer-ai),
# checked, ready for you to upload:
#
#   scripts/pypi-publish.sh v0.2.4
#
# It takes the release's tarball from dist/, where release.sh left it, or from the
# GitHub release, checked against its sha256 either way; builds the wheel with
# scripts/pypi-wheel.py; and runs `twine check`. The upload is yours, with your
# PyPI credentials: the command is printed at the end.
set -euo pipefail

VERSION=${1:?usage: scripts/pypi-publish.sh vX.Y.Z}
NAME=${KINFER_PYPI_NAME:-kinfer-ai}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TAR=kinfer-$VERSION-darwin-arm64.tar.gz
cd "$ROOT/dist" 2>/dev/null || { mkdir -p "$ROOT/dist" && cd "$ROOT/dist"; }

if [ ! -f "$TAR" ]; then
  echo "pypi-publish: $TAR is not in dist/, fetching it from the GitHub release"
  curl -fL --retry 3 -o "$TAR" "https://github.com/LocalKinAI/kinfer/releases/download/$VERSION/$TAR"
  curl -fsSL --retry 3 -o "$TAR.sha256" "https://github.com/LocalKinAI/kinfer/releases/download/$VERSION/$TAR.sha256"
fi
shasum -a 256 -c "$TAR.sha256"

WHEEL=$(python3 "$ROOT/scripts/pypi-wheel.py" "$TAR" --name "$NAME" --out .)
ls -l "$WHEEL"
python3 -m twine check "$WHEEL"

echo
echo "Ready: dist/$(basename "$WHEEL")"
echo "Upload it with your PyPI credentials:"
echo "  python3 -m twine upload dist/$(basename "$WHEEL")"
