#!/bin/bash
# pypi-publish.sh — the wheels of a kinfer release for PyPI (pip install kinfer-ai),
# one for each platform, checked, ready for you to upload:
#
#   scripts/pypi-publish.sh v0.3.0
#
# It takes the release's archives from dist/, where release.sh left them, or from
# the GitHub release, checked against their sha256 either way; builds the wheels
# with scripts/pypi-wheel.py; and runs `twine check`. The upload is yours, with
# your PyPI credentials: the command is printed at the end.
set -euo pipefail

VERSION=${1:?usage: scripts/pypi-publish.sh vX.Y.Z}
NAME=${KINFER_PYPI_NAME:-kinfer-ai}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/dist" 2>/dev/null || { mkdir -p "$ROOT/dist" && cd "$ROOT/dist"; }

WHEELS=()
for platform in darwin-arm64.tar.gz linux-amd64.tar.gz linux-arm64.tar.gz windows-amd64.zip windows-arm64.zip; do
  TAR=kinfer-$VERSION-$platform
  if [ ! -f "$TAR" ]; then
    echo "pypi-publish: $TAR is not in dist/, fetching it from the GitHub release"
    curl -fL --retry 3 -o "$TAR" "https://github.com/LocalKinAI/kinfer/releases/download/$VERSION/$TAR"
    curl -fsSL --retry 3 -o "$TAR.sha256" "https://github.com/LocalKinAI/kinfer/releases/download/$VERSION/$TAR.sha256"
  fi
  shasum -a 256 -c "$TAR.sha256"
  WHEELS+=("$(python3 "$ROOT/scripts/pypi-wheel.py" "$TAR" --name "$NAME" --out .)")
done
ls -l "${WHEELS[@]}"
python3 -m twine check "${WHEELS[@]}"

echo
echo "Ready: ${WHEELS[*]}"
echo "Upload them with your PyPI credentials:"
echo "  cd dist && python3 -m twine upload ${WHEELS[*]}"
