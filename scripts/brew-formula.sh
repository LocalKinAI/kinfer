#!/bin/bash
# brew-formula.sh — the Homebrew formula for a release scripts/release.sh packed,
# for the tap at github.com/LocalKinAI/homebrew-tap:
#
#   scripts/brew-formula.sh v0.1.0 > ../homebrew-tap/Formula/kinfer.rb
#
# It installs the release's binary as it is, which is why it needs neither
# llama.cpp nor Go: the libraries are inside.
set -euo pipefail

VERSION=${1:?usage: scripts/brew-formula.sh VERSION}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=kinfer-$VERSION-darwin-arm64
SUM=$(cut -d' ' -f1 "$ROOT/dist/$NAME.tar.gz.sha256")

cat <<RUBY
class Kinfer < Formula
  desc "Single-file local inference runtime for agent fleets"
  homepage "https://github.com/LocalKinAI/kinfer"
  url "https://github.com/LocalKinAI/kinfer/releases/download/$VERSION/$NAME.tar.gz"
  version "${VERSION#v}"
  sha256 "$SUM"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on :macos

  def install
    bin.install "kinfer"
  end

  test do
    assert_match "kinfer v#{version}", shell_output("#{bin}/kinfer version")
  end
end
RUBY
