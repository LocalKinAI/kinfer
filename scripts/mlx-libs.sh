#!/bin/bash
# Packs the MLX libraries kinfer downloads the first time it serves an MLX
# model: mlx-c built against MLX, MLX itself and its kernels, and both
# licences. Run on a Mac that has them:
#
#   scripts/mlx-libs.sh <mlx-c libmlxc.dylib> <site-packages/mlx dir> <mlx-c source dir> <version> <out dir>
#
# libmlxc.dylib must already have @loader_path as its rpath and a deployment
# target no newer than libmlx's (otool -l … | grep minos). Prints the sha256
# and size internal/mlxengine/libs.go pins.
set -euo pipefail
mlxc=$1 mlx=$2 mlxcsrc=$3 version=$4 out=$5

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp "$mlxc" "$stage/libmlxc.dylib"
cp "$mlx/lib/libmlx.dylib" "$mlx/lib/libjaccl.dylib" "$mlx/lib/mlx.metallib" "$stage/"
cp "$mlx/../mlx_metal-$version.dist-info/licenses/LICENSE" "$stage/LICENSE-mlx"
cp "$mlxcsrc/LICENSE" "$stage/LICENSE-mlx-c"

name="kinfer-mlx-$version-darwin-arm64.tar.gz"
mkdir -p "$out"
tar -C "$stage" -czf "$out/$name" libmlxc.dylib libmlx.dylib libjaccl.dylib mlx.metallib LICENSE-mlx LICENSE-mlx-c
echo "$out/$name"
echo "sha256 $(shasum -a 256 "$out/$name" | cut -d' ' -f1)"
echo "size $(stat -f %z "$out/$name")"
