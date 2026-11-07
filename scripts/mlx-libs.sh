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
#
# The MLX may be one not yet released, from MLX's own CI ("Release build",
# artifact backend-metal-macOS-ARM64, the macosx_26_0 wheel), with a version
# naming its commit, 0.32.5-9f806b3. mlx-c then needs patches/mlx-c/: MLX took
# a sliding window in scaled_dot_product_attention before mlx-c followed.
set -euo pipefail
mlxc=$1 mlx=$2 mlxcsrc=$3 version=$4 out=$5

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp "$mlxc" "$stage/libmlxc.dylib"
cp "$mlx/lib/libmlx.dylib" "$mlx/lib/libjaccl.dylib" "$mlx/lib/mlx.metallib" "$stage/"
cp "$(ls -d "$mlx"/../mlx_metal-*.dist-info | head -1)/licenses/LICENSE" "$stage/LICENSE-mlx"
cp "$mlxcsrc/LICENSE" "$stage/LICENSE-mlx-c"

name="kinfer-mlx-$version-darwin-arm64.tar.gz"
mkdir -p "$out"
# Nothing but the six files: macOS's tar would add an AppleDouble ._ entry for
# each file with extended attributes (a codesigned dylib, a copy off another
# volume), and kinfer refuses a package with entries it does not expect.
xattr -c "$stage"/*
COPYFILE_DISABLE=1 tar -C "$stage" -czf "$out/$name" libmlxc.dylib libmlx.dylib libjaccl.dylib mlx.metallib LICENSE-mlx LICENSE-mlx-c
echo "$out/$name"
echo "sha256 $(shasum -a 256 "$out/$name" | cut -d' ' -f1)"
echo "size $(stat -f %z "$out/$name")"
