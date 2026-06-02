#!/usr/bin/env bash
# Build ncnn's quantize CLI tools (ncnn2table, ncnn2int8, ncnnoptimize) from the vendored
# source. These are NOT in the pip `ncnn` wheel and the main static-lib build sets
# NCNN_BUILD_TOOLS=OFF, so the INT8 export pipeline needs them built once.
#
# Idempotent: if all three binaries already exist in tools-bin/, it does nothing.
# Built with NCNN_SIMPLEOCV=ON so ncnn2table compiles without OpenCV (we only use the
# .npy calibration path, type=1).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
NCNN="$ROOT/thirdparty/ncnn"
BUILD="$NCNN/build-tools"
BIN="$NCNN/tools-bin"

mkdir -p "$BIN"
if [ -x "$BIN/ncnn2table" ] && [ -x "$BIN/ncnn2int8" ] && [ -x "$BIN/ncnnoptimize" ]; then
	echo "ncnn quantize tools already built in $BIN"
	exit 0
fi

echo "Configuring ncnn quantize tools build in $BUILD ..."
cmake -S "$NCNN" -B "$BUILD" \
	-DCMAKE_BUILD_TYPE=Release \
	-DNCNN_BUILD_TOOLS=ON \
	-DNCNN_BUILD_EXAMPLES=OFF \
	-DNCNN_BUILD_BENCHMARK=OFF \
	-DNCNN_BUILD_TESTS=OFF \
	-DNCNN_SIMPLEOCV=ON \
	-DNCNN_INT8=ON \
	-DBUILD_SHARED_LIBS=OFF

echo "Building ncnn2table ncnn2int8 ncnnoptimize ..."
cmake --build "$BUILD" --config Release --target ncnn2table ncnn2int8 ncnnoptimize --parallel

# Collect the three binaries into a flat dir so export_int8.py doesn't depend on cmake's
# internal layout (ncnnoptimize lands in tools/, ncnn2{table,int8} in tools/quantize/).
found_all=1
for tool in ncnn2table ncnn2int8 ncnnoptimize; do
	src="$(find "$BUILD" -type f -name "$tool" -perm -u+x 2>/dev/null | head -n1 || true)"
	if [ -z "$src" ]; then
		echo "ERROR: built binary not found: $tool" >&2
		found_all=0
		continue
	fi
	cp -f "$src" "$BIN/$tool"
done

if [ "$found_all" -ne 1 ]; then
	echo "ERROR: one or more quantize tools failed to build" >&2
	exit 1
fi

echo "OK: quantize tools in $BIN"
ls -la "$BIN"
