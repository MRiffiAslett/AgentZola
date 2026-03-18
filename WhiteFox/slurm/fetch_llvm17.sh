#!/bin/bash
# One-time setup: download LLVM 17 tools for profraw v8 coverage merging.
# Run from the login node (needs internet access).
set -euo pipefail

LLVM17_DIR="/vol/bitbucket/mtr25/tfbuild/llvm17"
LLVM17_BIN="$LLVM17_DIR/bin"

if [ -x "$LLVM17_BIN/llvm-profdata" ]; then
  echo "LLVM 17 already installed at $LLVM17_BIN"
  "$LLVM17_BIN/llvm-profdata" --version 2>&1 | head -1
  exit 0
fi

LLVM_VER="17.0.6"
TARBALL="clang+llvm-${LLVM_VER}-x86_64-linux-gnu-ubuntu-22.04.tar.xz"
URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VER}/${TARBALL}"

echo "Downloading LLVM ${LLVM_VER} (only extracting llvm-profdata + llvm-cov) …"
mkdir -p "$LLVM17_BIN"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl -fSL --progress-bar "$URL" -o "$TMP/$TARBALL"

STRIP="clang+llvm-${LLVM_VER}-x86_64-linux-gnu-ubuntu-22.04"
tar -xf "$TMP/$TARBALL" -C "$LLVM17_DIR" --strip-components=1 \
  "${STRIP}/bin/llvm-profdata" \
  "${STRIP}/bin/llvm-cov"

echo ""
echo "Installed to $LLVM17_BIN:"
ls -l "$LLVM17_BIN"/llvm-{profdata,cov}
"$LLVM17_BIN/llvm-profdata" --version 2>&1 | head -1
echo "Done."
