#!/usr/bin/env bash
# ==============================================================================
# Tessera Multi-Platform Cross-Compilation Pipeline via Zig Toolchain
# ==============================================================================
set -euo pipefail

VERSION="1.0.0"
DIST_DIR="dist"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

echo "======================================================================"
echo "   TESSERA — MULTI-TARGET CROSS-COMPILER (ZIG TOOLCHAIN v$(zig version))"
echo "======================================================================"
echo ""

# Ensure assets exist
if [ ! -f "data/dawg.bin" ]; then
    echo ">> Precompiled DAWG binary not found, generating from lexicon..."
    make -j4
    ./tessera build-dict
fi

# Clean previous distribution artifacts
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}/bin" "${DIST_DIR}/packages"

TARGETS=(
    "x86_64-linux-gnu:linux-x86_64:tessera:tar.gz"
    "aarch64-linux-musl:linux-arm64:tessera:tar.gz"
    "x86_64-windows-gnu:windows-x86_64:tessera.exe:zip"
    "aarch64-macos:macos-arm64:tessera:tar.gz"
    "x86_64-macos:macos-x86_64:tessera:tar.gz"
)

echo ">> Compiling all platform targets with -Doptimize=ReleaseFast..."

for entry in "${TARGETS[@]}"; do
    IFS=":" read -r TRIPLE PLATFORM_NAME BIN_NAME PKG_FORMAT <<< "${entry}"
    echo "----------------------------------------------------------------------"
    echo ">> [TARGET] ${PLATFORM_NAME} (${TRIPLE})"
    
    STAGE_DIR="${DIST_DIR}/staging/${PLATFORM_NAME}"
    mkdir -p "${STAGE_DIR}/data"
    
    # Cross-compile target binary via Zig build
    zig build -Dtarget="${TRIPLE}" -Doptimize=ReleaseFast --prefix "${STAGE_DIR}"
    
    # Stage precompiled binary and shared data assets
    cp "${STAGE_DIR}/bin/${BIN_NAME}" "${DIST_DIR}/bin/tessera-${PLATFORM_NAME}$([ "${BIN_NAME}" = "tessera.exe" ] && echo ".exe" || echo "")"
    cp data/dawg.bin "${STAGE_DIR}/data/"
    cp data/nnue.bin "${STAGE_DIR}/data/"
    cp LICENSE "${STAGE_DIR}/"
    cp README.md "${STAGE_DIR}/"
    
    # Package bundle
    ARCHIVE_NAME="tessera-v${VERSION}-${PLATFORM_NAME}"
    if [ "${PKG_FORMAT}" = "zip" ]; then
        if command -v zip >/dev/null 2>&1; then
            (cd "${STAGE_DIR}" && zip -qr "${ROOT_DIR}/${DIST_DIR}/packages/${ARCHIVE_NAME}.zip" bin/"${BIN_NAME}" data/ LICENSE README.md)
        else
            (cd "${STAGE_DIR}" && python3 -c "import zipfile, os; z = zipfile.ZipFile('${ROOT_DIR}/${DIST_DIR}/packages/${ARCHIVE_NAME}.zip', 'w', zipfile.ZIP_DEFLATED); [z.write(os.path.join(r, f), os.path.relpath(os.path.join(r, f), '.')) for r, d, files in os.walk('.') for f in files]; z.close()")
        fi
        echo "   -> Created ${DIST_DIR}/packages/${ARCHIVE_NAME}.zip"
    else
        tar -czf "${DIST_DIR}/packages/${ARCHIVE_NAME}.tar.gz" -C "${STAGE_DIR}" "bin/${BIN_NAME}" "data/" "LICENSE" "README.md"
        echo "   -> Created ${DIST_DIR}/packages/${ARCHIVE_NAME}.tar.gz"
    fi
done

rm -rf "${DIST_DIR}/staging"

echo ""
echo "======================================================================"
echo ">> All target binaries successfully compiled and packaged!"
echo "======================================================================"
echo ""
echo "Platform Binaries:"
ls -lh "${DIST_DIR}/bin"
echo ""
echo "Distribution Archives:"
ls -lh "${DIST_DIR}/packages"
echo ""
echo "SHA-256 Checksums:"
(cd "${DIST_DIR}/packages" && sha256sum * | tee SHA256SUMS.txt)
echo ""
