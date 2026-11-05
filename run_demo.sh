#!/bin/bash
set -e

echo "======================================================================"
echo "          TESSERA — GRANDMASTER SCRABBLE ENGINE (NATIVE C)            "
echo "======================================================================"
echo ""

# 1. Build the native engine
echo "[1/3] Compiling native C engine with -O3 -flto -march=native..."
make -j4

# 2. Check binary DAWG dictionary
if [ ! -f "data/dawg.bin" ]; then
    echo "[2/3] Building precompiled DAWG from data/dictionary.txt..."
    ./tessera build-dict
else
    echo "[2/3] Verified precompiled DAWG (data/dawg.bin ready)."
fi

# 3. Launch Web GUI
echo "[3/3] Launching demonstration server on http://localhost:8080..."
echo "Press Ctrl+C to terminate the server."
echo ""

python3 gui/server.py
