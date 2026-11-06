#!/usr/bin/env bash
set -e

echo "=========================================================================="
echo "          TESSERA CHAMPIONSHIP ENGINE - 1-HOUR TRAINING PIPELINE          "
echo "=========================================================================="
date

mkdir -p data

# 1. Backup current champion weights
if [ -f data/nnue.bin ]; then
    echo "[1/5] Backing up current weights to data/nnue_backup.bin..."
    cp -v data/nnue.bin data/nnue_backup.bin
fi

# 2. Generate large-scale high-throughput self-play dataset (~45,000 games)
echo ""
echo "[2/5] Generating ~45,000 self-play games (approx 1.1 Million samples)..."
start_gen=$(date +%s)
./tools/generate_dataset 45000 data/train_data.bin
end_gen=$(date +%s)
echo "Dataset generation completed in $((end_gen - start_gen)) seconds."

# 3. Train NNUE with warm-start, baseline validation check, and Adam cosine decay
echo ""
echo "[3/5] Fine-tuning Deep NNUE model (60 epochs with warm-start)..."
start_train=$(date +%s)
./tools/train_nnue data/train_data.bin data/nnue.bin 60
end_train=$(date +%s)
echo "Model training completed in $((end_train - start_train)) seconds."

# 4. Clean up large training dataset binary to conserve disk space
echo ""
echo "[4/5] Cleaning up temporary dataset file..."
rm -f data/train_data.bin

# 5. Tournament validation benchmark against all 6 bot archetypes
echo ""
echo "[5/5] Running 60-game tournament benchmark against 6 engine archetypes..."
./tools/match_benchmark 10 | tee data/post_training_tournament.txt

echo ""
echo "=========================================================================="
echo "          TESSERA TRAINING & VALIDATION PIPELINE COMPLETE                 "
echo "=========================================================================="
date
