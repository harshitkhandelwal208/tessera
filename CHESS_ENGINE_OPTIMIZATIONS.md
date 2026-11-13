# CHESS ENGINE ARCHITECTURAL LESSONS FOR TESSERA
## High-Performance Blueprint from Stockfish, Lc0, and Reckless

This document details proven, state-of-the-art systems and algorithmic techniques from championship game engines (**Stockfish**, **Lc0**, and **Reckless**) specifically adapted to push **Tessera** to the physical and mathematical limit of tournament Scrabble strength.

---

## 1. Stockfish Lessons: NNUE Quantization & Incremental Accumulators

### A. Incremental Accumulator Architecture (The Single Biggest Speedup)
- **The Problem**: In naive NNUE, evaluating a move requires extracting all 768+ features and performing a full matrix-vector multiplication $W_1 \cdot x + b_1$ ($768 \times 256$ operations).
- **Stockfish Solution**: In Stockfish (HalfKP / HalfKAv2), the input features are sparse. Stockfish maintains an **accumulator** representing the active hidden layer:
  $$\text{Accumulator} = b_1 + \sum_{i \in \text{active}} W_1[i]$$
- **Scrabble Application**:
  - A Scrabble move places $k$ tiles ($1 \le k \le 7$) on the board and leaves $7 - k$ tiles on the rack.
  - Instead of recomputing the full input projection, update the accumulator incrementally:
    - Add the newly placed tile features on the board cells.
    - Subtract the tiles removed from the rack.
    - Add the new leave synergy features.
  - **Result**: Evaluator speed increases by **10x to 50x**, jumping from ~300k evals/s to **10,000,000+ evals/s**!

### B. INT8 / INT16 Quantization & AVX2 / AVX-512 VNNI Vectorization
- **Quantize Weights**:
  - Store $W_1$ as `int8_t` or `int16_t` with scale factor $S = 128$ or $256$.
  - Use ClippedReLU: $f(x) = \text{clamp}(x, 0, 127)$.
  - Use `_mm256_maddubs_epi16` and `_mm256_madd_epi16` (or `_mm512_dpbusd_epi32` on AVX-512) for vector dot products.
  - Output layers $W_2, W_3$ computed using fast integer SIMD or FP32 accumulation.
- **Memory Footprint**:
  - A 768 -> 256 -> 64 quantized NNUE model takes less than **250 KB**, fitting completely inside the CPU L2 cache!

### C. Transposition Table (TT) Cacheline Clustering
- **Cluster Buckets**: Pack 3 or 4 entries into exactly **64 bytes** (one CPU cacheline).
- **Zobrist XOR Trick**: Store `entry.key_xor_data = zobrist_64 ^ entry_data`. This verifies key identity without storing a separate 64-bit key, saving 8 bytes per entry.
- **Replacement Policy**:
  - Entry 1: Deepest search.
  - Entry 2: Most recent generation (age-based).
  - Entry 3: Unconditional replacement.

---

## 2. Lc0 (Leela Chess Zero) Lessons: Dual-Head Equity & Monte Carlo Synergy

### A. Dual-Head Value & Policy Representation
- **Head 1 (Net Spread / Equity)**:
  - Predicts expected point differential $\Delta \text{Pts} \in [-\infty, +\infty]$.
- **Head 2 (Win Probability)**:
  - Predicts $P(\text{Win}) \in [0.0, 1.0]$ via sigmoid $\sigma(x)$.
  - Vital in Scrabble endgame and close games: a move that scores +20 pts but gives the opponent an 80% chance of a 50-pt bingo is vastly inferior to a safe +15 pt move that seals a 98% win probability.

### B. Paired Rollout Ground Truths & Self-Play Generation
- Generate hundreds of thousands of self-play games where moves are chosen via Belief Lattice backward induction.
- Label every state with both the final game outcome (win/loss) and the net score spread.
- Train the enlarged NNUE using combined Loss:
  $$\mathcal{L} = \mathcal{L}_{\text{MSE}}(\text{Spread}) + \lambda \cdot \mathcal{L}_{\text{BCE}}(\text{WinProb})$$

---

## 3. Reckless Lessons: Bitboard Geometry & Zero-Allocation Search

### A. Compact 15x15 Bitboard Representation
- Represent the 15x15 Scrabble board using an array of 4x `uint64_t` bitboards (225 bits total):
  - `bb_occupied`: 1 where a tile is present.
  - `bb_anchors`: 1 where an empty square is adjacent to an existing tile.
  - `bb_tw`, `bb_dw`, `bb_tl`, `bb_dl`: Premium multiplier masks.
- Word placement legality and cross-checks become fast bitwise operations:
  - Finding empty anchors: `anchors = ((occupied << 1) | (occupied >> 1) | (occupied << 15) | (occupied >> 15)) & ~occupied`.
  - Bit scans using `__builtin_ctzll` and `__builtin_popcountll`.

### B. Absolute Zero Allocations & 64-Byte Cacheline Alignment
- Hot loops must **never** call `malloc()` or `free()`.
- Use thread-local scratchpad arenas with `alignas(64)` to prevent cacheline false sharing across OpenMP threads.
