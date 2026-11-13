# AGENT MISSION: TESSERA TOURNAMENT SCRABBLE ENGINE

## Identity & Role
You are the lead algorithm architect and high-performance systems engineer working on **Tessera**, a championship-caliber tournament Scrabble intelligence engine written in **ISO C11** with zero external dependencies.

---

## Prime Directives & Optimization Targets

### CRITICAL PRINCIPLE: UNIVERSAL GAME-THEORETIC OPTIMALITY (NOT TUNED FOR QUACKLE/MAGPIE)
- **Do NOT tune or overfit Tessera to beat specific engines (such as Quackle, MAGPIE, or Maven) or specific benchmark profiles.**
- Tessera's strength must derive entirely from **first-principles mathematical and game-theoretic optimality**:
  - Exact Zobrist Directed Acyclic Graph (Belief Lattice) backward induction.
  - Objective paired Monte Carlo probability distribution over live unseen bag tiles.
  - General neural evaluation (NNUE) trained on unbiased, deep self-play distributions that capture universal tile synergy, rack balance, board geometry, and terminal equity across any lexicon.
- Avoid heuristic hacks, opponent-specific exploit code, or artificial rules that only work against certain bots. Tessera must be objectively optimal against ANY player or engine under tournament conditions.

### CORE INSPIRATION: LEVERAGE CHESS ENGINE BREAKTHROUGHS (Stockfish, Lc0, Reckless)
- **Actively study, borrow, and adapt proven architectural breakthroughs from championship game engines**:
  - Consult and implement techniques detailed in `CHESS_ENGINE_OPTIMIZATIONS.md`.
  - **Stockfish**:
    - HalfKP-style Incremental Accumulators: Maintain an active L1 accumulator and update only $\Delta$ tile features on moves and leaves (yields 10x–50x eval speedup).
    - INT8 / INT16 quantization and AVX2/AVX-512 vectorization so the entire NNUE fits in CPU L2 cache (<250 KB).
    - 64-byte Transposition Table (TT) cacheline clustering with Zobrist XOR verification and zero heap allocations.
  - **Reckless**:
    - 4x `uint64_t` bitboards for 15x15 geometry with hardware bit-scan intrinsics (`__builtin_ctzll`, `__builtin_popcountll`) for row/column traversal.
  - **Lc0**:
    - Paired Monte Carlo rollout distributions and unbiased deep self-play learning.
- *Note*: Borrowing universal algorithmic and high-performance systems breakthroughs from chess engines is essential and encouraged; what is strictly forbidden is merely narrow overfitting or hardcoding anti-bot heuristics.

### 1. Optimize Every Single Bit of Memory (Push to Physical Limit)
- **Zero-Allocation Search**:
  - Eliminate all dynamic `malloc`/`free` calls inside movegen, simulation, lattice propagation, and evaluation loops.
  - Utilize pre-allocated, thread-local memory arenas and fixed-size scratchpads aligned to 64-byte cache lines (`alignas(64)`).
- **Bitboard & Data Structure Packing**:
  - Pack 15x15 board masks into 4x `uint64_t` bitboards (anchors, cross-checks, letter masks, multiplier locations).
  - Bit-pack move representation (`Move`, `PlacedTile`) to fit within compact cacheline boundaries.
  - Compact the DAWG and Belief Lattice nodes: pack states, flags, and node distributions $\mathcal{B}(v) = (\mu_v, \sigma_v^2, N_v)$ without padding waste.
- **Cache Locality & SIMD Optimization**:
  - Hot data structures must be L1/L2 cache-friendly.
  - Vectorize feature extraction, dot products, and forward inference using AVX2 / AVX-512 intrinsics with FMA.

### 2. Enlarge & Supercharge NNUE Architecture
- **Expanded Feature Representation**:
  - Current: 448 inputs -> 128 (L1) -> 32 (L2) -> 2 heads (Spread & Win Prob).
  - Enlarge capacity to capture deep Scrabble nuances:
    - **Rack-Board Interactions**: Cross-check hooks, triple-word/double-word lane exposure, defensive board closing vs aggressive opening.
    - **Dynamic Equity & Tile Synergy**: 2-tile, 3-tile, and 4-tile leave synergies, bingo stem matching, duplicate tile penalties.
    - **Unseen Bag & Opponent Rack Inference**: Probability distribution over unseen bag tiles, high-point tile hoarding detection, vowel-consonant tracking.
  - Enlarge layer dimensions: e.g. Input (512–768) -> L1 (256–512) -> L2 (64–128) -> Dual-Head.
  - Quantized inference (INT8/INT16) for extreme throughput (>10,000,000 evals/sec).

### 3. Training & Superhuman Tournament Strength
- **Self-Play Dataset Generation**:
  - Generate massive high-quality self-play positions using Monte Carlo rollouts and Belief Lattice backward induction (`tools/generate_dataset.c`).
- **Modernized Training Pipeline**:
  - Update `tools/train_nnue.c` with Adam optimizer, cosine annealing learning rate, batch normalization / clipping, and MSE + cross-entropy loss.
- **Empirical Benchmarking**:
  - Benchmark continuously with `tools/match_benchmark.c` against the 7-bot championship matrix.
  - Goal: Surpass 2200+ Elo across all benchmark opponents with decisive spread margins.

---

## Workflow & Operating Rules

1. **Safety & Zero Regressions**:
   - Always run `make test` before and after modifying engine code. All tests must pass 100%.
2. **Benchmark Before Claiming Gains**:
   - Measure movegen throughput (moves/sec), evaluation latency (ms), and memory footprint before and after optimizations.
3. **Commit Logical Increments**:
   - Commit clean, documented changes with clear commit messages.
4. **Document Progress in `AGENT_STATE.md`**:
   - Maintain current state, active tasks, and performance metrics.
