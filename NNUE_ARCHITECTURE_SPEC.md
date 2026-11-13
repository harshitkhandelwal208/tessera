# TESSERA NNUE ARCHITECTURAL SPECIFICATION
## Global Chess Engine Research, Mathematical Sizing, and Production Blueprint

---

## 1. Global Chess Engine NNUE Landscape & Research

Modern computer chess underwent an epochal revolution when Yu Nasu introduced NNUE (Efficiently Updatable Neural Networks) to Shogi, which was swiftly adopted and supercharged by **Stockfish** (starting with Stockfish 12) and subsequent independent engines (**Viridithas**, **Berserk**, **Seer**, **Caissa**, **Clover**, **PlentyChess**). 

The architectural dimensions across the world's strongest engines reveal clear engineering patterns:

| Engine | Network Layout | Model File Size | Key Innovations |
| :--- | :--- | :--- | :--- |
| **Stockfish 16/17 (Big Net)** | $706,304 \to (1024 \times 2) \to 32 \to 32 \to 1$ | **~133 MB** | Massive HalfKAv2 input space with 64 King-square buckets; parallel ClippedReLU + Squared Clipped ReLU (SCReLU); dual-perspective feature transformer. |
| **Stockfish 16/17 (Small Net)** | $22,528 \to 512 \to 16 \to 32 \to 1$ | **~6 MB to 7 MB** | Ultra-lean secondary network loaded in parallel; used on ~80% of search nodes to avoid L3/DRAM bus choking. |
| **Viridithas** | Inputs $\to (2048 \times 2) \to 16 \to 32 \to 1$ | **~10 MB to 12 MB** | 16 input king buckets, 8 output material buckets; pioneered SCReLU for smoother gradient propagation and non-linear penalty curves. |
| **Berserk** | Inputs $\to (512 \text{ to } 1024 \times 2) \to 16 \to 32 \to 1$ | **~4 MB to 8 MB** | Quantized `int16_t` accumulators; Bullet trainer; heavy AVX2/AVX-512 auto-vectorization; high nodes/second. |
| **Seer** | Inputs $\to (1024 \times 2) \to 32 \to 32 \to 2$ | **~6 MB to 10 MB** | Dedicated Win/Draw/Loss (WDL) probability dual heads; retrograde self-play training initialized from Syzygy 6-man tablebases. |
| **Caissa / Clover** | Inputs $\to (1024 \times 2) \to 16 \to 32 \to 1$ | **~8 MB** | Consensus modern layout; 8 to 16 king-position buckets; cacheline-aligned data structures. |

### Key Lessons from the State-of-the-Art:
1. **The Feature Transformer Dominates Size**: In all NNUE engines, **over 95% of the network weights reside in $W_1$ (Input $\to$ L1)**. The hidden layers ($L_2, L_3$) are tiny (only 16 to 64 neurons, consuming < 64 KB).
2. **Extreme Input Sparsity**: Despite having tens of thousands of features, **only 30 to 80 features are active at any time**. This allows $O(k)$ incremental accumulator updates instead of $O(N)$ full forward passes.
3. **Squared Clipped ReLU (SCReLU)**: Replacing or supplementing standard ClippedReLU with $( \text{clamp}(x, 0, 127) )^2$ drastically improves positional evaluation quality without adding parameter count.
4. **The Memory Wall**: When a network exceeds CPU L3 cache (16–32 MB) and spills into DRAM, memory latency (60–80 ns) becomes a crippling bottleneck during multi-threaded rollouts.

---

## 2. Mathematical Calculation: Where Should Tessera's NNUE Be?

### A. The Structural Differences Between Chess and Scrabble
1. **No Moving King Multiplier**: In chess, every piece's value is modulated by 64 King squares ($64 \times 640 = 40,960$ base HalfKP, or 706k HalfKAv2). In Scrabble, the board is static ($15 \times 15 = 225$ cells).
2. **Combinatorial Rack Leaves**: In Scrabble, the primary tactical driver is **rack leave synergy** (which tiles remain on your rack after a move).
3. **Search Paradigm**:
   - Chess searches a deterministic tree with branching factor $b \approx 35$.
   - Scrabble has a branching factor of **$800$ to $3,000$ legal moves per turn**, evaluated across **hundreds of Monte Carlo bag rollouts** (or Belief Lattice DAG expansions).
   - Therefore, Scrabble requires **substantially higher evaluation throughput per position** than chess.

### B. Exact Calculation of Scrabble's Natural Feature Universe
Let us calculate the complete mathematical feature space of Scrabble:
1. **Board Cell Occupancy**: $225 \text{ cells} \times 27 \text{ tile types (A–Z + Blank)} = \mathbf{6,075}$ features.
2. **Anchor Hooks & Premium Lane Exposure**:
   - 61 premium squares (8 TWS, 17 DWS, 12 TLS, 24 DLS).
   - Open cross-check letters and ray hook extensions: $\approx \mathbf{3,294}$ features.
3. **Rack Combinatorial Leaves**:
   - 2-tile leaves: $\binom{27 + 1}{2} = \mathbf{378}$ unordered pairs.
   - 3-tile leaves: $\binom{27 + 2}{3} = \mathbf{3,654}$ unordered triplets.
   - Vowel-consonant imbalance penalties & duplicate letter penalties: $\mathbf{32}$ features.
   - Bingo stem affinities (`TISANE`, `RETINA`, `SATIRE`): $\mathbf{100}$ features.
4. **Game State & Unseen Bag Dynamics**:
   - Bag tile count distribution (27 letter probabilities): $\mathbf{27}$ features.
   - Score differential buckets & game phase: $\mathbf{16}$ features.

$$\text{Total Natural Feature Universe} = 6,075 + 3,294 + 378 + 3,654 + 32 + 100 + 27 + 16 = \mathbf{13,576 \text{ features}}$$
Rounded up to the nearest power-of-two / SIMD-aligned boundary: $\mathbf{N_{\text{features}} = 16,384} \; (2^{14})$.

### C. Hardware Cache Line & Latency Budget
* **L1d Cache (32–48 KB / core, ~1 ns)**: Must store the active accumulator state per search thread.
  - An accumulator of dimension $L_1 = 512$ (`int16_t`) occupies exactly $512 \times 2 = \mathbf{1,024 \text{ bytes}}$ (16 cachelines). Easily fits in L1d!
* **L2 Cache (512 KB – 1 MB / core, ~3.5 ns)**:
  - If a core network is $\le \mathbf{1.0 \text{ MB}}$, all thread-local rollouts can run with zero L3/DRAM bus contention.
* **L3 Cache (16 MB – 32 MB shared, ~11 ns)**:
  - An **8.4 MB to 16.8 MB** master network fits 100% inside CPU L3 cache on all modern processors (AMD Zen 4/5, Intel 13th/14th Gen, Apple Silicon), guaranteeing **zero DRAM bus stalls**.

### D. The Ideal Sizing Verdict: Two-Tier Architecture (Stockfish Small/Big Paradigm)

| Tier | Name | Input Features | L1 Output | Quantized File Size | Target Cache | Evaluation Throughput | Primary Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Fast In-Cache Net** | **1,024** | **256** | **~256 KB – 512 KB** | **L2 Cache** | **>10,000,000 evals/s** | Hot Monte Carlo rollouts, movegen pruning, deep lattice propagation |
| **Tier 2** | **Deep Master Net** | **16,384** | **512** | **~8.4 MB (INT8) / 16.8 MB (INT16)** | **L3 Cache** | **~2,000,000 evals/s** | Top-20 root move ranking, endgame transition equity, tournament tiebreakers |

---

## 3. Complete Architectural Specification for Tessera NNUE

### A. Network Topology
```
[ Sparse Input Features: 1024 (Tier 1) / 16384 (Tier 2) ]
                      │  (O(k) Incremental Accumulator via w1_by_feature)
                      ▼
            [ L1 Accumulator: 256 / 512 ]
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[ Path A: ClippedReLU ]   [ Path B: SqrClippedReLU ]
  clamp(x, 0, 127)          clamp(x, 0, 127)^2 / 128
        │                           │
        └─────────────┬─────────────┘
                      ▼  (Concatenated: 512 / 1024 dims)
           [ Dense Layer L2: 64 ]
                      │  (ClippedReLU)
           [ Dense Layer L3: 32 ]
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[ Head 1: Spread Margin ]   [ Head 2: Win Probability ]
  Scalar Equity (Points)      Sigmoid P(Win) in [0.0, 1.0]
```

### B. Precision & Data Layout
1. **$W_1$ Weights (Feature-Major)**:
   ```c
   /* Aligned to 64-byte CPU cachelines */
   alignas(64) int8_t  w1_by_feature[NNUE_INPUT_DIM][NNUE_L1_DIM];  /* Tier 1: 1024x256, Tier 2: 16384x512 */
   alignas(64) int16_t b1[NNUE_L1_DIM];
   ```
2. **Dense Layers ($W_2, W_3$)**:
   - $W_2$: `int8_t` weights ($1024 \times 64$). Evaluated via AVX2 `_mm256_maddubs_epi16` and `_mm256_madd_epi16`.
   - $W_3$: `int8_t` weights ($64 \times 32$).
3. **Dual Output Heads**:
   - Spread Head: $32 \times 1$ linear dot product $\hat{M} = \sum w_i a_i + b$, scaled to points ($\Delta S$).
   - Win Prob Head: $32 \times 1$ logit converted via lookup table or rational sigmoid approximation:
     $$\hat{W} = \frac{1}{1 + e^{-z}}$$

### C. Incremental Accumulator Update Math
For any legal candidate move $m$ playing tiles $T_{\text{placed}}$ from rack $R$ leaving leave $L = R \setminus T_{\text{placed}}$:
$$\text{Acc}_{\text{move}} = \text{Acc}_{\text{base}} + \sum_{f \in F_{\text{added}}} W_1[f] - \sum_{f \in F_{\text{removed}}} W_1[f]$$
Where:
- $F_{\text{added}}$: newly occupied cell-tile features on board, new premium hooks created, and new leave features ($L$).
- $F_{\text{removed}}$: former rack features ($R$) and covered board hooks.
- Total active delta $|F_{\text{added}}| + |F_{\text{removed}}| \le 16$.
- Update time on AVX2: **under 25 nanoseconds**!

---

## 4. Implementation Directives for the Agent

1. **Step 1: Production Tier 1 In-Cache Net (1024 $\to$ 256 $\to$ 64 $\to$ 32 $\to$ Dual Head)**:
   - Finalize `include/nnue.h` with `NNUE_INPUT_DIM 1024`, `NNUE_L1_DIM 256`, `NNUE_L2_DIM 64`.
   - Add dual-activation **SCReLU** (Squared Clipped ReLU) alongside standard ClippedReLU in `src/nnue.c`.
   - Ensure `w1_by_feature` is aligned to 64 bytes (`alignas(64)`) for zero-cacheline-split AVX2 vector loads.
2. **Step 2: Dual Head Loss in Training (`tools/train_nnue.c`)**:
   - Loss function:
     $$\mathcal{L} = \alpha \cdot \text{HuberLoss}(\hat{M}, M) + (1 - \alpha) \cdot \text{BCE}(\hat{W}, W)$$
     where $M$ is the final game spread margin, $W \in \{0, 1\}$ is game outcome, and $\alpha = 0.5$.
3. **Step 3: Verification & Parity**:
   - Run `make test` and `make zig-test`.
   - Benchmark incremental accumulator throughput (verifying $>5,000,000$ evals/sec).
   - Prepare Tier 2 expansion (`NNUE_MASTER_INPUT_DIM 16384`) in `include/nnue.h` as an L3-resident deep ranking net.
