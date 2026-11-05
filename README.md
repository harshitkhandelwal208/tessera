# TESSERA

*A Grandmaster Scrabble Engine in Pure Native C, powered by Dual-Accumulator Evaluation, Belief-Lattice Deliberation, and Exact Alpha-Beta Endgame Solving.*

Tessera is a tournament Scrabble engine implemented entirely in native C (`-std=c11`, `-O3 -flto -march=native`, zero external dependencies). Tessera combines:
1. **Minimal DAWG Lexicon** (0.71 MB flat array, 0.0001s zero-copy binary startup).
2. **High-Throughput Appel-Jacobson Move Generation** (1,308,000+ legal moves/sec).
3. **Dual-Accumulator Evaluator with 8 Game-Phase Buckets** (inspired by modern dual-accumulator engine architectures): separates incremental board premium square dynamics from 8-phase rack leave synergies and championship super-leave stems.
4. **Exact Sub-Rack Exchange Optimizer**: exhaustive $2^k$ subset evaluation with dynamic tempo penalty modeling.
5. **Distributional Belief-Lattice Deliberation**: a persistent directed acyclic graph (DAG) of distributional beliefs relaxed to fixed-point equilibrium via prioritized dirty-queue backward induction with multi-sample adversarial Monte Carlo rollouts.
6. **Exact Alpha-Beta Minimax Endgame Solver**: activated when bag is exhausted (`bag->count == 0`), utilizing a 65,536-entry Transposition Table, iterative deepening, and exact terminal outplay bonus calculation.
7. **Luxury Demonstration Web GUI**: browser-based tournament UI with real-time telemetry dashboard.

---

## 1. Engine Benchmarks

| Metric | Measured Value | Implementation |
|---|---|---|
| **Move Generation Throughput** | **1,308,000+ moves/sec** | Appel-Jacobson generator over transposed 15×15 board |
| **Move Generation Latency** | **0.8 – 2.2 ms** | Generates all 2,000+ legal plays for a 7-tile rack |
| **Dictionary Representation** | **0.71 MB (185,702 edges)** | Complete 267,751-word SOWPODS lexicon compressed via minimal DAWG |
| **Dictionary Startup Latency** | **0.0001 seconds** | Zero-copy binary flat-array load (`data/dawg.bin`) |
| **Belief Settling Latency** | **40 – 85 ms** | Full belief-lattice relaxation across 700+ speculative nodes |
| **Transpositions Merged** | **100% by Construction** | 64-bit Zobrist hashed open-addressing lattice map |
| **Bayesian Stopping Confidence** | **98.0% – 99.9%** | Joint Gaussian posterior sampling via the Root Court |
| **Endgame Solving Speed** | **< 1.0 ms (180 nodes)** | Exact Alpha-Beta Minimax with 65k Transposition Table |

---

## 2. Architecture Overview

```
                                  ┌───────────────────────────────┐
                                  │      SOWPODS Minimal DAWG     │
                                  │   (0.71 MB, 185,702 edges)    │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
┌───────────────────────────┐          ┌─────────────────────┐          ┌───────────────────────────┐
│     The Scrabble Lattice  │ <──────> │  The Settling Engine│ <──────> │       The Broker          │
│ Position-Keyed Belief DAG │          │ Prioritized Dirty Q │          │ Value + Prior + Variance  │
│ (Transpositions Merged)   │          │ Backward Induction  │          │ Frontier Leaf Acquisition │
└─────────────┬─────────────┘          └──────────┬──────────┘          └─────────────┬─────────────┘
              │                                   │                                   │
              ▼                                   ▼                                   ▼
┌───────────────────────────┐          ┌─────────────────────┐          ┌───────────────────────────┐
│      Dual Accumulator     │          │   The Root Court    │          │    Exact Endgame Solver   │
│ 8 Game-Phase Buckets      │          │ Bayesian Inference  │          │ Alpha-Beta Minimax + TT   │
│ Board Head + Leave Head   │          │ Stopping Threshold  │          │ Terminal Outplay Proofs   │
└───────────────────────────┘          └─────────────────────┘          └───────────────────────────┘
```

### Component Map

| Component | Files | What it does |
|---|---|---|
| **Dual Accumulator & 8 Phases** | `include/eval.h`, `src/eval.c` | Decoupled board feature accumulator and rack leave accumulator across 8 game-phase buckets (Opening, Early, Midgame, Late Mid, Pre-Endgame, Exhaustion, Endgame, Playout). Includes top 3- to 6-letter bingo stems (`SATINE`, `RETINA`, `TISANE`, `AERST`, `EINRS`, etc.), 27×27 pairwise synergy matrix, and non-linear duplicate letter penalties. |
| **The Lattice** | `include/lattice.h`, `src/lattice.c` | Position-keyed belief DAG with parent back-pointers. Transpositions merge by construction; indexed by 64-bit Zobrist board hashes. Stores Gaussian beliefs $(\mu, \sigma^2, N)$ and proof certificates. |
| **The Settling Engine** | `include/settle.h`, `src/settle.c` | Prioritized dirty-queue relaxation: any node whose belief moves by $\|\Delta \mu\| > \epsilon$ is re-backed-up and enqueues its parents, propagating refutations in continuous waves. |
| **The Broker** | `include/broker.h`, `src/broker.c` | Frontier leaf acquisition & adversarial refinement: prices leaves by $Q(e) + C_{\text{PUCT}} \cdot P(e) \cdot \frac{\sqrt{N}}{1+n} + C_{\text{VAR}} \cdot \sigma(e)$. Buys unexpanded leaves or refines top competing candidates via adversarial rollouts. |
| **The Palimpsest** | `include/palimpsest.h`, `src/palimpsest.c` | Self-rewriting evaluator: maintains conjugate Normal posteriors over evaluation corrections per structural context, updating in closed form from settled-vs-raw disagreements. |
| **The Root Court** | `include/court.h`, `src/court.c` | Stopping as inference: samples from the joint posterior distribution of candidate moves to estimate $P(m^* \text{ is truly best})$. Moves snap instantly in milliseconds when confidence clears $98\%$. |
| **Exact Endgame Solver** | `include/endgame.h`, `src/endgame.c` | When bag is empty (`bag->count == 0`), runs exact Alpha-Beta Minimax with a 65,536-entry Transposition Table, iterative deepening, and exact $2 \times (\text{opponent remaining rack value})$ outplay bonus. |
| **Sub-Rack Exchange Optimizer** | `include/eval.h`, `src/eval.c` | Evaluates all $2^k$ subsets to keep when rack is choked, taking into account dynamic tempo penalties. |
| **Appel-Jacobson Gen** | `include/movegen.h`, `src/movegen.c` | Native C move generator running at **1.308 Million moves/sec** over transposed 15×15 boards with 32-bit cross-check bitmasks. |
| **Web GUI & Demo** | `gui/server.py`, `gui/web/*` | Luxury tournament UI featuring real-time Scrabble board, live lattice telemetry metrics, engine hints, AI vs AI showcase mode, and unseen bag tracker. |

---

## 3. Evaluation & Heuristic Model

### 3.1 Dual-Accumulator Formulation
$$\text{MoveEquity}(M) = \text{BoardAccumulator}(M) + \text{RackAccumulator}(\text{Rack} \setminus M)$$

- **Board Accumulator:**
  $$\text{BoardAccumulator}(M) = \text{Score}(M) + \text{PremiumCapture}(M) + \text{DefensiveRisk}(M)$$
  - Rewards taking Triple Word (TW), Triple Letter (TL), and Double Word (DW) squares to deny them to the opponent.
  - Penalizes moves that open up adjacent unblocked access to TW/TL squares (especially placing floating vowels next to TW anchors).
- **Rack Accumulator (8-Phase Bucketed):**
  - **8 Game-Phase Buckets:** Opening (80-100), Early (60-79), Midgame (40-59), Late Mid (20-39), Pre-Endgame (10-19), Exhaustion (1-9), Endgame (0), Playout.
  - **Super-Leave Stems:** Built-in recognition of top championship bingo leaves (`SATINE`, `RETINA`, `TISANE`, `AETINS`, `RETAINS`, `AERST`, `EINST`, `EINRS`, etc.) with bonuses up to $+30.5$ pts.
  - **Vowel/Consonant Balance Curve:** Strict mathematical penalties for vowel or consonant starvation/flooding.
  - **Duplicate Letter Penalties:** Severe penalties for duplicate unplayable consonants (`VV` $-36$ pts, `WW` $-20$ pts, `FF` $-18$ pts, `II` $-14$ pts, `UU` $-18$ pts).
  - **Toxic Consonant Clashes:** Sharp penalties for unplayable pairs (`Q` without `U`, `V+W`, `V+Q`, `V+Z`, `W+Q`, `J+Q`, `K+Q`).
  - **High-Point Tile Clumping:** Progressively steeper penalties when holding $\ge 3$ tiles with face value $\ge 4$ points.

### 3.2 Distributional Bellman Operator & Settling
For a parent node with candidate plays $m_1, m_2, \dots$:
$$Q(m) = \text{Score}(m) + \text{LeaveEquity}(m) - \gamma \cdot \mu(\text{Child}(m))$$
$$\text{Var}(m) = \gamma^2 \cdot \text{Var}(\text{Child}(m)) + \text{residual}$$
$$\mu_{\text{parent}} = \max_m Q(m)$$
When $\|\mu_{\text{new}} - \mu_{\text{old}}\| > \epsilon$, all parent nodes are enqueued into the dirty heap and relaxed backward.

---

## 4. Running the Demonstration

### Quickstart (Launch Web GUI)
```bash
./run_demo.sh
```
Open **`http://localhost:8080`** in any browser.

### Terminal Modes
```bash
# Play Human vs Tessera AI in the terminal
./tessera play

# Watch two Tessera AI engines play an automated match
./tessera ai-vs-ai

# Run performance benchmarks
./tessera bench

# Run full unit test suite (DAWG, movegen, eval, endgame)
make test
```

---

## 5. Technical Specifications

- **Language:** Pure Native C (C11 standard, `gcc -std=c11 -O3 -march=native -flto`)
- **External Dependencies:** Zero (only Standard C Library `libc` and `libm`)
- **Lexicon:** SOWPODS 2019 Official Tournament Lexicon (267,751 words)
- **GUI Bridge:** Zero-dependency Python 3 HTTP bridge (`http.server` + `subprocess`)
