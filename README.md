# TESSERA

*A Grandmaster Scrabble Engine in Pure Native C, powered by Distributional Belief-Lattice Search.*

Tessera is a tournament Scrabble engine implemented entirely in native C (`-std=c11`, `-O3`, zero external dependencies). Instead of traditional game trees or alpha-beta search, Tessera uses a **position-keyed belief-lattice paradigm** for Scrabble: a persistent directed acyclic graph (DAG) of distributional beliefs relaxed to fixed-point equilibrium via prioritized dirty-queue backward induction.

---

## 1. Engine Benchmarks

| Metric | Measured Value | Implementation |
|---|---|---|
| **Move Generation Throughput** | **1,283,000+ moves/sec** | Appel-Jacobson generator over transposed 15×15 board |
| **Move Generation Latency** | **0.8 – 2.6 ms** | Generates all 2,000+ legal plays for a 7-tile rack |
| **Dictionary Representation** | **0.71 MB (185,702 edges)** | Complete 267,751-word SOWPODS lexicon compressed via minimal DAWG |
| **Dictionary Startup Latency** | **0.0001 seconds** | Zero-copy binary flat-array load (`data/dawg.bin`) |
| **Belief Settling Latency** | **35 – 50 ms** | Full belief-lattice relaxation across 500+ speculative nodes |
| **Transpositions Merged** | **100% by Construction** | 64-bit Zobrist hashed open-addressing lattice map |
| **Bayesian Stopping Confidence** | **98.0% – 99.9%** | Joint Gaussian posterior sampling via the Root Court |

---

## 2. The Belief-Lattice Scrabble Architecture

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
│      The Palimpsest       │          │   The Root Court    │          │          Proofs           │
│ Conjugate Normal Learner  │          │ Bayesian Inference  │          │ Variance-Zero Pins        │
│ Closed-Form Context Prior │          │ Stopping Threshold  │          │ Exact Endgame Solvers     │
└───────────────────────────┘          └─────────────────────┘          └───────────────────────────┘
```

### Component Map

| Component | Files | What it does |
|---|---|---|
| **The Lattice** | `include/lattice.h`, `src/lattice.c` | Position-keyed belief DAG with parent back-pointers. Transpositions merge by construction; indexed by 64-bit Zobrist board hashes. Stores Gaussian beliefs $(\mu, \sigma^2, N)$ and proof certificates. |
| **The Settling Engine** | `include/settle.h`, `src/settle.c` | Prioritized dirty-queue relaxation: any node whose belief moves by $\|\Delta \mu\| > \epsilon$ is re-backed-up and enqueues its parents, so refutations found deep at the frontier propagate rootward in a single continuous wave. |
| **The Broker** | `include/broker.h`, `src/broker.c` | Best-first frontier leaf acquisition: prices leaves by $Q(e) + C_{\text{PUCT}} \cdot P(e) \cdot \frac{\sqrt{N}}{1+n} + C_{\text{VAR}} \cdot \sigma(e)$. Actively buys down uncertainty while pricing proven subtrees at zero. |
| **The Palimpsest** | `include/palimpsest.h`, `src/palimpsest.c` | The evaluator rewrites itself mid-move: maintains conjugate Normal posteriors over evaluation corrections per structural context (vowels $\times$ heavy tiles $\times$ openness), updating in closed form from settled-vs-raw disagreements. |
| **The Root Court** | `include/court.h`, `src/court.c` | Stopping as inference: samples from the joint posterior distribution of candidate moves to estimate $P(m^* \text{ is truly best})$. Moves snap instantly in milliseconds when confidence clears $98\%$; critical choices absorb deliberation. |
| **Proofs** | `src/settle.c`, `src/simulation.c` | In endgames (empty bag), winning sequences enter the lattice as **pinned, variance-zero beliefs** ($\sigma^2 = 0$). Winning replies prove the parent won via ordinary dirty-queue propagation. |
| **Appel-Jacobson Gen** | `include/movegen.h`, `src/movegen.c` | Native C move generator running at **1.28 Million moves/sec** over transposed 15×15 boards with 32-bit cross-check bitmasks. |
| **Web GUI & Demo** | `gui/server.py`, `gui/web/*` | Luxury tournament UI featuring real-time Scrabble board, live lattice telemetry metrics, engine hints, AI vs AI showcase mode, and unseen bag tracker. |

---

## 3. Evaluation & Heuristic Model

### 3.1 Static Move Equity
$$\text{StaticEquity}(M) = \text{Score}(M) + \text{LeaveEquity}(\text{Rack} \setminus M) + \text{BoardRisk}(M)$$

- **Single-Tile Equities:** Blanks (`?`) carry $+26.5$ equity pts; `S` carries $+8.0$; toxic tiles like `Q` (without `U`) or duplicate `V`/`W` carry sharp penalties.
- **Vowel/Consonant Balance:** Severe penalties for all-vowel ($-18.0$ pts) or all-consonant ($-16.0$ pts) leaves.
- **Duplicate Tile Penalties:** Deducts points for duplicate vowels (`II`, `UU`, `OO`) or consonants (`VV`, `WW`, `FF`).
- **High-Bingo Synergies:** Positive equity for key prefixes/suffixes (`QU`, `ING`, `ER`, `ED`, `RE`, `Blank+S`).
- **Board Risk / Hotspots:** Penalizes plays that expose unblocked access to Triple Word (TW) or Triple Letter (TL) squares.

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

# Run full unit test suite
make test
```
