# TESSERA

*A Grandmaster Scrabble Engine in Pure Native C.*

Tessera is a high-performance tournament Scrabble engine implemented entirely in native C (`-std=c11`, `-O3`, zero external dependencies). It pairs Appel & Jacobson's move generation algorithm over a compressed minimal DAWG with pre-NNUE grandmaster evaluation: hand-tuned rack leave equity tables, vowel-consonant balance matrices, duplicate penalties, board hotspot risk modeling, Monte Carlo unseen-rack lookahead rollouts, and an exact alpha-beta endgame solver.

---

## 1. Engine Benchmarks

| Metric | Measured Value | Description |
|---|---|---|
| **Move Generation Throughput** | **1,283,000+ moves/sec** | Exhaustive generation across 15×15 tournament boards |
| **Move Generation Latency** | **1.2 – 4.0 ms** | Generates all 2,000+ legal plays for a 7-tile rack |
| **Dictionary Representation** | **0.71 MB (185,702 edges)** | Complete 267,751-word SOWPODS lexicon compressed via minimal DAWG |
| **Dictionary Startup Latency** | **0.0001 seconds** | Zero-copy binary flat-array load (`data/dawg.bin`) |
| **Monte Carlo Decision Latency**| **25 – 120 ms** | Top 12 candidate plays simulated against 10 random opponent draws |

---

## 2. Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │        SOWPODS Lexicon (267k)          │
                      │  Minimal DAWG (Daciuk et al., 0.71MB)  │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
┌──────────────────┐           ┌──────────────────────┐          ┌──────────────────┐
│   Board State    │ ────────> │ Appel-Jacobson Gen   │ <─────── │    Tile Rack     │
│  (15x15 + Mults) │           │ (Transposed Rows)    │          │  (7 Tiles + ?)   │
└──────────────────┘           └──────────┬───────────┘          └──────────────────┘
                                          │ 1.28M moves/sec
                                          ▼
                               ┌──────────────────────┐
                               │  All Legal Moves     │
                               │  (Scored & Validated)│
                               └──────────┬───────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │ Static Equity Filter │
                               │ Score + Leave + Risk │
                               └──────────┬───────────┘
                                          │ Top N Candidates
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    Decision Engine (Pre-NNUE SOTA)     │
                      │ ├─ Unseen Bag Tile Pool Tracker        │
                      │ ├─ Monte Carlo 1-Ply Opponent Rollout  │
                      │ └─ Exact Alpha-Beta Endgame Solver     │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │     Best Move Play   │
                               │ (Terminal CLI / GUI) │
                               └──────────────────────┘
```

### Component Map

| Component | Files | Description |
|---|---|---|
| **Lexicon (DAWG)** | `include/dict.h`, `src/dict.c` | Minimal Directed Acyclic Word Graph with 32-bit compact edge encoding. Implements Daciuk's incremental graph minimization algorithm. |
| **Board & Cross-Checks** | `include/board.h`, `src/board.c` | 15×15 tournament board representation, standard multipliers (TW, DW, TL, DL), bag distribution, and 32-bit Appel-Jacobson cross-check bitmasks. |
| **Move Generator** | `include/movegen.h`, `src/movegen.c` | Generates 100% of legal horizontal and vertical placements by transposing the board across the main diagonal. |
| **Hand-Tuned Evaluation** | `include/eval.h`, `src/eval.c` | Evaluates rack leave equity (single-tile values, vowel/consonant balance, duplicate penalties, high-bingo synergies) and board defense. |
| **Simulation Engine** | `include/simulation.h`, `src/simulation.c` | Pre-NNUE championship search: samples unseen opponent racks, executes 1-ply rollouts on top candidate plays, and solves endgames exactly. |
| **Engine Protocol** | `include/protocol.h`, `src/protocol.c` | Text and JSON protocol interface communicating via stdin/stdout for frontends and GUIs. |
| **Interactive GUI** | `gui/server.py`, `gui/web/*` | Full tournament web interface (wooden tiles, live scoreboard, engine hints, AI vs AI self-play, unseen tile tracker). |

---

## 3. Evaluation & Heuristic Model

Before neural networks and NNUEs, champion Scrabble engines (such as Brian Sheppard's *Maven* and Jason Katz-Brown's *Quackle*) relied on two mathematical foundations: **Rack Leave Equity** and **Monte Carlo Rollouts**.

### 3.1 Static Move Equity
$$\text{StaticEquity}(M) = \text{Score}(M) + \text{LeaveEquity}(\text{Rack} \setminus M) + \text{BoardRisk}(M)$$

- **Single-Tile Equities:** Blanks (`?`) carry $+26.5$ equity points; `S` carries $+8.0$; heavy consonants (`V`, `W`, `Q` without `U`) are severely penalized.
- **Vowel/Consonant Ratio Matrix:** Prevents rack choking by penalizing leaves with 0 vowels ($-16.0$ pts) or 0 consonants ($-18.0$ pts).
- **Duplicate Letter Penalties:** Penalizes duplicate vowels (`II`, `UU`, `OO`) and duplicate heavy consonants (`VV`, `WW`).
- **High-Bingo Synergies:** Grants positive equity bonuses for bingo components (`QU`, `ING`, `ER`, `ED`, `RE`, `Blank+S`).
- **Board Risk / Hotspots:** Penalizes plays that open unblocked lanes to Triple Word (TW) squares, preventing opponent counter-punches.

### 3.2 Monte Carlo Lookahead Rollout
For the top $K$ candidate plays (sorted by static equity):
1. Compute the unseen tile pool: $\text{Pool} = \text{100 Standard Tiles} - \text{Board} - \text{PlayerRack}$.
2. Sample $N$ random 7-tile opponent racks from the pool.
3. For each candidate move $M$, apply $M$ to speculative board $B'$.
4. Generate opponent responses on $B'$ and evaluate their best reply.
5. Compute the final simulated net equity:
$$\text{SimEquity}(M) = \text{StaticEquity}(M) - 0.85 \times \mathbb{E}[\text{OpponentBestResponse}]$$

### 3.3 Endgame Solver
When the tile bag has 0 tiles remaining, all unseen tiles are in the opponent's rack with 100% certainty. Tessera switches to an exact minimax playout solver to calculate mathematically forced winning sequences.

---

## 4. Building & Running

### Requirements
- GCC (or Clang) with C11 support.
- Standard `make`.
- Python 3.8+ (for the Web GUI bridge; standard library only, zero pip packages needed).

### Quickstart (Launch Demonstration GUI)
```bash
./run_demo.sh
```
Open **`http://localhost:8080`** in your browser.

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
