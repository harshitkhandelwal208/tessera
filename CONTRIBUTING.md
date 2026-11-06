# Contributing to Tessera

Thank you for your interest in contributing to **Tessera**! Tessera is an open-source, championship-grade tournament Scrabble intelligence engine implemented in standard ISO C11 with zero external dependencies.

Like modern open-source engines (such as Stockfish and Reckless), Tessera follows a strict, data-driven development workflow where every code modification is evaluated through rigorous benchmarking and statistical testing.

---

## Table of Contents
1. [General Philosophy & Tenets](#general-philosophy--tenets)
2. [Types of Contributions](#types-of-contributions)
   - [Functional Changes](#1-functional-changes)
   - [Non-Functional Simplifications](#2-non-functional-simplifications)
   - [Performance Optimizations](#3-performance-optimizations)
3. [Testing & Validation Protocol](#testing--validation-protocol)
   - [Unit Test Suite](#unit-test-suite)
   - [Movegen & Search Benchmark](#movegen--search-benchmark)
   - [Tournament Matrix Evaluation](#tournament-matrix-evaluation)
   - [SPRT (Sequential Probability Ratio Test)](#sprt-sequential-probability-ratio-test)
4. [Contributing Compute (Tessera-Test)](#contributing-compute-tessera-test)
5. [Coding Standards & Style Guide](#coding-standards--style-guide)
6. [Submitting a Pull Request](#submitting-a-pull-request)

---

## General Philosophy & Tenets

* **Zero External Dependencies:** The engine core must compile and run using strictly the standard C library (`libc`, `libm`, and OpenMP for offline training/benchmark tools). No external libraries (Boost, GLib, etc.) are permitted.
* **ISO C11 Standard:** All code must strictly conform to ISO C11 (`-std=c11 -Wall -Wextra`).
* **Zero Allocations on the Hot Path:** Deliberation, move generation, and evaluation routines must perform zero heap allocations (`malloc`/`free`/`calloc`) during execution. All scratch buffers, move lists, and transposition nodes must reside in static pre-allocated memory or depth-indexed pools.
* **Portability First:** Architecture-specific acceleration (such as x86-64 AVX2 SIMD) must always feature a portable scalar fallback for ARM64 (Apple Silicon, Raspberry Pi, AWS Graviton) and legacy architectures.
* **Empirical Validation:** Changes to search, evaluation, or move sorting must demonstrate statistical superiority via head-to-head matches or Sequential Probability Ratio Testing (SPRT).

---

## Types of Contributions

### 1. Functional Changes
A functional change alters the moves played by the engine, changes evaluation scores, adjusts leaf pruning/pricing, or modifies neural network weights.
* **Requirement:** Must be tested via SPRT or tournament match benchmarks against the baseline master branch.
* Typical SPRT bounds for evaluation or search patches:
  * Elo bounds: $[0.0, 5.0]$ or $[-1.0, 4.0]$
  * Significance levels: $\alpha = 0.05, \beta = 0.05$

### 2. Non-Functional Simplifications
A non-functional simplification improves code readability, refactors logic, or eliminates unnecessary complexity without altering move choices or evaluation results.
* **Requirement:** Must produce the **exact same movegen count and bench signature** as the baseline code on identical test boards.
* Must not introduce performance regressions (measured via `./tessera bench`).

### 3. Performance Optimizations
An optimization speeds up move generation, transposition lookups, or neural network inference without altering gameplay decisions.
* **Requirement:** Must preserve identical outputs while demonstrating a measurable increase in moves per second (MPS) or nodes per second across repeated runs.

---

## Testing & Validation Protocol

Before opening a pull request, verify that your changes pass all testing tiers:

### Unit Test Suite
Verify that all core subsystems (DAWG lexicon, Appel-Jacobson generator, leave evaluations, proof propagation, and exact endgame solver) pass cleanly:
```bash
make clean
make -j$(nproc)
make test
```
*All tests must pass with zero assertions or errors.*

### Movegen & Search Benchmark
Execute the built-in throughput benchmark to verify moves-per-second and search latency:
```bash
./tessera bench
```
Ensure move counts and board evaluations match expected baseline signatures.

### Tournament Matrix Evaluation
Benchmark your patch against the 7 standard bot archetypes (Greedy, Turtler, Bingo Hunter, Heuristic GM, Neural Static, Quackle Simulation, and MAGPIE 2-Ply):
```bash
make tools/match_benchmark
./tools/match_benchmark 10   # Plays 70 tournament games
```
Verify that the overall Tournament Performance Rating (TPR) remains $\ge 2000$ Elo with a positive score spread.

### SPRT (Sequential Probability Ratio Test)
For functional patches, run a head-to-head SPRT test using the community worker client:
```bash
python3 worker.py --mode sprt --games 70
```

---

## Contributing Compute (Tessera-Test)

You don't need to write C code to contribute! Training the dual-headed NNUE and evaluating proposed patches requires tens of thousands of self-play games.

You can donate idle CPU cycles by running the standalone Python 3 worker:
```bash
curl -sSL https://harshitkhandelwal208.github.io/tessera/worker.py -o worker.py
python3 worker.py --concurrency 4 --user "YourGitHubUsername"
```

The worker script automatically:
1. Detects your CPU architecture and core count.
2. Compiles or verifies the engine binary.
3. Generates high-quality self-play games and paired matches.
4. Flushes training batches to `training_batches/` for neural network training.

---

## Coding Standards & Style Guide

* **Indentation & Formatting:** 4 spaces per indentation level. No hard tabs.
* **Naming Conventions:**
  * Functions: `module_action_target()` (e.g., `dict_load_binary()`, `lattice_insert_node()`)
  * Structs & Types: `PascalCase` (e.g., `Board`, `Rack`, `LatticeNode`, `NNUEModel`)
  * Macros & Constants: `UPPER_SNAKE_CASE` (e.g., `NNUE_L1_DIM`, `MAX_LEGAL_MOVES`)
  * Static Helper Functions: Always mark file-internal helper functions with `static` or `static inline`.
* **Memory Safety:**
  * Check all array bounds explicitly.
  * Never use unsafe functions (`gets`, `strcpy` without bound checks).
  * Check file pointers returned by `fopen()`.
* **Compiler Cleanliness:**
  * Code must compile without any warnings under `-Wall -Wextra -std=c11`.

---

## Submitting a Pull Request

1. **Fork the Repository:** Fork `harshitkhandelwal208/tessera` on GitHub.
2. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/my-optimization
   ```
3. **Commit Your Changes:**
   Follow conventional commit messages:
   * `feat: ...` for new heuristics or search logic
   * `fix: ...` for bug or memory fixes
   * `perf: ...` for throughput optimizations
   * `refactor: ...` for non-functional simplifications
   * `docs: ...` for documentation and wiki additions
4. **Push and Open a Pull Request:**
   * Provide a clear description of the problem, the proposed solution, and empirical benchmark numbers.
   * Attach the output of `make test` and `./tessera bench`.
   * For functional changes, include the SPRT or tournament matrix result.
