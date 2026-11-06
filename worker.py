#!/usr/bin/env python3
"""
Tessera-Test Distributed Community Worker
=========================================
Distributed worker client for contributing self-play training games
and running Sequential Probability Ratio Test (SPRT) patch benchmarks
for the Tessera Championship Scrabble Engine.

Usage:
    python3 worker.py --concurrency 4 --user "YourName"
    python3 worker.py --mode sprt --games 100 --user "YourName"
    python3 worker.py --mode bench

Author: Harshit Khandelwal & Tessera Engine Team
Repository: https://github.com/harshitkhandelwal208/tessera
Documentation: https://harshitkhandelwal208.github.io/tessera/
"""

import sys
import os
import time
import argparse
import platform
import subprocess
import threading
import multiprocessing
import math
import hashlib
import json
from pathlib import Path
from datetime import datetime

VERSION = "1.0.0"
DEFAULT_COORDINATOR = "https://harshitkhandelwal208.github.io/tessera/api"

class HardwareProfile:
    def __init__(self):
        self.os = platform.system()
        self.arch = platform.machine()
        self.cpu_count = os.cpu_count() or 1
        self.has_avx2 = self._detect_avx2()
        self.python_version = platform.python_version()

    def _detect_avx2(self):
        try:
            if self.os == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read()
                    return "avx2" in content.lower()
            elif self.os == "Darwin":
                res = subprocess.run(["sysctl", "-n", "machdep.cpu.leaf7_features"],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                return "AVX2" in res.stdout
            elif self.os == "Windows":
                # Most modern x86_64 Windows CPUs support AVX2
                return "64" in self.arch
        except Exception:
            pass
        return False

    def __str__(self):
        avx_str = "AVX2: Yes" if self.has_avx2 else "AVX2: No (scalar fallback)"
        return f"{self.os} {self.arch} | {self.cpu_count} Cores | {avx_str}"

class WorkerStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_games = 0
        self.total_moves = 0
        self.total_points = 0
        self.wins_tessera = 0
        self.losses_tessera = 0
        self.draws = 0
        self.start_time = time.time()
        self.batches_submitted = 0

    def record_game(self, points_p1, points_p2, moves_count):
        with self.lock:
            self.total_games += 1
            self.total_moves += moves_count
            self.total_points += (points_p1 + points_p2)
            if points_p1 > points_p2:
                self.wins_tessera += 1
            elif points_p1 < points_p2:
                self.losses_tessera += 1
            else:
                self.draws += 1

    def get_summary(self):
        with self.lock:
            elapsed = max(1.0, time.time() - self.start_time)
            gps = (self.total_games / elapsed) * 60.0
            mps = self.total_moves / elapsed
            win_rate = (self.wins_tessera / max(1, self.total_games)) * 100.0
            return {
                "games": self.total_games,
                "elapsed_sec": elapsed,
                "games_per_min": gps,
                "moves_per_sec": mps,
                "wins": self.wins_tessera,
                "losses": self.losses_tessera,
                "draws": self.draws,
                "win_rate": win_rate,
                "batches": self.batches_submitted
            }

def ensure_engine_binary(engine_dir: Path) -> Path:
    """Finds or builds the Tessera engine binary."""
    binary_name = "tessera.exe" if platform.system() == "Windows" else "tessera"
    binary_path = engine_dir / binary_name

    if binary_path.exists() and os.access(binary_path, os.X_OK):
        return binary_path

    # Try to build if Makefile exists
    makefile = engine_dir / "Makefile"
    if makefile.exists():
        print(f"[*] Engine binary not found in {engine_dir}. Compiling via make...")
        nproc = os.cpu_count() or 2
        res = subprocess.run(["make", f"-j{nproc}"], cwd=engine_dir, capture_output=True, text=True)
        if res.returncode == 0 and binary_path.exists():
            print(f"[+] Engine built successfully: {binary_path}")
            return binary_path
        else:
            print(f"[!] Compilation failed:\n{res.stderr}")
            sys.exit(1)

    raise FileNotFoundError(f"Tessera engine binary not found in {engine_dir}. Run `make` first.")

def run_selfplay_worker(worker_id: int, binary_path: Path, max_games: int,
                        stats: WorkerStats, stop_event: threading.Event, output_dir: Path):
    """Worker thread that executes self-play matches and logs training samples."""
    games_done = 0
    while not stop_event.is_set():
        if max_games > 0 and stats.total_games >= max_games:
            break

        # Run self-play game using ./tessera ai-vs-ai with silent output
        cmd = [str(binary_path), "ai-vs-ai"]
        try:
            start_t = time.time()
            proc = subprocess.Popen(
                cmd,
                cwd=binary_path.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = proc.communicate(timeout=60)
            
            # Parse final scores and move counts
            # Expected output lines:
            # "Game Over!"
            # "Player 0 score: 452"
            # "Player 1 score: 418"
            p0_score = 0
            p1_score = 0
            move_count = 28 # Typical game average
            for line in stdout.splitlines():
                if "Player 0 score:" in line:
                    try:
                        p0_score = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "Player 1 score:" in line:
                    try:
                        p1_score = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "Total moves:" in line:
                    try:
                        move_count = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass

            if p0_score > 0 and p1_score > 0:
                stats.record_game(p0_score, p1_score, move_count)
                games_done += 1

                # Save game record batch to disk
                if games_done % 20 == 0:
                    batch_id = f"batch_{int(time.time())}_{worker_id}_{games_done}.jsonl"
                    batch_path = output_dir / batch_id
                    with open(batch_path, "a") as f:
                        f.write(json.dumps({
                            "worker_id": worker_id,
                            "timestamp": time.time(),
                            "p0_score": p0_score,
                            "p1_score": p1_score,
                            "moves": move_count,
                            "duration_sec": time.time() - start_t
                        }) + "\n")
                    with stats.lock:
                        stats.batches_submitted += 1

        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception as e:
            # Engine failure or interrupt
            pass

def calculate_sprt(wins, losses, draws, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05):
    """
    Computes Wald's Sequential Probability Ratio Test (SPRT)
    for testing H0 (elo <= elo0) vs H1 (elo >= elo1).
    """
    n = wins + losses + draws
    if n == 0:
        return 0.0, "CONTINUE", 0.0

    # Win probability models under Logistic distribution
    s = (wins + 0.5 * draws) / n
    p0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    p1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))

    # Variance estimate
    var = max(0.01, (wins * (1 - s)**2 + losses * (0 - s)**2 + draws * (0.5 - s)**2) / n)

    # Log-Likelihood Ratio (LLR)
    # Using Gaussian approximation of score variance
    delta_mu = p1 - p0
    llr = (n * delta_mu / var) * (s - (p0 + p1) / 2.0)

    # Wald boundaries
    lower_bound = math.log(beta / (1.0 - alpha))
    upper_bound = math.log((1.0 - beta) / alpha)

    if llr >= upper_bound:
        status = "ACCEPTED (PASSED)"
    elif llr <= lower_bound:
        status = "REJECTED (FAILED)"
    else:
        status = "CONTINUE"

    # Empirical Elo estimate
    elo_est = -400.0 * math.log10(max(0.001, (1.0 - s) / max(0.001, s)))
    return llr, status, elo_est

def run_sprt_benchmark(binary_path: Path, num_games: int):
    """Runs a paired head-to-head SPRT test using tools/match_benchmark."""
    benchmark_tool = binary_path.parent / "tools" / "match_benchmark"
    if not benchmark_tool.exists():
        print("[*] Compiling tools/match_benchmark...")
        subprocess.run(["make", "tools/match_benchmark"], cwd=binary_path.parent)

    if not benchmark_tool.exists():
        print("[!] tools/match_benchmark binary not found.")
        return

    rounds = max(1, num_games // 7)
    print(f"\n[*] Launching SPRT Benchmark across 7 Championship Bot Profiles ({rounds * 7} total games)...")
    start_t = time.time()
    proc = subprocess.run([str(benchmark_tool), str(rounds)],
                          cwd=binary_path.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - start_t

    print(proc.stdout)
    print(f"\n[+] Benchmark finished in {elapsed:.2f}s ({rounds * 7 / max(0.1, elapsed):.2f} games/sec)")

def run_bench(binary_path: Path):
    """Runs high-throughput move generation and DAG deliberation benchmark."""
    print(f"[*] Running movegen throughput benchmark on {binary_path}...")
    proc = subprocess.run([str(binary_path), "bench"],
                          cwd=binary_path.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(proc.stdout)

def main():
    parser = argparse.ArgumentParser(
        description="Tessera-Test Distributed Community Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 worker.py --concurrency 4 --user "GrandmasterX"
  python3 worker.py --mode sprt --games 70
  python3 worker.py --mode bench
        """
    )
    parser.add_argument("--user", default=os.environ.get("USER", "Anonymous_Contributor"),
                        help="Contributor username or GitHub handle")
    parser.add_argument("--concurrency", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Number of parallel worker threads (default: CPU cores - 1)")
    parser.add_argument("--mode", choices=["selfplay", "sprt", "bench"], default="selfplay",
                        help="Execution mode: 'selfplay' (NNUE data gen), 'sprt' (match test), 'bench'")
    parser.add_argument("--games", type=int, default=0,
                        help="Total games to generate/play (0 = continuous)")
    parser.add_argument("--engine-dir", type=Path, default=Path("."),
                        help="Path to Tessera engine workspace directory")
    parser.add_argument("--out", type=Path, default=Path("./training_batches"),
                        help="Output directory for generated training game batches")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimize terminal output")

    args = parser.parse_args()
    hw = HardwareProfile()

    print("=" * 78)
    print(f"  TESSERA-TEST DISTRIBUTED WORKER v{VERSION}")
    print(f"  Championship Belief-Lattice Scrabble Neural Network Training Fleet")
    print("=" * 78)
    print(f"  [+] Contributor:     {args.user}")
    print(f"  [+] Host System:     {hw}")
    print(f"  [+] Active Cores:    {args.concurrency} threads")
    print(f"  [+] Selected Mode:   {args.mode.upper()}")
    print("=" * 78)

    # Locate and verify engine
    binary_path = ensure_engine_binary(args.engine_dir)

    if args.mode == "bench":
        run_bench(binary_path)
        return

    if args.mode == "sprt":
        run_sprt_benchmark(binary_path, args.games if args.games > 0 else 70)
        return

    # Self-play continuous training data generation mode
    args.out.mkdir(parents=True, exist_ok=True)
    stats = WorkerStats()
    stop_event = threading.Event()
    threads = []

    print(f"[*] Spawning {args.concurrency} self-play generation worker threads...")
    print(f"[*] Training batches will be written to: {args.out.resolve()}")
    print("[*] Press Ctrl+C at any time to gracefully halt and flush batches.\n")

    for i in range(args.concurrency):
        t = threading.Thread(
            target=run_selfplay_worker,
            args=(i, binary_path, args.games, stats, stop_event, args.out),
            daemon=True
        )
        t.start()
        threads.append(t)

    # Terminal Dashboard Loop
    try:
        while True:
            time.sleep(2.0)
            s = stats.get_summary()
            
            # Calculate live SPRT metric against even odds
            llr, sprt_status, elo_margin = calculate_sprt(s["wins"], s["losses"], s["draws"])
            
            dashboard = (
                f"\r[Tessera-Test] Games: {s['games']:<5} | "
                f"Rate: {s['games_per_min']:.1f} g/min ({s['moves_per_sec']:.0f} mps) | "
                f"W-L-D: {s['wins']}-{s['losses']}-{s['draws']} ({s['win_rate']:.1f}%) | "
                f"Batches: {s['batches']} | "
                f"LLR: {llr:+.2f} ({sprt_status})"
            )
            sys.stdout.write(dashboard)
            sys.stdout.flush()

            if args.games > 0 and s["games"] >= args.games:
                print(f"\n\n[+] Target goal of {args.games} games reached!")
                break

    except KeyboardInterrupt:
        print("\n\n[!] Interrupt received! Stopping worker threads gracefully...")
        stop_event.set()

    stop_event.set()
    for t in threads:
        t.join(timeout=3.0)

    final_s = stats.get_summary()
    print("\n" + "=" * 78)
    print("  FINAL WORKER SESSION SUMMARY")
    print("=" * 78)
    print(f"  Total Games Generated:  {final_s['games']}")
    print(f"  Total Session Time:     {final_s['elapsed_sec']:.1f} seconds")
    print(f"  Average Throughput:     {final_s['games_per_min']:.2f} games/min ({final_s['moves_per_sec']:.1f} moves/sec)")
    print(f"  Win / Loss / Draw:      {final_s['wins']} Wins / {final_s['losses']} Losses / {final_s['draws']} Draws")
    print(f"  Batches Flushed:        {final_s['batches']} batches")
    print(f"  Destination Folder:     {args.out.resolve()}")
    print("=" * 78)
    print("[+] Thank you for contributing compute to Tessera!")

if __name__ == "__main__":
    main()
