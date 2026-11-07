#ifndef PROTOCOL_H
#define PROTOCOL_H

#include "types.h"
#include "board.h"
#include "dict.h"
#include "movegen.h"
#include "eval.h"
#include "simulation.h"

/*
 * Runs the interactive terminal game (Human vs AI or AI vs AI).
 */
void protocol_run_terminal_game(const Dawg *dawg, bool ai_vs_ai);

/*
 * Runs the JSON/line protocol loop over stdin/stdout for external clients.
 */
void protocol_run_server_loop(const Dawg *dawg);

/*
 * Evaluates a single position query from CLI arguments.
 */
void protocol_solve_single(const Dawg *dawg, const char *rack_str, const char *board_file);

/*
 * Benchmarks engine speed (nodes/sec, moves/sec, sim speed).
 */
void protocol_run_benchmark(const Dawg *dawg);

#endif /* PROTOCOL_H */
