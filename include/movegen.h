#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "types.h"
#include "board.h"
#include "dict.h"

/*
 * Precomputed Board Cross-Check and Transposition Cache (Stockfish / Reckless Blueprint).
 * Eliminates 100% of redundant cross-check recomputations and board transpositions
 * across Monte Carlo rollout counterplay and draw realization steps.
 */
typedef struct {
    Board transposed;
    CrossCheck h_checks[BOARD_SIZE][BOARD_SIZE];
    CrossCheck v_checks[BOARD_SIZE][BOARD_SIZE];
} BoardCrossCache;

void board_cross_cache_init(BoardCrossCache *cache, const Board *board, const Dawg *dawg);

/* Initialize and destroy a move list */
void move_list_init(MoveList *list);
void move_list_destroy(MoveList *list);

/*
 * Generate all legal moves for the given rack on the board.
 * Returns the total count of legal moves generated.
 * All moves are fully scored and validated against the DAWG.
 */
int movegen_generate_all(const Board *board, const Dawg *dawg, const Rack *rack, MoveList *list);
int movegen_generate_all_cached(const Board *board, const BoardCrossCache *cache, const Dawg *dawg, const Rack *rack, MoveList *list);

/*
 * High-performance on-the-fly opponent response evaluator.
 * Evaluates legal moves directly in registers without allocating MoveList or computing leaves.
 * Returns the highest immediate score the given rack can achieve on this board.
 */
int movegen_find_best_response_score(const Board *board, const Dawg *dawg, const Rack *rack);
int movegen_find_best_response_score_cached(const Board *board, const BoardCrossCache *cache, const Dawg *dawg, const Rack *rack);

/*
 * High-performance on-the-fly opponent response equity evaluator.
 * Evaluates legal moves directly in registers, incorporating score and rack leave equity.
 * Returns the highest total equity (score + 0.85 * leave) the given rack can achieve.
 */
float movegen_find_best_response_equity(const Board *board, const Dawg *dawg, const Rack *rack, int bag_count);
float movegen_find_best_response_equity_cached(const Board *board, const BoardCrossCache *cache,
                                               const Dawg *dawg, const Rack *rack, int bag_count);

/* Transpose helper */
void board_transpose(Board *dst, const Board *src);

#endif /* MOVEGEN_H */
