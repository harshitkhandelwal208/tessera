#ifndef ENDGAME_H
#define ENDGAME_H

#include <stdbool.h>
#include "types.h"
#include "board.h"
#include "dict.h"

/*
 * Exact Deterministic Graph Endgame Solver for Scrabble.
 * Solves the final game-out sequence with zero uncertainty when the bag is empty.
 * Operates on a pure position-keyed DAG with Zobrist transposition hashing.
 * Returns true if an exact optimal sequence was solved, setting out_best_move and out_net_margin.
 */
bool endgame_solve_exact(const Board *board, const Dawg *dawg,
                         const Rack *my_rack, const Rack *opp_rack,
                         Move *out_best_move, int *out_net_margin);

/*
 * Release thread-local memory pools allocated during endgame search.
 */
void endgame_cleanup(void);

#endif /* ENDGAME_H */
