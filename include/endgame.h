#ifndef ENDGAME_H
#define ENDGAME_H

#include "types.h"
#include "board.h"
#include "dict.h"
#include "movegen.h"

/*
 * Exact Minimax Endgame Solver with Alpha-Beta Pruning.
 * Activated when the tile bag is empty (bag->count == 0) and the opponent's
 * rack is known with 100% mathematical certainty.
 * Finds the move sequence that maximizes the final game point spread:
 * Spread = (MyScore - OppScore) + 2 * (Opponent Remaining Rack Value).
 */

typedef struct {
    Move best_move;
    int final_spread;
    uint32_t nodes_searched;
    bool is_proven;
} EndgameResult;

EndgameResult endgame_solve(const Board *board, const Dawg *dawg,
                            const Rack *my_rack, const Rack *opp_rack,
                            int max_depth);

#endif /* ENDGAME_H */
