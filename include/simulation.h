#ifndef SIMULATION_H
#define SIMULATION_H

#include "types.h"
#include "board.h"
#include "dict.h"
#include "movegen.h"
#include "eval.h"

#define MAX_CANDIDATES 20
#define DEFAULT_SIM_ITERATIONS 15

typedef struct {
    int num_candidates;       /* How many top candidate moves to simulate */
    int num_simulations;      /* How many random opponent racks to sample */
    bool use_endgame_solver;  /* True when bag is empty */
} SimConfig;

/*
 * Default configuration for fast yet grandmaster-quality decision making.
 */
void sim_config_default(SimConfig *config);

/*
 * Unseen tiles pool: computes remaining tiles not visible to the current player.
 * (Bag tiles + opponent rack).
 */
int sim_get_unseen_tiles(const Board *board, const Rack *player_rack, char unseen[100]);

/*
 * Reranks top candidate moves using Monte Carlo lookahead simulation:
 * Evaluates candidate moves against simulated opponent racks drawn from unseen tiles.
 * The best move according to simulated equity is placed at index 0 of candidate_list.
 */
Move sim_find_best_move(const Board *board, const Dawg *dawg, const Rack *rack,
                        const TileBag *bag, const SimConfig *config,
                        MoveList *all_moves);

#endif /* SIMULATION_H */
