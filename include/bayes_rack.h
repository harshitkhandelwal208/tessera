#ifndef BAYES_RACK_H
#define BAYES_RACK_H

#include <stdint.h>
#include <stdbool.h>
#include "types.h"
#include "board.h"

#define NUM_DISTINCT_TILES 27 /* A-Z + '?' */

typedef struct {
    float tile_weights[NUM_DISTINCT_TILES]; /* Relative Bayesian likelihood [0.1 .. 4.0] */
    int unseen_counts[NUM_DISTINCT_TILES];
    int total_unseen;
} BayesRackTracker;

void bayes_rack_init(BayesRackTracker *tracker);

/* Recompute unseen tile distribution from board and active rack */
void bayes_rack_update_unseen(BayesRackTracker *tracker, const Board *board, const Rack *active_rack);

/* Update likelihoods based on opponent's played move and previous board context */
void bayes_rack_observe_opponent(BayesRackTracker *tracker, const Board *board_before, const Move *opp_move);

/* Sample an opponent rack conditionally according to Bayesian posterior weights */
void bayes_rack_sample_rack(const BayesRackTracker *tracker, Rack *out_rack, int draw_count);

/* Sample an opponent rack conditionally from posterior AND return the remaining unseen bag tiles shuffled */
void bayes_rack_sample_with_bag(const BayesRackTracker *tracker, Rack *out_opp_rack, int opp_draw_count,
                                char *out_remaining_bag, int *out_remaining_count);

#endif /* BAYES_RACK_H */
