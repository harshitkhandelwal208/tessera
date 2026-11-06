#ifndef SIMULATION_H
#define SIMULATION_H

#include "types.h"
#include "board.h"
#include "dict.h"
#include "movegen.h"
#include "eval.h"
#include "zobrist.h"
#include "palimpsest.h"
#include "lattice.h"
#include "settle.h"
#include "broker.h"
#include "court.h"

typedef struct {
    int max_deliberation_steps; /* Max leaf expansion iterations (e.g. 50-100) */
    float confidence_bar;       /* Root Court Bayesian confidence threshold (e.g. 0.98) */
    bool use_palimpsest;        /* Whether evaluator rewrites itself mid-move */
    bool use_endgame_solver;    /* Exact playout solver when bag is empty */
    void *persistent_tracker;   /* Optional persistent BayesRackTracker across game turns */
} EngineConfig;

typedef struct {
    uint32_t nodes_in_lattice;
    uint32_t transpositions_merged;
    uint32_t dirty_passes_settled;
    uint32_t frontier_leaves_bought;
    float court_confidence;     /* P(argmax is best) */
    float palimpsest_max_bias;  /* Max learned evaluation correction */
    double deliberation_ms;
} EngineMetrics;

void engine_config_default(EngineConfig *config);

/*
 * The Distributional Belief-Lattice Scrabble Engine:
 * Replaces tree rollouts with a position-keyed belief lattice relaxed by
 * prioritized dirty-queue backward induction, guided by the best-first
 * uncertainty broker, learning via the self-rewriting Palimpsest, and
 * concluding deliberation via the Bayesian Root Court.
 */
Move lattice_search(const Board *board, const Dawg *dawg, const Rack *rack,
                    const TileBag *bag, const EngineConfig *config,
                    MoveList *all_moves, EngineMetrics *metrics);

/* Unseen tiles pool helper */
int sim_get_unseen_tiles(const Board *board, const Rack *player_rack, char unseen[100]);

/* Backward compatibility wrapper for existing callers */
Move sim_find_best_move(const Board *board, const Dawg *dawg, const Rack *rack,
                        const TileBag *bag, void *unused_config,
                        MoveList *all_moves);

#endif /* SIMULATION_H */
