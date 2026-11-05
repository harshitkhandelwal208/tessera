#include "simulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

void engine_config_default(EngineConfig *config) {
    config->max_deliberation_steps = 45;
    config->confidence_bar = 0.98f;
    config->use_palimpsest = true;
    config->use_endgame_solver = true;
}

int sim_get_unseen_tiles(const Board *board, const Rack *player_rack, char unseen[100]) {
    int counts[27];
    for (int i = 0; i < 27; i++) {
        counts[i] = TILE_BAG_DISTRIBUTION[i];
    }

    /* Subtract tiles on board */
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            char t = board->cells[r][c].tile;
            if (t != 0) {
                if (board->cells[r][c].is_blank) {
                    counts[26]--;
                } else if (t >= 'A' && t <= 'Z') {
                    counts[t - 'A']--;
                }
            }
        }
    }

    /* Subtract player rack */
    for (int i = 0; i < player_rack->count; i++) {
        char t = player_rack->tiles[i];
        if (t == '?') {
            counts[26]--;
        } else if (t >= 'A' && t <= 'Z') {
            counts[t - 'A']--;
        }
    }

    int unseen_count = 0;
    for (int i = 0; i < 26; i++) {
        for (int k = 0; k < counts[i]; k++) {
            unseen[unseen_count++] = (char)('A' + i);
        }
    }
    for (int k = 0; k < counts[26]; k++) {
        unseen[unseen_count++] = '?';
    }

    return unseen_count;
}

static void shuffle_tiles(char *tiles, int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        char tmp = tiles[i];
        tiles[i] = tiles[j];
        tiles[j] = tmp;
    }
}

Move lattice_search(const Board *board, const Dawg *dawg, const Rack *rack,
                    const TileBag *bag, const EngineConfig *config,
                    MoveList *all_moves, EngineMetrics *metrics) {
    clock_t start_time = clock();

    EngineConfig cfg;
    if (config) cfg = *config;
    else engine_config_default(&cfg);

    int bag_remaining = bag ? bag->count : 50;

    /* 1. Initialize lattice components */
    Lattice *lat = lattice_create(16384);
    DirtyQueue dq;
    dirty_queue_init(&dq);
    Palimpsest palimpsest;
    palimpsest_init(&palimpsest);
    Broker broker;
    broker_init(&broker);

    /* 2. Register and expand Root Node */
    bool created = false;
    uint32_t root_idx = lattice_find_or_insert(lat, board, 0, &created);
    lat->root_index = root_idx;

    lattice_expand_node(lat, root_idx, dawg, rack, cfg.use_palimpsest ? &palimpsest : NULL, bag_remaining);
    LatticeNode *root = &lat->nodes[root_idx];

    /* Handle pass or exchange if no moves */
    if (root->num_edges == 0) {
        lattice_free(lat);
        Move ex_move;
        if (eval_should_exchange(board, rack, bag_remaining, 0.0f, &ex_move)) {
            return ex_move;
        }
        Move pass_move;
        memset(&pass_move, 0, sizeof(Move));
        pass_move.type = MOVE_PASS;
        return pass_move;
    }

    /* Check if exchange is superior to best initial play */
    Move ex_move;
    float best_play_static_equity = eval_move_static_bucketed(board, &root->edges[0].move, bag_remaining);
    if (eval_should_exchange(board, rack, bag_remaining,
                            best_play_static_equity, &ex_move)) {
        lattice_free(lat);
        return ex_move;
    }

    /* 3. Gather Unseen Tiles */
    char unseen[100];
    int unseen_count = sim_get_unseen_tiles(board, rack, unseen);

    uint32_t total_dirty_passes = 0;

    CourtVerdict verdict = court_deliberate(lat, root_idx);

    /* --- LATTICE SETTLING DELIBERATION LOOP WITH PAIRED COUNTERPLAY --- */
    int top_k = root->num_edges < 5 ? root->num_edges : 5;

    for (int step = 0; step < cfg.max_deliberation_steps && !verdict.should_adjourn; step++) {
        /* Sample a concrete opponent rack from the unseen pool */
        Rack sim_opp_rack;
        rack_init(&sim_opp_rack);

        if (bag_remaining > 0) {
            shuffle_tiles(unseen, unseen_count);
        }
        int draw_cnt = unseen_count < RACK_CAPACITY ? unseen_count : RACK_CAPACITY;
        for (int i = 0; i < draw_cnt; i++) {
            rack_add_tile(&sim_opp_rack, unseen[i]);
        }

        /* Paired evaluation: test all top candidate replies against the EXACT SAME opponent rack */
        for (int k = 0; k < top_k; k++) {
            uint32_t c_idx = root->edges[k].target_node_idx;
            if (c_idx > 0 && c_idx < lat->num_nodes) {
                float delta = lattice_sample_adversarial_counter(lat, c_idx, dawg, &sim_opp_rack,
                                                                 cfg.use_palimpsest ? &palimpsest : NULL,
                                                                 bag_remaining);
                LatticeNode *leaf = &lat->nodes[c_idx];
                for (int p = 0; p < leaf->num_parents; p++) {
                    dirty_queue_push(&dq, leaf->parent_indices[p], fmaxf(1.0f, delta * 2.0f));
                }
                root->edges[k].visits++;
            }
        }

        /* Relax the lattice by prioritized backward induction */
        total_dirty_passes += settle_lattice(lat, &dq,
                                            cfg.use_palimpsest ? &palimpsest : NULL,
                                            128);

        /* The Root Court deliberates: P(argmax is best) */
        verdict = court_deliberate(lat, root_idx);
    }

    /* Final root settling pass */
    settle_node_backup(lat, root_idx, &palimpsest);
    verdict = court_deliberate(lat, root_idx);

    /* 4. Extract candidates into all_moves */
    if (all_moves) {
        all_moves->count = root->num_edges;
        for (int i = 0; i < root->num_edges; i++) {
            all_moves->moves[i] = root->edges[i].move;
            all_moves->moves[i].leave_equity = root->edges[i].q_value;
            all_moves->moves[i].sim_equity = root->edges[i].q_value;
        }
        eval_sort_moves(all_moves);
    }

    Move best_move = root->edges[verdict.best_edge_idx].move;
    best_move.leave_equity = root->edges[verdict.best_edge_idx].q_value;
    best_move.sim_equity = root->edges[verdict.best_edge_idx].q_value;

    clock_t end_time = clock();
    double elapsed_ms = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000.0;

    if (metrics) {
        metrics->nodes_in_lattice = lat->num_nodes;
        metrics->transpositions_merged = lat->total_transpositions_merged;
        metrics->dirty_passes_settled = total_dirty_passes;
        metrics->frontier_leaves_bought = broker.leaves_purchased;
        metrics->court_confidence = verdict.confidence;
        metrics->palimpsest_max_bias = palimpsest.max_absolute_correction;
        metrics->deliberation_ms = elapsed_ms;
    }

    lattice_free(lat);
    return best_move;
}

Move sim_find_best_move(const Board *board, const Dawg *dawg, const Rack *rack,
                        const TileBag *bag, void *unused_config,
                        MoveList *all_moves) {
    (void)unused_config;
    EngineConfig cfg;
    engine_config_default(&cfg);
    EngineMetrics metrics;
    return lattice_search(board, dawg, rack, bag, &cfg, all_moves, &metrics);
}
