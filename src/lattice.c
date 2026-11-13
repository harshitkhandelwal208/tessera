#include "lattice.h"
#include "settle.h"
#include "movegen.h"
#include "eval.h"
#include "simulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline uint64_t tt_pack_data(uint32_t node_idx, uint16_t depth, uint8_t generation, uint8_t flags) {
    return ((uint64_t)node_idx) |
           (((uint64_t)depth) << 32) |
           (((uint64_t)generation) << 48) |
           (((uint64_t)flags) << 56);
}

Lattice* lattice_create(uint32_t capacity) {
    if (capacity < 1024) capacity = LATTICE_DEFAULT_CAPACITY;

    Lattice *lat = (Lattice*)calloc(1, sizeof(Lattice));
    lat->capacity = capacity;
    lat->nodes = (LatticeNode*)calloc(capacity, sizeof(LatticeNode));
    lat->edge_pool_capacity = capacity * MAX_LATTICE_EDGES;
    lat->edge_pool = (LatticeEdge*)calloc(lat->edge_pool_capacity, sizeof(LatticeEdge));
    lat->clusters = (TTCluster*)aligned_alloc(64, TT_NUM_CLUSTERS * sizeof(TTCluster));
    memset(lat->clusters, 0, TT_NUM_CLUSTERS * sizeof(TTCluster));
    lat->search_generation = 1;
    lat->num_edges_allocated = 0;
    lat->num_nodes = 1; /* Reserve index 0 as null/sentinel */
    lat->root_index = 0;
    lat->total_transpositions_merged = 0;
    return lat;
}

void lattice_clear(Lattice *lat) {
    if (!lat) return;
    memset(lat->clusters, 0, TT_NUM_CLUSTERS * sizeof(TTCluster));
    lat->search_generation++;
    if (lat->search_generation == 0) lat->search_generation = 1;
    lat->num_nodes = 1;
    lat->num_edges_allocated = 0;
    lat->root_index = 0;
    lat->total_transpositions_merged = 0;
}

void lattice_free(Lattice *lat) {
    if (!lat) return;
    if (lat->nodes) free(lat->nodes);
    if (lat->edge_pool) free(lat->edge_pool);
    if (lat->clusters) free(lat->clusters);
    free(lat);
}

static bool boards_equal(const Board *a, const Board *b) {
    if (a->occupied.b[0] != b->occupied.b[0] ||
        a->occupied.b[1] != b->occupied.b[1] ||
        a->occupied.b[2] != b->occupied.b[2] ||
        a->occupied.b[3] != b->occupied.b[3]) {
        return false;
    }
    return memcmp(a->cells, b->cells, sizeof(a->cells)) == 0;
}

uint32_t lattice_find_or_insert_with_key(Lattice *lat, const Board *board, uint64_t key, int turn, bool *created) {
    uint32_t cluster_idx = (uint32_t)(key & (TT_NUM_CLUSTERS - 1));

    /* Probe 64-byte primary cluster */
    TTCluster *cl = &lat->clusters[cluster_idx];
    TTEntry *best_victim = NULL;
    int lowest_priority = 9999999;

    for (int i = 0; i < TT_CLUSTER_SIZE; i++) {
        TTEntry *e = &cl->entries[i];
        if (e->node_idx == 0) {
            if (!best_victim) best_victim = e;
            continue;
        }

        uint64_t data = tt_pack_data(e->node_idx, e->depth, e->generation, e->flags);
        if ((e->key_xor_data ^ data) == key) {
            uint32_t node_idx = e->node_idx;
            if (node_idx < lat->num_nodes) {
                LatticeNode *n = &lat->nodes[node_idx];
                if (n->turn == turn && boards_equal(&n->board, board)) {
                    /* Transposition hit! Merged by construction */
                    lat->total_transpositions_merged++;
                    e->generation = lat->search_generation;
                    e->key_xor_data = key ^ tt_pack_data(e->node_idx, e->depth, e->generation, e->flags);
                    if (created) *created = false;
                    return node_idx;
                }
            }
        }

        int priority = (e->generation == lat->search_generation ? 1000 : 0) + e->depth;
        if (priority < lowest_priority) {
            lowest_priority = priority;
            best_victim = e;
        }
    }

    /* Secondary probe: check adjacent cluster if primary cluster was completely full */
    if (!best_victim || best_victim->node_idx != 0) {
        TTCluster *cl2 = &lat->clusters[(cluster_idx + 1) & (TT_NUM_CLUSTERS - 1)];
        for (int i = 0; i < TT_CLUSTER_SIZE; i++) {
            TTEntry *e = &cl2->entries[i];
            if (e->node_idx == 0) {
                best_victim = e;
                break;
            }
            uint64_t data = tt_pack_data(e->node_idx, e->depth, e->generation, e->flags);
            if ((e->key_xor_data ^ data) == key) {
                uint32_t node_idx = e->node_idx;
                if (node_idx < lat->num_nodes) {
                    LatticeNode *n = &lat->nodes[node_idx];
                    if (n->turn == turn && boards_equal(&n->board, board)) {
                        lat->total_transpositions_merged++;
                        e->generation = lat->search_generation;
                        e->key_xor_data = key ^ tt_pack_data(e->node_idx, e->depth, e->generation, e->flags);
                        if (created) *created = false;
                        return node_idx;
                    }
                }
            }
            int priority = (e->generation == lat->search_generation ? 1000 : 0) + e->depth;
            if (priority < lowest_priority) {
                lowest_priority = priority;
                best_victim = e;
            }
        }
    }

    /* Insert new node into Lattice */
    if (lat->num_nodes >= lat->capacity) {
        if (created) *created = false;
        return 0; /* Lattice full */
    }

    uint32_t new_idx = lat->num_nodes++;
    LatticeNode *n = &lat->nodes[new_idx];
    memset(n, 0, sizeof(LatticeNode));
    n->zobrist_key = key;
    board_copy(&n->board, board);
    n->turn = turn;

    /* Default initial belief prior */
    n->belief.mean = (turn == 1) ? BASELINE_OPP_SCORE : 0.0f;
    n->belief.variance = 36.0f;
    n->belief.samples = 0;
    n->belief.is_proven = false;
    n->belief.proof_sign = 0;
    n->raw_prior_mean = n->belief.mean;
    n->raw_prior_var = 36.0f;

    if (best_victim) {
        best_victim->node_idx = new_idx;
        best_victim->depth = 0;
        best_victim->generation = lat->search_generation;
        best_victim->flags = 0;
        best_victim->key_xor_data = key ^ tt_pack_data(new_idx, 0, lat->search_generation, 0);
    }

    if (created) *created = true;
    return new_idx;
}

uint32_t lattice_find_or_insert(Lattice *lat, const Board *board, int turn, bool *created) {
    uint64_t key = zobrist_compute_board(board, turn);
    return lattice_find_or_insert_with_key(lat, board, key, turn, created);
}

void lattice_add_parent(Lattice *lat, uint32_t child_idx, uint32_t parent_idx) {
    if (child_idx == 0 || child_idx >= lat->num_nodes) return;
    if (parent_idx == 0 || parent_idx >= lat->num_nodes) return;

    LatticeNode *child = &lat->nodes[child_idx];

    /* Check if already present */
    for (int i = 0; i < child->num_parents; i++) {
        if (child->parent_indices[i] == parent_idx) return;
    }

    if (child->num_parents < MAX_LATTICE_PARENTS) {
        child->parent_indices[child->num_parents++] = parent_idx;
    }
}

void lattice_expand_node(Lattice *lat, uint32_t node_idx, const Dawg *dawg,
                         const Rack *rack, const Palimpsest *palimpsest, int bag_count) {
    if (node_idx == 0 || node_idx >= lat->num_nodes) return;
    LatticeNode *node = &lat->nodes[node_idx];
    if (node->is_expanded) return;

    /* Classify Palimpsest context */
    node->bag_count = bag_count;
    node->context_id = palimpsest_classify(rack, &node->board);
    float palimpsest_bias = palimpsest ? palimpsest_get_correction(palimpsest, node->context_id) : 0.0f;

    /* Generate all legal moves */
    MoveList all_moves = {0};
    movegen_generate_all(&node->board, dawg, rack, &all_moves);

    if (all_moves.count == 0) {
        move_list_destroy(&all_moves);
        node->is_expanded = true;
        node->num_edges = 0;
        node->belief.is_proven = true;
        node->belief.proof_sign = -1; /* Proven lost: no moves possible */
        node->belief.mean = -50.0f;   /* Stymied penalty */
        node->belief.variance = 0.0f;
        node->belief.samples = 1;
        return;
    }

    /* In endgame (bag is exhausted), compute exact opponent unplayed tile value */
    int opp_rack_val = 0;
    if (bag_count == 0) {
        char unseen[100];
        int unseen_count = sim_get_unseen_tiles(&node->board, rack, unseen);
        for (int i = 0; i < unseen_count; i++) {
            if (unseen[i] >= 'A' && unseen[i] <= 'Z') {
                opp_rack_val += TILE_VALUES[unseen[i] - 'A'];
            }
        }
    }

    /* Compute static equity for candidate moves with Stockfish HalfKP-style incremental accumulator */
    NNUEAccumulator base_acc;
    bool has_base_acc = false;
    if (g_nnue.is_loaded) {
        nnue_accumulator_build_base(&g_nnue, &node->board, rack, bag_count, lat->score_diff, &base_acc);
        has_base_acc = true;
    }

    for (int i = 0; i < all_moves.count; i++) {
        if (bag_count == 0 && all_moves.moves[i].num_placed == rack->count) {
            /* Outplay game-out: banked score + 2 * opponent rack value */
            all_moves.moves[i].leave_equity = (float)all_moves.moves[i].score + 2.0f * (float)opp_rack_val;
        } else {
            float base_eq;
            if (has_base_acc) {
                base_eq = eval_move_static_incremental(&node->board, &all_moves.moves[i], &base_acc, bag_count, lat->score_diff);
            } else {
                base_eq = eval_move_static_with_score(&node->board, &all_moves.moves[i], bag_count, lat->score_diff);
            }
            all_moves.moves[i].leave_equity = base_eq + palimpsest_bias;
        }
    }
    eval_sort_moves(&all_moves);

    size_t num_candidates = (size_t)all_moves.count;
    if (num_candidates > MAX_LATTICE_EDGES) {
        num_candidates = MAX_LATTICE_EDGES;
    }
    if (lat->num_edges_allocated + num_candidates > lat->edge_pool_capacity) {
        num_candidates = (lat->edge_pool_capacity > lat->num_edges_allocated) ?
                         (lat->edge_pool_capacity - lat->num_edges_allocated) : 0;
    }
    node->edges = &lat->edge_pool[lat->num_edges_allocated];
    lat->num_edges_allocated += (uint32_t)num_candidates;
    memset(node->edges, 0, num_candidates * sizeof(LatticeEdge));
    node->num_edges = (uint16_t)num_candidates;

    /* Softmax priors over candidate static equities */
    float max_eq = all_moves.moves[0].leave_equity;
    float sum_exp = 0.0f;
    float exps[MAX_LATTICE_EDGES];
    for (size_t i = 0; i < num_candidates; i++) {
        exps[i] = expf((all_moves.moves[i].leave_equity - max_eq) / 4.0f);
        sum_exp += exps[i];
    }

    float best_q = -10000.0f;

    for (size_t i = 0; i < num_candidates; i++) {
        LatticeEdge *edge = &node->edges[i];
        edge->move = all_moves.moves[i];
        edge->prior_prob = exps[i] / (sum_exp > 0.0f ? sum_exp : 1.0f);
        edge->q_value = all_moves.moves[i].leave_equity;
        edge->variance = 25.0f;
        edge->visits = 0;

        if (edge->q_value > best_q) best_q = edge->q_value;

        /* Create resulting speculative child board */
        Board spec_board;
        board_copy(&spec_board, &node->board);
        board_apply_move(&spec_board, &edge->move);

        uint64_t child_key = zobrist_toggle_turn(zobrist_update_move(node->zobrist_key, &edge->move));
        bool created = false;
        uint32_t child_idx = lattice_find_or_insert_with_key(lat, &spec_board, child_key, 1 - node->turn, &created);
        edge->target_node_idx = child_idx;

        if (child_idx > 0) {
            lattice_add_parent(lat, child_idx, node_idx);
            lat->nodes[child_idx].bag_count = (bag_count >= edge->move.num_placed) ? bag_count - edge->move.num_placed : 0;

            /* If bag is exhausted and this move empties the rack, pin child as terminal proof */
            if (bag_count == 0 && edge->move.num_placed == rack->count) {
                LatticeNode *child = &lat->nodes[child_idx];
                child->is_expanded = true;
                child->num_edges = 0;
                child->belief.is_proven = true;
                child->belief.proof_sign = -1; /* Opponent is proven outplayed */
                child->belief.variance = 0.0f;
                child->belief.mean = 2.0f * (float)opp_rack_val; /* Exact terminal bonus */
            }
        }
    }

    move_list_destroy(&all_moves);

    node->raw_prior_mean = best_q;
    node->belief.mean = best_q;
    node->belief.variance = 20.0f;
    node->belief.samples = 1;
    node->is_expanded = true;
}
