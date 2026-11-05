#include "lattice.h"
#include "movegen.h"
#include "eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INITIAL_PARENT_CAP 4

Lattice* lattice_create(uint32_t capacity) {
    if (capacity < 1024) capacity = LATTICE_DEFAULT_CAPACITY;

    Lattice *lat = (Lattice*)calloc(1, sizeof(Lattice));
    lat->capacity = capacity;
    lat->nodes = (LatticeNode*)calloc(capacity, sizeof(LatticeNode));
    lat->num_nodes = 1; /* Reserve index 0 as null/sentinel */
    lat->root_index = 0;
    lat->total_transpositions_merged = 0;
    return lat;
}

void lattice_clear(Lattice *lat) {
    if (!lat) return;
    for (uint32_t i = 1; i < lat->num_nodes; i++) {
        if (lat->nodes[i].edges) {
            free(lat->nodes[i].edges);
            lat->nodes[i].edges = NULL;
        }
        if (lat->nodes[i].parent_indices) {
            free(lat->nodes[i].parent_indices);
            lat->nodes[i].parent_indices = NULL;
        }
    }
    memset(lat->hash_buckets, 0, sizeof(lat->hash_buckets));
    lat->num_nodes = 1;
    lat->root_index = 0;
    lat->total_transpositions_merged = 0;
}

void lattice_free(Lattice *lat) {
    if (!lat) return;
    lattice_clear(lat);
    if (lat->nodes) free(lat->nodes);
    free(lat);
}

static bool boards_equal(const Board *a, const Board *b) {
    return memcmp(a->cells, b->cells, sizeof(a->cells)) == 0;
}

uint32_t lattice_find_or_insert(Lattice *lat, const Board *board, int turn, bool *created) {
    uint64_t key = zobrist_compute_board(board, turn);
    uint32_t bucket = (uint32_t)(key & (LATTICE_HASH_SIZE - 1));

    /* Probe hash table */
    for (uint32_t probe = 0; probe < LATTICE_HASH_SIZE; probe++) {
        uint32_t idx = (bucket + probe) & (LATTICE_HASH_SIZE - 1);
        uint32_t node_idx = lat->hash_buckets[idx];

        if (node_idx == 0) {
            /* Empty slot: insert new node */
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
            n->belief.mean = 0.0f;
            n->belief.variance = 36.0f; /* Prior standard deviation ~6.0 pts */
            n->belief.samples = 0;
            n->belief.is_proven = false;
            n->belief.proof_sign = 0;

            n->raw_prior_mean = 0.0f;
            n->raw_prior_var = 36.0f;

            lat->hash_buckets[idx] = new_idx;
            if (created) *created = true;
            return new_idx;
        }

        /* Check if match */
        if (lat->nodes[node_idx].zobrist_key == key &&
            lat->nodes[node_idx].turn == turn &&
            boards_equal(&lat->nodes[node_idx].board, board)) {
            /* Transposition hit! Merged by construction */
            lat->total_transpositions_merged++;
            if (created) *created = false;
            return node_idx;
        }
    }

    if (created) *created = false;
    return 0;
}

void lattice_add_parent(Lattice *lat, uint32_t child_idx, uint32_t parent_idx) {
    if (child_idx == 0 || child_idx >= lat->num_nodes) return;
    if (parent_idx == 0 || parent_idx >= lat->num_nodes) return;

    LatticeNode *child = &lat->nodes[child_idx];

    /* Check if already present */
    for (int i = 0; i < child->num_parents; i++) {
        if (child->parent_indices[i] == parent_idx) return;
    }

    if (child->num_parents >= child->parent_cap) {
        uint16_t new_cap = child->parent_cap == 0 ? INITIAL_PARENT_CAP : child->parent_cap * 2;
        uint32_t *new_arr = (uint32_t*)realloc(child->parent_indices, sizeof(uint32_t) * new_cap);
        if (!new_arr) return;
        child->parent_indices = new_arr;
        child->parent_cap = new_cap;
    }

    child->parent_indices[child->num_parents++] = parent_idx;
}

void lattice_expand_node(Lattice *lat, uint32_t node_idx, const Dawg *dawg,
                         const Rack *rack, const Palimpsest *palimpsest) {
    if (node_idx == 0 || node_idx >= lat->num_nodes) return;
    LatticeNode *node = &lat->nodes[node_idx];
    if (node->is_expanded) return;

    /* Classify Palimpsest context */
    node->context_id = palimpsest_classify(rack, &node->board);
    float palimpsest_bias = palimpsest ? palimpsest_get_correction(palimpsest, node->context_id) : 0.0f;

    /* Generate all legal moves */
    MoveList all_moves;
    movegen_generate_all(&node->board, dawg, rack, &all_moves);

    if (all_moves.count == 0) {
        node->is_expanded = true;
        node->num_edges = 0;
        node->belief.mean = -50.0f; /* Stymied penalty */
        node->belief.variance = 4.0f;
        node->belief.samples = 1;
        return;
    }

    /* Compute static equity for candidate moves */
    for (int i = 0; i < all_moves.count; i++) {
        float base_eq = eval_move_static(&node->board, &all_moves.moves[i]);
        all_moves.moves[i].leave_equity = base_eq + palimpsest_bias;
    }
    eval_sort_moves(&all_moves);

    size_t num_candidates = (size_t)all_moves.count;
    if (num_candidates > MAX_LATTICE_EDGES) {
        num_candidates = MAX_LATTICE_EDGES;
    }
    node->edges = (LatticeEdge*)calloc(num_candidates, sizeof(LatticeEdge));
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
        edge->visits = 1;

        if (edge->q_value > best_q) best_q = edge->q_value;

        /* Create resulting speculative child board */
        Board spec_board;
        board_copy(&spec_board, &node->board);
        board_apply_move(&spec_board, &edge->move);

        bool created = false;
        uint32_t child_idx = lattice_find_or_insert(lat, &spec_board, 1 - node->turn, &created);
        edge->target_node_idx = child_idx;

        if (child_idx > 0) {
            lattice_add_parent(lat, child_idx, node_idx);
        }
    }

    node->raw_prior_mean = best_q;
    node->belief.mean = best_q;
    node->belief.variance = 20.0f;
    node->belief.samples = 1;
    node->is_expanded = true;
}
