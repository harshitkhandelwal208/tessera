#ifndef LATTICE_H
#define LATTICE_H

#include <stdint.h>
#include <stdbool.h>
#include "types.h"
#include "board.h"
#include "dict.h"
#include "zobrist.h"
#include "palimpsest.h"

#define LATTICE_DEFAULT_CAPACITY 32768
#define LATTICE_HASH_SIZE        65536 /* Power of 2 */
#define MAX_LATTICE_EDGES        24    /* Pruned candidate beam per node */

/* Distributional belief object: Gaussian belief + variance + proof certificate */
typedef struct {
    float mean;         /* Expected net score / equity (points) */
    float variance;     /* Uncertainty variance (sigma^2) */
    uint32_t samples;   /* Observation count */
    bool is_proven;     /* True if mathematically proven (variance = 0) */
    int8_t proof_sign;  /* +1 = proven win/dominance, -1 = proven refutation */
} Belief;

typedef struct {
    Move move;
    uint32_t target_node_idx; /* Target board state in lattice */
    float prior_prob;         /* Prior probability */
    float q_value;            /* Settled action value */
    float variance;           /* Edge action variance */
    uint32_t visits;          /* Edge purchase count */
} LatticeEdge;

typedef struct {
    uint64_t zobrist_key;
    Board board;
    int turn;                 /* 0 = AI, 1 = Opponent */
    Belief belief;
    float raw_prior_mean;
    float raw_prior_var;
    int context_id;           /* Palimpsest structural context */

    LatticeEdge *edges;
    uint16_t num_edges;
    bool is_expanded;

    /* Parent back-pointers for prioritized dirty relaxation */
    uint32_t *parent_indices;
    uint16_t num_parents;
    uint16_t parent_cap;

    uint32_t settle_generation;
} LatticeNode;

typedef struct {
    LatticeNode *nodes;
    uint32_t num_nodes;
    uint32_t capacity;

    uint32_t hash_buckets[LATTICE_HASH_SIZE]; /* Key -> node index (1-based, 0 = empty) */
    uint32_t root_index;
    uint32_t total_transpositions_merged;
} Lattice;

Lattice* lattice_create(uint32_t capacity);
void     lattice_free(Lattice *lat);
void     lattice_clear(Lattice *lat);

/* Lookup or register position in lattice */
uint32_t lattice_find_or_insert(Lattice *lat, const Board *board, int turn, bool *created);

/* Add parent back-pointer for backward settling */
void     lattice_add_parent(Lattice *lat, uint32_t child_idx, uint32_t parent_idx);

/* Expand a frontier node: generates candidate plays and creates child edges */
void     lattice_expand_node(Lattice *lat, uint32_t node_idx, const Dawg *dawg,
                             const Rack *rack, const Palimpsest *palimpsest, int bag_count);

/* Sample an adversarial opponent counter on a child node using Welford's algorithm */
float    lattice_sample_adversarial_counter(Lattice *lat, uint32_t node_idx, const Dawg *dawg,
                                            const Rack *opp_rack, const Palimpsest *palimpsest,
                                            int bag_count);

#endif /* LATTICE_H */
