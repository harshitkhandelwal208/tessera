#ifndef LATTICE_H
#define LATTICE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>
#include "types.h"
#include "board.h"
#include "dict.h"
#include "zobrist.h"
#include "palimpsest.h"

#define LATTICE_DEFAULT_CAPACITY 32768
#define TT_CLUSTER_SIZE          4
#define TT_NUM_CLUSTERS          16384 /* 16,384 clusters * 64 bytes = 1 MB cacheline-aligned TT */
#define MAX_LATTICE_EDGES        24    /* Pruned candidate beam per node */
#define MAX_LATTICE_PARENTS      8     /* Embedded parent capacity for zero allocations */

/*
 * 16-byte Transposition Table entry with Zobrist XOR verification (Stockfish-style)
 */
typedef struct {
    uint64_t key_xor_data; /* Zobrist key XOR'd with packed metadata */
    uint32_t node_idx;     /* 1-based node index into lat->nodes (0 = empty) */
    uint16_t depth;        /* Subtree depth / sample count */
    uint8_t  generation;   /* Replacement epoch */
    uint8_t  flags;        /* Bound/proof flags */
} TTEntry;

/*
 * Exactly 64-byte cluster (1 CPU cacheline) containing 4 TT entries
 */
typedef struct {
    alignas(64) TTEntry entries[TT_CLUSTER_SIZE];
} TTCluster;

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
    int bag_count;            /* Remaining tiles in bag */
    Belief belief;
    float raw_prior_mean;
    float raw_prior_var;
    float defensive_delta;    /* Paired relative counterplay advantage (points) */
    int context_id;           /* Palimpsest structural context */

    LatticeEdge *edges;
    uint16_t num_edges;
    bool is_expanded;

    /* Embedded parent back-pointers for zero-allocation prioritized relaxation */
    uint32_t parent_indices[MAX_LATTICE_PARENTS];
    uint16_t num_parents;

    uint32_t settle_generation;
} LatticeNode;

typedef struct {
    LatticeNode *nodes;
    uint32_t num_nodes;
    uint32_t capacity;

    /* Pre-allocated edge pool for zero-allocation node expansion */
    LatticeEdge *edge_pool;
    uint32_t num_edges_allocated;
    uint32_t edge_pool_capacity;

    TTCluster *clusters;
    uint8_t   search_generation;
    uint32_t root_index;
    uint32_t total_transpositions_merged;
    int score_diff; /* Live score margin for game-theoretic modulation */
} Lattice;

Lattice* lattice_create(uint32_t capacity);
void     lattice_free(Lattice *lat);
void     lattice_clear(Lattice *lat);

/* Lookup or register position in lattice */
uint32_t lattice_find_or_insert(Lattice *lat, const Board *board, int turn, bool *created);
uint32_t lattice_find_or_insert_with_key(Lattice *lat, const Board *board, uint64_t key, int turn, bool *created);

/* Add parent back-pointer for backward settling */
void     lattice_add_parent(Lattice *lat, uint32_t child_idx, uint32_t parent_idx);

/* Expand a frontier node: generates candidate plays and creates child edges */
void     lattice_expand_node(Lattice *lat, uint32_t node_idx, const Dawg *dawg,
                             const Rack *rack, const Palimpsest *palimpsest, int bag_count);

#endif /* LATTICE_H */
