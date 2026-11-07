#ifndef SETTLE_H
#define SETTLE_H

#include <stdint.h>
#include <stdbool.h>
#include "lattice.h"
#include "palimpsest.h"

#define SETTLE_EPS 0.05f     /* Minimum belief change to enqueue parents */
#define SETTLE_GAMMA 0.35f   /* Calibrated defensive counterplay discount factor */
#define BASELINE_OPP_SCORE 26.0f /* Expected average opponent counterplay equity */
#define DIRTY_QUEUE_CAPACITY 8192

typedef struct {
    uint32_t node_idx;
    float priority; /* |Delta Mean| */
} DirtyQueueEntry;

typedef struct {
    DirtyQueueEntry entries[DIRTY_QUEUE_CAPACITY];
    uint32_t count;
} DirtyQueue;

void dirty_queue_init(DirtyQueue *dq);
bool dirty_queue_push(DirtyQueue *dq, uint32_t node_idx, float priority);
bool dirty_queue_push_unique(DirtyQueue *dq, Lattice *lat, uint32_t node_idx, float priority, uint32_t gen);
bool dirty_queue_pop(DirtyQueue *dq, uint32_t *out_node_idx, float *out_priority);
bool dirty_queue_is_empty(const DirtyQueue *dq);

/*
 * Settling Engine:
 * Relaxes beliefs across the lattice until fixed-point equilibrium or budget reached.
 * Returns total number of relaxation steps executed.
 */
uint32_t settle_lattice(Lattice *lat, DirtyQueue *dq, Palimpsest *palimpsest, uint32_t max_passes);

/* Backs up a single node from its children */
float settle_node_backup(Lattice *lat, uint32_t node_idx, Palimpsest *palimpsest);

#endif /* SETTLE_H */
