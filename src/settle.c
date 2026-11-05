#include "settle.h"
#include "eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void dirty_queue_init(DirtyQueue *dq) {
    dq->count = 0;
}

bool dirty_queue_is_empty(const DirtyQueue *dq) {
    return dq->count == 0;
}

static void heap_swap(DirtyQueueEntry *a, DirtyQueueEntry *b) {
    DirtyQueueEntry tmp = *a;
    *a = *b;
    *b = tmp;
}

bool dirty_queue_push(DirtyQueue *dq, uint32_t node_idx, float priority) {
    if (dq->count >= DIRTY_QUEUE_CAPACITY) return false;

    uint32_t i = dq->count++;
    dq->entries[i].node_idx = node_idx;
    dq->entries[i].priority = priority;

    /* Bubble up */
    while (i > 0) {
        uint32_t parent = (i - 1) / 2;
        if (dq->entries[i].priority > dq->entries[parent].priority) {
            heap_swap(&dq->entries[i], &dq->entries[parent]);
            i = parent;
        } else {
            break;
        }
    }
    return true;
}

bool dirty_queue_pop(DirtyQueue *dq, uint32_t *out_node_idx, float *out_priority) {
    if (dq->count == 0) return false;

    if (out_node_idx) *out_node_idx = dq->entries[0].node_idx;
    if (out_priority) *out_priority = dq->entries[0].priority;

    dq->entries[0] = dq->entries[--dq->count];

    /* Bubble down */
    uint32_t i = 0;
    while (1) {
        uint32_t left = 2 * i + 1;
        uint32_t right = 2 * i + 2;
        uint32_t largest = i;

        if (left < dq->count && dq->entries[left].priority > dq->entries[largest].priority) {
            largest = left;
        }
        if (right < dq->count && dq->entries[right].priority > dq->entries[largest].priority) {
            largest = right;
        }

        if (largest != i) {
            heap_swap(&dq->entries[i], &dq->entries[largest]);
            i = largest;
        } else {
            break;
        }
    }

    return true;
}

float settle_node_backup(Lattice *lat, uint32_t node_idx, Palimpsest *palimpsest) {
    if (node_idx == 0 || node_idx >= lat->num_nodes) return 0.0f;
    LatticeNode *n = &lat->nodes[node_idx];
    if (!n->is_expanded || n->num_edges == 0) return 0.0f;

    float old_mean = n->belief.mean;
    float best_q = -100000.0f;
    float best_var = 100.0f;
    bool has_proven_win = false;

    for (int i = 0; i < n->num_edges; i++) {
        LatticeEdge *edge = &n->edges[i];
        float child_mean = 0.0f;
        float child_var = 25.0f;
        bool child_is_loss = false;

        if (edge->target_node_idx > 0 && edge->target_node_idx < lat->num_nodes) {
            LatticeNode *child = &lat->nodes[edge->target_node_idx];
            child_mean = child->belief.mean;
            child_var = child->belief.variance;

            if (child->belief.is_proven && child->belief.proof_sign < 0) {
                /* Opponent is proven lost -> this move proves parent won! */
                has_proven_win = true;
                child_is_loss = true;
                edge->variance = 0.0f;
            }
        }

        /*
         * Calibrated Bellman Operator on Relative Spread:
         * Q(e) = MoveStatic - gamma * (ChildOpponentMean - BaselineOpponentScore)
         */
        DualAccumulator acc = eval_move_accumulate(&n->board, &edge->move, 50);
        float move_static = acc.net_equity;

        if (child_is_loss) {
            edge->q_value = move_static + 50.0f;
            edge->variance = 0.0f;
        } else {
            float opp_excess = child_mean - BASELINE_OPP_SCORE;
            edge->q_value = move_static - (SETTLE_GAMMA * opp_excess);
            edge->variance = (SETTLE_GAMMA * SETTLE_GAMMA * child_var) + 2.0f;
        }

        if (edge->q_value > best_q) {
            best_q = edge->q_value;
            best_var = edge->variance;
        }
    }

    if (has_proven_win) {
        n->belief.is_proven = true;
        n->belief.proof_sign = +1;
        n->belief.variance = 0.0f;
    } else {
        /* Check if all children are proven winning replies for the opponent */
        bool all_children_losses = true;
        for (int i = 0; i < n->num_edges; i++) {
            uint32_t tid = n->edges[i].target_node_idx;
            if (tid == 0 || !lat->nodes[tid].belief.is_proven || lat->nodes[tid].belief.proof_sign <= 0) {
                all_children_losses = false;
                break;
            }
        }
        if (all_children_losses && n->num_edges > 0) {
            n->belief.is_proven = true;
            n->belief.proof_sign = -1;
            n->belief.variance = 0.0f;
        } else {
            n->belief.variance = best_var * 0.8f;
        }
    }

    n->belief.mean = best_q;
    n->belief.samples++;

    float delta = fabsf(n->belief.mean - old_mean);

    /* Record disagreement in Palimpsest if meaningful */
    if (palimpsest && delta > SETTLE_EPS) {
        float disagreement = n->belief.mean - n->raw_prior_mean;
        palimpsest_record_disagreement(palimpsest, n->context_id, disagreement);
    }

    return delta;
}

uint32_t settle_lattice(Lattice *lat, DirtyQueue *dq, Palimpsest *palimpsest, uint32_t max_passes) {
    uint32_t passes = 0;

    while (!dirty_queue_is_empty(dq) && passes < max_passes) {
        uint32_t node_idx = 0;
        float priority = 0.0f;
        if (!dirty_queue_pop(dq, &node_idx, &priority)) break;

        passes++;
        float delta = settle_node_backup(lat, node_idx, palimpsest);

        /* If this node's belief moved significantly, enqueue all its parents! */
        if (delta > SETTLE_EPS) {
            LatticeNode *n = &lat->nodes[node_idx];
            for (int p = 0; p < n->num_parents; p++) {
                uint32_t parent_idx = n->parent_indices[p];
                dirty_queue_push(dq, parent_idx, delta);
            }
        }
    }

    return passes;
}
