#ifndef BROKER_H
#define BROKER_H

#include <stdint.h>
#include <stdbool.h>
#include "lattice.h"
#include "settle.h"
#include "palimpsest.h"

#define BROKER_C_PUCT 2.5f
#define BROKER_C_VAR  0.6f

typedef struct {
    float c_puct;
    float c_var;
    uint32_t leaves_purchased;
    uint32_t proven_bypasses;
} Broker;

void     broker_init(Broker *b);

/*
 * Selects the next frontier leaf to buy:
 * Descends from root along highest acquisition score (Q + PUCT + Uncertainty).
 * Returns node index of the selected unexpanded frontier leaf (or 0 if exhausted).
 */
uint32_t broker_select_frontier_leaf(Broker *b, const Lattice *lat, uint32_t root_idx);

#endif /* BROKER_H */
