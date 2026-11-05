#ifndef BROKER_H
#define BROKER_H

#include <stdint.h>
#include <stdbool.h>
#include "lattice.h"

#define BROKER_C_PUCT 1.414f
#define BROKER_C_VAR  0.75f

typedef struct {
    float c_puct;
    float c_var;
    uint32_t leaves_purchased;
    uint32_t proven_bypasses;
} Broker;

void     broker_init(Broker *b);

/* Selects the highest priority frontier leaf in the lattice for expansion */
uint32_t broker_select_frontier_leaf(Broker *b, const Lattice *lat, uint32_t root_idx);

/* Selects top competing child for adversarial Monte Carlo sample refinement */
uint32_t broker_select_top_child(Broker *b, Lattice *lat, uint32_t root_idx);

#endif /* BROKER_H */
