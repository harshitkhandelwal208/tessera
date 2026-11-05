#ifndef COURT_H
#define COURT_H

#include <stdint.h>
#include <stdbool.h>
#include "lattice.h"

#define COURT_CONFIDENCE_THRESHOLD 0.98f
#define COURT_SAMPLES              1000

typedef struct {
    float confidence;          /* P(current argmax is truly best) [0.0 .. 1.0] */
    int best_edge_idx;         /* Current argmax move index */
    bool should_adjourn;       /* True if confidence clears threshold or proven */
    uint32_t deliberated_steps;
} CourtVerdict;

/*
 * Deliberates on the root node's edges using joint Gaussian posterior sampling.
 * Returns the CourtVerdict containing P(argmax is best) and whether to adjourn.
 */
CourtVerdict court_deliberate(const Lattice *lat, uint32_t root_idx);

#endif /* COURT_H */
