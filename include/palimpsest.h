#ifndef PALIMPSEST_H
#define PALIMPSEST_H

#include <stdint.h>
#include <stdbool.h>
#include "types.h"
#include "board.h"

#define PALIMPSEST_NUM_CONTEXTS 128

/* Conjugate Normal posterior over evaluation correction for a structural context */
typedef struct {
    float mean;         /* Posterior mean correction (points) */
    float precision;    /* 1 / variance (certainty) */
    uint32_t updates;   /* Observation count */
} ContextPosterior;

typedef struct {
    ContextPosterior contexts[PALIMPSEST_NUM_CONTEXTS];
    uint32_t total_disagreements;
    float max_absolute_correction;
} Palimpsest;

void  palimpsest_init(Palimpsest *p);
void  palimpsest_reset(Palimpsest *p);

/* Determine structural context ID from rack and board properties */
int   palimpsest_classify(const Rack *rack, const Board *board);

/* Get current posterior correction for this context */
float palimpsest_get_correction(const Palimpsest *p, int context_id);

/* Closed-form Bayesian update on settled-vs-raw disagreement */
void  palimpsest_record_disagreement(Palimpsest *p, int context_id, float disagreement);

#endif /* PALIMPSEST_H */
