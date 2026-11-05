#include "palimpsest.h"
#include <math.h>
#include <ctype.h>

#define PRIOR_PRECISION 0.15f /* Prior variance ~6.67 pts */
#define OBS_PRECISION   0.35f /* Precision of settling observations */

void palimpsest_init(Palimpsest *p) {
    palimpsest_reset(p);
}

void palimpsest_reset(Palimpsest *p) {
    for (int i = 0; i < PALIMPSEST_NUM_CONTEXTS; i++) {
        p->contexts[i].mean = 0.0f;
        p->contexts[i].precision = PRIOR_PRECISION;
        p->contexts[i].updates = 0;
    }
    p->total_disagreements = 0;
    p->max_absolute_correction = 0.0f;
}

static inline bool is_vowel_char(char c) {
    c = (char)toupper((unsigned char)c);
    return (c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U');
}

static inline bool is_heavy_char(char c) {
    c = (char)toupper((unsigned char)c);
    return (c == 'Q' || c == 'Z' || c == 'X' || c == 'J' || c == 'V' || c == 'W');
}

int palimpsest_classify(const Rack *rack, const Board *board) {
    /* 1. Vowel count (0..7) */
    int vowels = 0;
    int heavies = 0;
    for (int i = 0; i < rack->count; i++) {
        if (is_vowel_char(rack->tiles[i])) vowels++;
        if (is_heavy_char(rack->tiles[i])) heavies++;
    }
    if (vowels > 7) vowels = 7;
    if (heavies > 3) heavies = 3;

    /* 2. Board Openness (0..3) */
    int occupied_count = 0;
    int tw_open_count = 0;
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (board->cells[r][c].tile != 0) {
                occupied_count++;
            } else if (board_get_multiplier(r, c) == MULT_TW && board_is_anchor(board, r, c)) {
                tw_open_count++;
            }
        }
    }

    int openness = 0;
    if (occupied_count < 10) openness = 0;       /* Opening phase */
    else if (tw_open_count >= 2) openness = 3;   /* Wide open hotspots */
    else if (occupied_count > 35) openness = 2;  /* Dense cluster */
    else openness = 1;                           /* Normal midgame */

    /* Context ID: 0..127 */
    return (openness << 5) | (heavies << 3) | vowels;
}

float palimpsest_get_correction(const Palimpsest *p, int context_id) {
    if (context_id < 0 || context_id >= PALIMPSEST_NUM_CONTEXTS) return 0.0f;
    return p->contexts[context_id].mean;
}

void palimpsest_record_disagreement(Palimpsest *p, int context_id, float disagreement) {
    if (context_id < 0 || context_id >= PALIMPSEST_NUM_CONTEXTS) return;

    ContextPosterior *cp = &p->contexts[context_id];
    float new_prec = cp->precision + OBS_PRECISION;
    /* Closed form conjugate normal update: (tau1 * mu1 + tau2 * mu2) / (tau1 + tau2) */
    cp->mean = (cp->precision * cp->mean + OBS_PRECISION * disagreement) / new_prec;
    cp->precision = new_prec;
    cp->updates++;

    p->total_disagreements++;
    float abs_m = fabsf(cp->mean);
    if (abs_m > p->max_absolute_correction) {
        p->max_absolute_correction = abs_m;
    }
}
