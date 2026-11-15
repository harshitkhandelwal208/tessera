#include "eval.h"
#include "nnue.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

/* Phase names for telemetry and UI */
const char* eval_phase_name(PhaseBucket phase) {
    switch (phase) {
        case PHASE_BUCKET_OPENING:     return "Opening (80-100)";
        case PHASE_BUCKET_EARLY:       return "Early (60-79)";
        case PHASE_BUCKET_MIDGAME:     return "Midgame (40-59)";
        case PHASE_BUCKET_LATE_MID:    return "Late Midgame (20-39)";
        case PHASE_BUCKET_PRE_ENDGAME: return "Pre-Endgame (10-19)";
        case PHASE_BUCKET_EXHAUSTION:  return "Exhaustion (1-9)";
        case PHASE_BUCKET_ENDGAME:     return "Endgame (0)";
        case PHASE_BUCKET_PLAYOUT:     return "Playout";
        default:                       return "Midgame";
    }
}

const char* eval_openness_name(OpennessBucket openness) {
    switch (openness) {
        case OPENNESS_CLOSED:   return "Closed";
        case OPENNESS_STANDARD: return "Standard";
        case OPENNESS_EXPOSED:  return "Exposed";
        case OPENNESS_WIDE:     return "Wide Open";
        default:                return "Standard";
    }
}

/* 
 * Phase-specific single tile equities (Index 0..25 for A..Z, 26 for Blank '?')
 * Calibrated from grandmaster simulation playouts across 8 bag-depletion stages.
 */
static const float PHASE_TILE_EQUITY[8][27] = {
    /* PHASE 0 (PLAYOUT): Immediate scoring dominates */
    { 0.0f, -0.5f,  0.5f,  0.2f,  0.2f, -1.0f, -0.8f,  0.3f, -0.5f, -2.0f,
     -0.5f, -0.1f,  0.3f,  0.2f, -1.0f, -0.2f, -6.0f,  0.5f,  4.0f,  0.2f,
     -1.5f, -4.0f, -1.5f,  2.0f, -0.8f,  3.0f, 15.0f },
    /* PHASE 1 (ENDGAME) */
    { 0.1f, -0.8f,  0.8f,  0.3f,  0.3f, -1.2f, -1.0f,  0.4f, -0.8f, -2.5f,
     -0.8f, -0.1f,  0.4f,  0.3f, -1.5f, -0.3f, -7.0f,  0.7f,  5.0f,  0.3f,
     -2.0f, -4.5f, -1.8f,  2.5f, -1.0f,  3.5f, 18.0f },
    /* PHASE 2 (EXHAUSTION): Unplayable Q, V penalties escalate */
    { 0.1f, -1.0f,  1.0f,  0.4f,  0.4f, -1.5f, -1.2f,  0.5f, -1.0f, -3.0f,
     -1.0f, -0.2f,  0.5f,  0.4f, -2.0f, -0.4f, -8.0f,  0.8f,  6.0f,  0.3f,
     -2.5f, -5.0f, -2.0f,  3.0f, -1.2f,  3.8f, 20.0f },
    /* PHASE 3 (PRE-ENDGAME): Q and V toxicities peak */
    { 0.2f, -1.0f,  1.2f,  0.5f,  0.4f, -1.8f, -1.3f,  0.6f, -1.2f, -3.2f,
     -1.1f, -0.2f,  0.6f,  0.4f, -2.2f, -0.5f, -9.0f,  1.0f,  7.0f,  0.3f,
     -2.8f, -6.0f, -2.2f,  3.2f, -1.4f,  3.9f, 22.0f },
    /* PHASE 4 (LATE MIDGAME) */
    { 0.2f, -1.0f,  1.3f,  0.5f,  0.5f, -1.8f, -1.4f,  0.7f, -1.3f, -3.0f,
     -1.1f, -0.2f,  0.7f,  0.5f, -2.3f, -0.5f, -8.0f,  1.1f,  7.5f,  0.4f,
     -2.8f, -5.8f, -2.0f,  3.4f, -1.5f,  4.0f, 24.0f },
    /* PHASE 5 (MIDGAME) */
    { 0.2f, -1.0f,  1.5f,  0.6f,  0.5f, -2.0f, -1.5f,  0.8f, -1.5f, -3.0f,
     -1.2f, -0.2f,  0.8f,  0.5f, -2.5f, -0.5f, -7.5f,  1.2f,  8.0f,  0.4f,
     -3.0f, -5.5f, -2.0f,  3.5f, -1.5f,  4.0f, 26.5f },
    /* PHASE 6 (EARLY) */
    { 0.3f, -0.8f,  1.5f,  0.6f,  0.6f, -1.8f, -1.3f,  0.8f, -1.3f, -2.8f,
     -1.0f, -0.1f,  0.8f,  0.6f, -2.2f, -0.4f, -7.0f,  1.3f,  8.2f,  0.5f,
     -2.7f, -5.0f, -1.8f,  3.5f, -1.3f,  4.0f, 27.5f },
    /* PHASE 7 (OPENING): Blank and S are at maximum bingo value */
    { 0.3f, -0.7f,  1.6f,  0.7f,  0.6f, -1.6f, -1.2f,  0.9f, -1.2f, -2.5f,
     -0.9f, -0.1f,  0.9f,  0.6f, -2.0f, -0.3f, -6.5f,  1.4f,  8.5f,  0.5f,
     -2.5f, -4.5f, -1.6f,  3.6f, -1.2f,  4.2f, 28.5f }
};

/* 27x27 Pairwise Letter Synergy Matrix (Upper Triangular + Diagonal) */
static const float PAIRWISE_SYNERGY[27][27] = {
    /* QU synergy */
    ['Q' - 'A']['U' - 'A'] = 10.0f,
    ['U' - 'A']['Q' - 'A'] = 10.0f,

    /* Blank + S & Blank + Key Consonants */
    [26]['S' - 'A'] = 4.5f,
    ['S' - 'A'][26] = 4.5f,
    [26]['R' - 'A'] = 2.5f,
    ['R' - 'A'][26] = 2.5f,
    [26]['T' - 'A'] = 2.2f,
    ['T' - 'A'][26] = 2.2f,
    [26][26]        = 3.0f, /* Double blank retains incredible bingo power */

    /* High synergy bigrams (common prefixes, suffixes, digraphs) */
    ['I' - 'A']['N' - 'A'] = 2.5f,
    ['N' - 'A']['I' - 'A'] = 2.5f,
    ['N' - 'A']['G' - 'A'] = 3.0f,
    ['G' - 'A']['N' - 'A'] = 3.0f,
    ['E' - 'A']['R' - 'A'] = 2.8f,
    ['R' - 'A']['E' - 'A'] = 2.8f,
    ['E' - 'A']['D' - 'A'] = 2.2f,
    ['D' - 'A']['E' - 'A'] = 2.2f,
    ['E' - 'A']['S' - 'A'] = 2.5f,
    ['S' - 'A']['E' - 'A'] = 2.5f,
    ['S' - 'A']['T' - 'A'] = 2.2f,
    ['T' - 'A']['S' - 'A'] = 2.2f,
    ['R' - 'A']['T' - 'A'] = 2.0f,
    ['T' - 'A']['R' - 'A'] = 2.0f,
    ['C' - 'A']['H' - 'A'] = 2.5f,
    ['H' - 'A']['C' - 'A'] = 2.5f,
    ['S' - 'A']['H' - 'A'] = 2.2f,
    ['H' - 'A']['S' - 'A'] = 2.2f,
    ['T' - 'A']['H' - 'A'] = 2.0f,
    ['H' - 'A']['T' - 'A'] = 2.0f,
    ['P' - 'A']['H' - 'A'] = 1.8f,
    ['H' - 'A']['P' - 'A'] = 1.8f,

    /* Toxic consonant clash pairs (severely clunky together) */
    ['V' - 'A']['W' - 'A'] = -8.5f,
    ['W' - 'A']['V' - 'A'] = -8.5f,
    ['V' - 'A']['Q' - 'A'] = -9.0f,
    ['Q' - 'A']['V' - 'A'] = -9.0f,
    ['V' - 'A']['Z' - 'A'] = -6.5f,
    ['Z' - 'A']['V' - 'A'] = -6.5f,
    ['V' - 'A']['J' - 'A'] = -7.0f,
    ['J' - 'A']['V' - 'A'] = -7.0f,
    ['W' - 'A']['Q' - 'A'] = -8.0f,
    ['Q' - 'A']['W' - 'A'] = -8.0f,
    ['J' - 'A']['Q' - 'A'] = -9.0f,
    ['Q' - 'A']['J' - 'A'] = -9.0f,
    ['Q' - 'A']['Z' - 'A'] = -10.0f,
    ['Z' - 'A']['Q' - 'A'] = -10.0f,
    ['K' - 'A']['Q' - 'A'] = -7.0f,
    ['Q' - 'A']['K' - 'A'] = -7.0f,
    ['B' - 'A']['V' - 'A'] = -4.5f,
    ['V' - 'A']['B' - 'A'] = -4.5f,
    ['F' - 'A']['V' - 'A'] = -6.0f,
    ['V' - 'A']['F' - 'A'] = -6.0f,
    ['F' - 'A']['W' - 'A'] = -5.5f,
    ['W' - 'A']['F' - 'A'] = -5.5f,

    /* Duplicate penalties (Diagonal) */
    ['A' - 'A']['A' - 'A'] = -3.0f,
    ['B' - 'A']['B' - 'A'] = -4.5f,
    ['C' - 'A']['C' - 'A'] = -4.5f,
    ['D' - 'A']['D' - 'A'] = -3.0f,
    ['E' - 'A']['E' - 'A'] = -1.2f,
    ['F' - 'A']['F' - 'A'] = -6.0f,
    ['G' - 'A']['G' - 'A'] = -4.5f,
    ['H' - 'A']['H' - 'A'] = -4.0f,
    ['I' - 'A']['I' - 'A'] = -4.5f,
    ['J' - 'A']['J' - 'A'] = -8.0f,
    ['K' - 'A']['K' - 'A'] = -7.0f,
    ['L' - 'A']['L' - 'A'] = -2.5f,
    ['M' - 'A']['M' - 'A'] = -3.5f,
    ['N' - 'A']['N' - 'A'] = -2.0f,
    ['O' - 'A']['O' - 'A'] = -4.5f,
    ['P' - 'A']['P' - 'A'] = -4.0f,
    ['Q' - 'A']['Q' - 'A'] = -12.0f,
    ['R' - 'A']['R' - 'A'] = -2.0f,
    ['S' - 'A']['S' - 'A'] = -1.5f, /* Second S is still good, just less unique */
    ['T' - 'A']['T' - 'A'] = -2.0f,
    ['U' - 'A']['U' - 'A'] = -6.0f,
    ['V' - 'A']['V' - 'A'] = -12.0f, /* VV is devastating */
    ['W' - 'A']['W' - 'A'] = -8.0f,
    ['X' - 'A']['X' - 'A'] = -8.0f,
    ['Y' - 'A']['Y' - 'A'] = -4.5f,
    ['Z' - 'A']['Z' - 'A'] = -6.0f,
};

PhaseBucket eval_classify_phase(int bag_count) {
    if (bag_count >= 80) return PHASE_BUCKET_OPENING;
    if (bag_count >= 60) return PHASE_BUCKET_EARLY;
    if (bag_count >= 40) return PHASE_BUCKET_MIDGAME;
    if (bag_count >= 20) return PHASE_BUCKET_LATE_MID;
    if (bag_count >= 10) return PHASE_BUCKET_PRE_ENDGAME;
    if (bag_count >= 1)  return PHASE_BUCKET_EXHAUSTION;
    return PHASE_BUCKET_ENDGAME;
}

OpennessBucket eval_classify_openness(const Board *board) {
    int occupied = bb_popcount(&board->occupied);
    int tw_exposed = bb_popcount_and(&board->anchors, &BB_TW);

    if (occupied < 8) return OPENNESS_CLOSED;
    if (tw_exposed >= 3) return OPENNESS_WIDE;
    if (tw_exposed >= 1) return OPENNESS_EXPOSED;
    return OPENNESS_STANDARD;
}

static inline bool is_vowel_idx(int idx) {
    return (0x104111U & (1U << idx)) != 0;
}

static inline bool is_vowel_c(char c) {
    int idx = (c >= 'a' && c <= 'z') ? (c - 'a') : (c - 'A');
    return (idx >= 0 && idx < 26) && is_vowel_idx(idx);
}

typedef struct {
    const char *stem;
    float value;
} StemBonus;

/* 6-letter top championship bingo stems (sorted alphabetically for bsearch) */
static const StemBonus STEMS_6[] = {
    {"AAERST", 19.0f},
    {"ABENRT", 18.5f},
    {"ACENRT", 20.5f},
    {"ACERST", 21.0f},
    {"ACEIST", 19.5f},
    {"ADEINR", 22.0f},
    {"ADEINT", 22.5f},
    {"ADEIRS", 23.0f},
    {"ADEIRT", 22.5f},
    {"ADEIST", 22.0f},
    {"ADENRS", 24.0f},
    {"ADENRT", 23.5f},
    {"ADENST", 23.0f},
    {"ADERST", 23.0f},
    {"AEGINR", 21.0f},
    {"AEGINS", 21.5f},
    {"AEGIST", 19.5f},
    {"AEGNRT", 20.5f},
    {"AEILNR", 23.0f},
    {"AEILNS", 22.5f},
    {"AEILNT", 23.0f},
    {"AEILRS", 22.5f},
    {"AEILRT", 23.5f},
    {"AEILST", 23.0f},
    {"AEIMNR", 20.5f},
    {"AEIMNT", 20.5f},
    {"AEIMRS", 20.5f},
    {"AEIMST", 20.0f},
    {"AEINPR", 21.5f},
    {"AEINPT", 21.0f},
    {"AEINRS", 28.0f},
    {"AEINRT", 29.5f},
    {"AEINST", 30.5f}, /* SATINE / TISANE */
    {"AEIPRS", 21.0f},
    {"AEIPRT", 21.5f},
    {"AEIPST", 20.5f},
    {"AEIRRS", 20.0f},
    {"AEIRRT", 20.5f},
    {"AEIRST", 26.5f},
    {"AELNRT", 22.5f},
    {"AELNST", 22.0f},
    {"AELRST", 25.0f},
    {"AEMORT", 18.0f},
    {"AENORT", 22.0f},
    {"AENRST", 25.5f},
    {"AEOPRS", 19.5f},
    {"AEORRS", 19.0f},
    {"AEORST", 24.5f},
    {"AGINRS", 22.0f},
    {"AGINRT", 21.5f},
    {"AGINST", 21.5f},
    {"AINRST", 22.5f},
    {"CEINRS", 20.5f},
    {"CEINRT", 21.0f},
    {"CEINST", 20.5f},
    {"DEINRS", 22.5f},
    {"DEINRT", 22.5f},
    {"DEINST", 22.0f},
    {"EEINRS", 23.0f},
    {"EEINRT", 23.5f},
    {"EEINST", 24.0f},
    {"EILNRT", 21.5f},
    {"EILNST", 21.0f},
    {"EILRST", 22.5f},
    {"EIMNRT", 20.0f},
    {"EINORT", 21.0f},
    {"EINPRT", 20.5f},
    {"EINRST", 27.0f},
    {"EIORST", 22.0f},
    {"ENORST", 22.0f}
};

/* 5-letter top championship bingo stems (sorted alphabetically for bsearch) */
static const StemBonus STEMS_5[] = {
    {"ACEIN", 12.0f},
    {"ACERS", 12.5f},
    {"ACERT", 12.5f},
    {"ADEIN", 13.0f},
    {"ADEIR", 13.5f},
    {"ADEIS", 13.5f},
    {"ADEIT", 13.5f},
    {"ADERS", 14.0f},
    {"AEINR", 15.5f},
    {"AEINS", 15.5f},
    {"AEINT", 15.5f},
    {"AEIRS", 16.5f},
    {"AEIRT", 16.5f},
    {"AELRT", 14.5f},
    {"AELST", 14.5f},
    {"AENRS", 16.0f},
    {"AENST", 16.0f},
    {"AEORS", 14.0f},
    {"AEORT", 14.0f},
    {"AEPRS", 12.5f},
    {"AEPRT", 12.5f},
    {"AERST", 18.0f},
    {"AGIN",  10.5f},
    {"AGINS", 13.5f},
    {"AGIRT", 12.0f},
    {"AINRS", 14.0f},
    {"AINST", 14.5f},
    {"DENRS", 13.5f},
    {"EGIRT", 13.0f},
    {"EGNST", 13.0f},
    {"EILRS", 14.0f},
    {"EILRT", 14.0f},
    {"EILST", 14.0f},
    {"EINRS", 17.0f},
    {"EINRT", 16.0f},
    {"EINST", 17.5f},
    {"EIRST", 15.0f},
    {"ELNST", 14.0f},
    {"ENORT", 14.0f},
    {"ENOST", 13.5f},
    {"ENRST", 15.5f},
    {"EORST", 14.5f},
    {"INRST", 15.0f}
};

/* 4-letter top cores (sorted alphabetically for bsearch) */
static const StemBonus STEMS_4[] = {
    {"ACER",  7.5f},
    {"ACES",  7.5f},
    {"ACET",  7.5f},
    {"AEIN",  9.5f},
    {"AEIR",  9.5f},
    {"AEIS",  9.5f},
    {"AEIT",  9.5f},
    {"AERS", 10.5f},
    {"AERT", 10.0f},
    {"AEST", 10.5f},
    {"AIRS",  9.5f},
    {"AIST",  9.0f},
    {"ANST",  9.0f},
    {"APER",  7.5f},
    {"APES",  7.5f},
    {"APET",  7.5f},
    {"DERS",  8.0f},
    {"DEST",  8.0f},
    {"EINT",  9.5f},
    {"EINR",  9.5f},
    {"EINS", 10.0f},
    {"EIRS", 10.5f},
    {"ELRT",  8.5f},
    {"ELST",  8.5f},
    {"ENRS",  9.0f},
    {"ENST", 10.0f},
    {"EORS",  8.5f},
    {"EORT",  8.5f},
    {"ERST", 11.5f},
    {"INRS",  9.0f},
    {"INRT",  9.0f},
    {"INST",  9.5f},
    {"IRST",  9.5f},
    {"LNST",  8.0f},
    {"NRST",  8.5f},
    {"ORST",  8.5f}
};

/* 3-letter top cores (sorted alphabetically for bsearch) */
static const StemBonus STEMS_3[] = {
    {"AER",  4.5f},
    {"AES",  4.5f},
    {"AET",  4.5f},
    {"AIR",  3.8f},
    {"AIS",  3.8f},
    {"AIT",  3.8f},
    {"EIN",  4.0f},
    {"EIR",  4.0f},
    {"EIS",  4.0f},
    {"EIT",  4.0f},
    {"ENT",  4.0f},
    {"ERS",  5.5f},
    {"EST",  5.0f},
    {"ING",  4.8f},
    {"INR",  3.8f},
    {"INS",  3.8f},
    {"INT",  4.0f},
    {"LRT",  3.5f},
    {"LST",  3.5f},
    {"NST",  3.5f},
    {"RST",  5.0f},
    {"SIN",  4.0f},
    {"TIN",  4.0f}
};

/*
 * Higher-Order Super-Leave & Stem Recognition:
 * Binary searches sorted leave substrings against championship bingo stems.
 */
static inline float lookup_stem_6(const char *key) {
    int low = 0, high = (int)(sizeof(STEMS_6) / sizeof(STEMS_6[0])) - 1;
    while (low <= high) {
        int mid = (low + high) >> 1;
        int cmp = strcmp(key, STEMS_6[mid].stem);
        if (cmp == 0) return STEMS_6[mid].value * 0.15f;
        if (cmp < 0) high = mid - 1;
        else low = mid + 1;
    }
    return 0.0f;
}

static inline float lookup_stem_5(const char *key) {
    int low = 0, high = (int)(sizeof(STEMS_5) / sizeof(STEMS_5[0])) - 1;
    while (low <= high) {
        int mid = (low + high) >> 1;
        int cmp = strcmp(key, STEMS_5[mid].stem);
        if (cmp == 0) return STEMS_5[mid].value * 0.15f;
        if (cmp < 0) high = mid - 1;
        else low = mid + 1;
    }
    return 0.0f;
}

static inline float lookup_stem_4(const char *key) {
    int low = 0, high = (int)(sizeof(STEMS_4) / sizeof(STEMS_4[0])) - 1;
    while (low <= high) {
        int mid = (low + high) >> 1;
        int cmp = strcmp(key, STEMS_4[mid].stem);
        if (cmp == 0) return STEMS_4[mid].value * 0.15f;
        if (cmp < 0) high = mid - 1;
        else low = mid + 1;
    }
    return 0.0f;
}

static inline float lookup_stem_3(const char *key) {
    int low = 0, high = (int)(sizeof(STEMS_3) / sizeof(STEMS_3[0])) - 1;
    while (low <= high) {
        int mid = (low + high) >> 1;
        int cmp = strcmp(key, STEMS_3[mid].stem);
        if (cmp == 0) return STEMS_3[mid].value * 0.15f;
        if (cmp < 0) high = mid - 1;
        else low = mid + 1;
    }
    return 0.0f;
}

/*
 * Higher-Order Super-Leave & Stem Recognition:
 * Inlined branch-predicted binary searches across sorted championship bingo stems.
 */
static inline float eval_super_leave_stems(const char *sorted_leave, int len) {
    if (len == 6) return lookup_stem_6(sorted_leave);
    if (len == 5) return lookup_stem_5(sorted_leave);
    if (len == 4) return lookup_stem_4(sorted_leave);
    if (len == 3) return lookup_stem_3(sorted_leave);
    return 0.0f;
}

float eval_rack_leave_bucketed(const char *leave, PhaseBucket phase) {
    if (!leave) return 0.0f;
    int len = (int)strlen(leave);
    if (len == 0) return 0.0f;

    int p_idx = (int)phase;
    if (p_idx < 0) p_idx = 0;
    if (p_idx > 7) p_idx = 7;

    float equity = 0.0f;
    int counts[27] = {0};
    int vowels = 0, consonants = 0, blanks = 0;
    int high_point_tiles = 0; /* F, H, V, W, Y, K, J, X, Q, Z */

    /* 1. Accumulator A: Single Tile Values */
    for (int i = 0; i < len; i++) {
        char c = leave[i];
        if (c == '?') {
            counts[26]++;
            blanks++;
            equity += PHASE_TILE_EQUITY[p_idx][26];
        } else {
            int idx = (c >= 'a' && c <= 'z') ? (c - 'a') : (c - 'A');
            if (idx >= 0 && idx < 26) {
                counts[idx]++;
                equity += PHASE_TILE_EQUITY[p_idx][idx];
                if (is_vowel_idx(idx)) {
                    vowels++;
                } else {
                    consonants++;
                }
                if (TILE_VALUES[idx] >= 4) {
                    high_point_tiles++;
                }
            }
        }
    }

    /* Counting-sort leave for stem lookup in O(N) */
    char sorted_leave[RACK_CAPACITY + 1];
    int slen = 0;
    for (int c = 0; c < 26; c++) {
        for (int k = 0; k < counts[c]; k++) {
            sorted_leave[slen++] = (char)('A' + c);
        }
    }
    for (int k = 0; k < counts[26]; k++) {
        sorted_leave[slen++] = '?';
    }
    sorted_leave[slen] = '\0';

    /* 2. Accumulator B: Pairwise Letter Synergy directly from sorted uppercase leave */
    for (int i = 0; i < slen; i++) {
        int idx1 = (sorted_leave[i] == '?') ? 26 : (sorted_leave[i] - 'A');
        for (int j = i + 1; j < slen; j++) {
            int idx2 = (sorted_leave[j] == '?') ? 26 : (sorted_leave[j] - 'A');
            equity += PAIRWISE_SYNERGY[idx1][idx2];
        }
    }

    /* 3. Higher-Order Super-Leaves and Bingo Stems */
    if (blanks == 0) {
        equity += eval_super_leave_stems(sorted_leave, slen);
    } else if (blanks == 1 && slen >= 4) {
        /* With a blank, any 4+ letter balanced leave gets a bingo readiness multiplier */
        equity += 4.5f;
    }

    /* 4. Non-linear Duplicate Letter Penalties */
    if (counts['A' - 'A'] > 2) equity -= (counts['A' - 'A'] - 2) * 3.5f;
    if (counts['E' - 'A'] > 2) equity -= (counts['E' - 'A'] - 2) * 2.5f;
    if (counts['I' - 'A'] > 2) equity -= (counts['I' - 'A'] - 2) * 5.5f;
    if (counts['O' - 'A'] > 2) equity -= (counts['O' - 'A'] - 2) * 5.0f;
    if (counts['U' - 'A'] > 1) equity -= (counts['U' - 'A'] - 1) * 4.0f;

    /* Duplicate S penalties (1 S is golden; 2 S is slightly redundant; 3+ S is toxic) */
    if (counts['S' - 'A'] == 2) equity -= 2.5f;
    else if (counts['S' - 'A'] == 3) equity -= 18.0f;
    else if (counts['S' - 'A'] >= 4) equity -= 35.0f;

    /* Duplicate consonant penalties across other consonants */
    const char dupe_cons[] = "BCDFGHJKLMNPRTVWYZ";
    for (size_t dc = 0; dc < sizeof(dupe_cons) - 1; dc++) {
        int c_idx = dupe_cons[dc] - 'A';
        if (counts[c_idx] >= 2) {
            equity -= (counts[c_idx] - 1) * 3.5f;
        }
        if (counts[c_idx] >= 3) {
            equity -= 10.0f;
        }
    }

    /* Double V / W / J / Q / Z devastating rack penalties */
    if (counts['V' - 'A'] >= 2) equity -= 8.0f; /* Combined with diagonal: total ~ -20 pts */
    if (counts['W' - 'A'] >= 2) equity -= 5.0f;
    if (counts['Q' - 'A'] >= 1 && counts['U' - 'A'] == 0 && counts[26] == 0) {
        /* Q without U or Blank penalty grows drastically as bag empties */
        if (p_idx <= PHASE_BUCKET_PRE_ENDGAME) {
            equity -= 6.0f; /* Extra penalty in late game */
        }
    }

    /* 5. High-Point Tile Clumping Penalty */
    if (high_point_tiles == 2) equity -= 2.5f;
    else if (high_point_tiles == 3) equity -= 7.5f;
    else if (high_point_tiles >= 4) equity -= 15.0f;

    /* 6. Vowel / Consonant Ratio Matrix */
    int non_blanks = vowels + consonants;
    if (non_blanks >= 2) {
        if (vowels == 0) {
            if (non_blanks == 2) equity -= 2.5f;
            else if (non_blanks == 3) equity -= 6.5f;
            else if (non_blanks == 4) equity -= 12.0f;
            else equity -= 20.0f;
        } else if (consonants == 0) {
            if (non_blanks == 2) equity -= 3.5f;
            else if (non_blanks == 3) equity -= 8.0f;
            else if (non_blanks == 4) equity -= 14.0f;
            else equity -= 24.0f;
        } else {
            /* Balanced ratio reward & imbalance penalty */
            if (non_blanks == 2 && vowels == 1 && consonants == 1) equity += 1.5f;
            else if (non_blanks == 3 && (vowels == 1 || vowels == 2)) equity += 1.8f;
            else if (non_blanks == 4) {
                if (vowels == 2 && consonants == 2) equity += 2.5f;
                else if (vowels == 1 && consonants == 3) equity += 1.2f;
                else if (vowels >= 3) equity -= 3.5f;
            } else if (non_blanks == 5) {
                if (vowels == 2 && consonants == 3) equity += 2.8f;
                else if (vowels == 3 && consonants == 2) equity += 2.0f;
                else if (vowels == 1 && consonants == 4) equity -= 2.5f;
                else if (vowels >= 4) equity -= 9.0f;
            } else if (non_blanks == 6) {
                if (vowels == 2 || vowels == 3) equity += 2.2f;
                else if (vowels == 1) equity -= 4.0f;
                else if (vowels >= 4) equity -= 10.0f;
            }
        }
    }

    return equity;
}

float eval_rack_leave(const char *leave) {
    return eval_rack_leave_bucketed(leave, PHASE_BUCKET_MIDGAME);
}

float eval_board_risk_bucketed(const Board *board, const Move *move, OpennessBucket openness) {
    if (move->type != MOVE_PLAY) return 0.0f;

    float risk = 0.0f;
    float penalty_scale = (openness == OPENNESS_WIDE) ? 1.25f : 1.0f;

    for (int p = 0; p < move->num_placed; p++) {
        int r = move->placed[p].row;
        int c = move->placed[p].col;
        char let = move->placed[p].letter;
        bool is_v = is_vowel_c(let);

        int dr[4] = {-1, 1, 0, 0};
        int dc[4] = {0, 0, -1, 1};

        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d];
            int nc = c + dc[d];
            if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                if (board->cells[nr][nc].tile == 0) {
                    /* Penalize opening unanchored Triple Word and Double Word squares to the opponent */
                    MultiplierType mult = board_get_multiplier(nr, nc);
                    if (mult == MULT_TW && !board_is_anchor(board, nr, nc)) {
                        risk -= (is_v ? 4.5f : 2.5f) * penalty_scale;
                    } else if (mult == MULT_DW && !board_is_anchor(board, nr, nc)) {
                        risk -= (is_v ? 1.5f : 0.8f) * penalty_scale;
                    }
                }
            }
        }
    }

    return risk;
}

float eval_win_probability(int score_diff, int bag_count) {
    float spread_std = 15.0f + 0.40f * (float)bag_count;
    float z = (float)score_diff / spread_std;
    return 1.0f / (1.0f + expf(-z));
}

float eval_score_margin_risk_multiplier(int score_diff) {
    if (score_diff >= 60) return 2.2f;   /* Heavy lead: maximum defense */
    if (score_diff >= 30) return 1.6f;   /* Moderate lead: conservative */
    if (score_diff >= 10) return 1.2f;
    if (score_diff <= -60) return 0.35f; /* Heavy deficit: aggressive offense */
    if (score_diff <= -30) return 0.55f; /* Moderate deficit: open board */
    if (score_diff <= -10) return 0.80f;
    return 1.0f;                         /* Close game: balanced */
}

float eval_board_risk_with_score(const Board *board, const Move *move, OpennessBucket openness, int score_diff) {
    float base_risk = eval_board_risk_bucketed(board, move, openness);
    float mult = eval_score_margin_risk_multiplier(score_diff);
    return base_risk * mult;
}

float eval_rack_leave_bucketed_with_score(const char *leave, PhaseBucket phase, int score_diff) {
    float eq = eval_rack_leave_bucketed(leave, phase);
    /* Trailing: amplify stem/bingo retention for comeback */
    if (score_diff <= -30 && eq > 10.0f) {
        eq *= 1.25f;
    } else if (score_diff >= 40 && eq > 15.0f) {
        /* Leading: slight discount on speculative bingo stems in favor of immediate score */
        eq *= 0.90f;
    }
    return eq;
}

float eval_board_risk(const Board *board, const Move *move) {
    OpennessBucket open = eval_classify_openness(board);
    return eval_board_risk_bucketed(board, move, open);
}

DualAccumulator eval_move_accumulate_with_score(const Board *board, const Move *move, int bag_count, int score_diff) {
    DualAccumulator acc;
    acc.board_acc = 0.0f;
    acc.rack_acc = 0.0f;
    acc.net_equity = 0.0f;

    if (move->type != MOVE_PLAY) return acc;

    PhaseBucket phase = eval_classify_phase(bag_count);
    OpennessBucket openness = eval_classify_openness(board);

    /* Hand-tuned Grandmaster Heuristic: board risk + 8-phase rack leave tables with score context */
    float board_risk = eval_board_risk_with_score(board, move, openness, score_diff);
    float rack_leave = eval_rack_leave_bucketed_with_score(move->leave, phase, score_diff);
    float heuristic_future = board_risk + rack_leave;

    /* If NNUE is loaded, combine Hand-Tuned Heuristic and Deep NNUE into a calibrated convex ensemble */
    if (g_nnue.is_loaded) {
        alignas(64) float features[NNUE_INPUT_DIM];
        nnue_extract_features_with_score(board, move, NULL, bag_count, score_diff, features);
        float nnue_pred = nnue_forward(&g_nnue, features);
        float blended_future = (0.65f * heuristic_future) + (0.35f * nnue_pred);
        acc.board_acc = (float)move->score + (0.65f * board_risk);
        acc.rack_acc = (0.65f * rack_leave) + (0.35f * nnue_pred);
        acc.net_equity = (float)move->score + blended_future;
        return acc;
    }

    acc.board_acc = (float)move->score + board_risk;
    acc.rack_acc = rack_leave;
    acc.net_equity = (float)move->score + heuristic_future;
    return acc;
}

DualAccumulator eval_move_accumulate(const Board *board, const Move *move, int bag_count) {
    return eval_move_accumulate_with_score(board, move, bag_count, 0);
}

DualAccumulator eval_move_accumulate_with_score_incremental(const Board *board, const Move *move,
                                                           const NNUEAccumulator *base_acc,
                                                           int bag_count, int score_diff) {
    DualAccumulator acc = {0};
    if (move->type != MOVE_PLAY) return acc;

    PhaseBucket phase = eval_classify_phase(bag_count);
    OpennessBucket openness = eval_classify_openness(board);

    float board_risk = eval_board_risk_with_score(board, move, openness, score_diff);
    float rack_leave = eval_rack_leave_bucketed_with_score(move->leave, phase, score_diff);
    float heuristic_future = board_risk + rack_leave;

    if (g_nnue.is_loaded && base_acc) {
        float nnue_pred = nnue_forward_move_incremental(&g_nnue, base_acc, board, move, bag_count, score_diff);
        float blended_future = (0.65f * heuristic_future) + (0.35f * nnue_pred);
        acc.board_acc = (float)move->score + (0.65f * board_risk);
        acc.rack_acc = (0.65f * rack_leave) + (0.35f * nnue_pred);
        acc.net_equity = (float)move->score + blended_future;
        return acc;
    }

    return eval_move_accumulate_with_score(board, move, bag_count, score_diff);
}

float eval_move_static_incremental(const Board *board, const Move *move,
                                  const NNUEAccumulator *base_acc,
                                  int bag_count, int score_diff) {
    DualAccumulator acc = eval_move_accumulate_with_score_incremental(board, move, base_acc, bag_count, score_diff);
    return acc.net_equity;
}

DualAccumulator eval_move_accumulate_with_score_quantized_incremental(const Board *board, const Move *move,
                                                                     const NNUEQuantizedAccumulator *base_acc,
                                                                     int bag_count, int score_diff) {
    DualAccumulator acc = {0};
    if (move->type != MOVE_PLAY) return acc;

    PhaseBucket phase = eval_classify_phase(bag_count);
    OpennessBucket openness = eval_classify_openness(board);

    float board_risk = eval_board_risk_with_score(board, move, openness, score_diff);
    float rack_leave = eval_rack_leave_bucketed_with_score(move->leave, phase, score_diff);
    float heuristic_future = board_risk + rack_leave;

    if (g_nnue_quantized.is_loaded && base_acc) {
        float nnue_pred = nnue_forward_move_quantized_incremental(&g_nnue_quantized, base_acc, board, move, bag_count, score_diff);
        float blended_future = (0.65f * heuristic_future) + (0.35f * nnue_pred);
        acc.board_acc = (float)move->score + (0.65f * board_risk);
        acc.rack_acc = (0.65f * rack_leave) + (0.35f * nnue_pred);
        acc.net_equity = (float)move->score + blended_future;
        return acc;
    }

    return eval_move_accumulate_with_score(board, move, bag_count, score_diff);
}

float eval_move_static_quantized_incremental(const Board *board, const Move *move,
                                            const NNUEQuantizedAccumulator *base_acc,
                                            int bag_count, int score_diff) {
    DualAccumulator acc = eval_move_accumulate_with_score_quantized_incremental(board, move, base_acc, bag_count, score_diff);
    return acc.net_equity;
}

float eval_move_static_with_score(const Board *board, const Move *move, int bag_count, int score_diff) {
    DualAccumulator acc = eval_move_accumulate_with_score(board, move, bag_count, score_diff);
    return acc.net_equity;
}

float eval_move_static_bucketed(const Board *board, const Move *move, int bag_count) {
    DualAccumulator acc = eval_move_accumulate(board, move, bag_count);
    return acc.net_equity;
}

float eval_move_static(const Board *board, const Move *move) {
    return eval_move_static_bucketed(board, move, 50);
}

float eval_move_heuristic_fast(const Board *board, const Move *move, int bag_count) {
    if (move->type != MOVE_PLAY) return 0.0f;
    PhaseBucket phase = eval_classify_phase(bag_count);
    OpennessBucket openness = eval_classify_openness(board);
    float board_acc = (float)move->score + eval_board_risk_bucketed(board, move, openness);
    float rack_acc = eval_rack_leave_bucketed(move->leave, phase);
    return board_acc + rack_acc;
}

static int compare_moves_descending(const void *a, const void *b) {
    const Move *ma = (const Move*)a;
    const Move *mb = (const Move*)b;
    float ea = ma->sim_equity != 0.0f ? ma->sim_equity : ma->leave_equity;
    float eb = mb->sim_equity != 0.0f ? mb->sim_equity : mb->leave_equity;
    if (eb > ea) return 1;
    if (eb < ea) return -1;
    return 0;
}

void eval_sort_moves(MoveList *list) {
    if (list->count > 1) {
        qsort(list->moves, list->count, sizeof(Move), compare_moves_descending);
    }
}

/*
 * Exact Sub-rack Exchange Optimizer:
 * Evaluates all 2^k subsets of the rack to find the optimal keep/discard split.
 */
bool eval_should_exchange(const Board *board, const Rack *rack, int bag_count,
                         float best_play_equity, Move *out_exchange_move) {
    (void)board;
    if (bag_count < RACK_CAPACITY || rack->count < RACK_CAPACITY) {
        return false;
    }
    if (best_play_equity >= 14.0f) {
        return false;
    }

    PhaseBucket phase = eval_classify_phase(bag_count);

    /*
     * Tempo Cost of an Exchange:
     * In exchange, player scores 0 and gives opponent a free tempo.
     * Estimated tempo loss is ~16 pts in early/mid game, increasing to 22 pts in late game.
     */
    float tempo_cost = (phase >= PHASE_BUCKET_MIDGAME) ? 16.0f : 20.0f;

    /* Best achievable leave by exchanging a subset of tiles */
    float best_subset_equity = -1000.0f;
    int best_subset_mask = 0;

    int num_subsets = 1 << rack->count; /* 128 subsets for 7 tiles */

    for (int mask = 0; mask < num_subsets - 1; mask++) {
        /* Discard at least 1 tile (mask cannot contain all 7 tiles) */
        char keep[RACK_CAPACITY + 1];
        int klen = 0;

        for (int i = 0; i < rack->count; i++) {
            if ((mask & (1 << i)) != 0) {
                keep[klen++] = rack->tiles[i];
            }
        }
        keep[klen] = '\0';

        /* Evaluate keep equity */
        float eq = eval_rack_leave_bucketed(keep, phase);

        if (eq > best_subset_equity) {
            best_subset_equity = eq;
            best_subset_mask = mask;
        }
    }

    float net_exchange_equity = best_subset_equity - tempo_cost;

    /* Only exchange if optimal exchange equity strictly outperforms best board play */
    if (net_exchange_equity > best_play_equity && best_play_equity < 14.0f) {
        out_exchange_move->type = MOVE_EXCHANGE;
        out_exchange_move->score = 0;
        out_exchange_move->row = 0;
        out_exchange_move->col = 0;
        out_exchange_move->dir = DIR_HORIZONTAL;

        int discard_count = 0;
        char keep[RACK_CAPACITY + 1];
        int klen = 0;

        for (int i = 0; i < rack->count; i++) {
            if ((best_subset_mask & (1 << i)) == 0) {
                out_exchange_move->placed[discard_count].letter = rack->tiles[i];
                out_exchange_move->placed[discard_count].is_blank = (rack->tiles[i] == '?');
                discard_count++;
            } else {
                keep[klen++] = rack->tiles[i];
            }
        }
        keep[klen] = '\0';

        out_exchange_move->num_placed = discard_count;
        strcpy(out_exchange_move->leave, keep);
        out_exchange_move->leave_equity = best_subset_equity;
        out_exchange_move->sim_equity = net_exchange_equity;
        return true;
    }

    return false;
}
