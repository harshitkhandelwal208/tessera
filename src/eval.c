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
static float PAIRWISE_SYNERGY[27][27];
static bool synergy_matrix_initialized = false;

static void init_synergy_matrix(void) {
    if (synergy_matrix_initialized) return;
    memset(PAIRWISE_SYNERGY, 0, sizeof(PAIRWISE_SYNERGY));

    /* QU synergy */
    PAIRWISE_SYNERGY['Q' - 'A']['U' - 'A'] = 10.0f;
    PAIRWISE_SYNERGY['U' - 'A']['Q' - 'A'] = 10.0f;

    /* Blank + S & Blank + Key Consonants */
    PAIRWISE_SYNERGY[26]['S' - 'A'] = 4.5f;
    PAIRWISE_SYNERGY['S' - 'A'][26] = 4.5f;
    PAIRWISE_SYNERGY[26]['R' - 'A'] = 2.5f;
    PAIRWISE_SYNERGY['R' - 'A'][26] = 2.5f;
    PAIRWISE_SYNERGY[26]['T' - 'A'] = 2.2f;
    PAIRWISE_SYNERGY['T' - 'A'][26] = 2.2f;
    PAIRWISE_SYNERGY[26][26]        = 3.0f; /* Double blank retains incredible bingo power */

    /* High synergy bigrams (common prefixes, suffixes, digraphs) */
    PAIRWISE_SYNERGY['I' - 'A']['N' - 'A'] = 2.5f;
    PAIRWISE_SYNERGY['N' - 'A']['I' - 'A'] = 2.5f;
    PAIRWISE_SYNERGY['N' - 'A']['G' - 'A'] = 3.0f;
    PAIRWISE_SYNERGY['G' - 'A']['N' - 'A'] = 3.0f;
    PAIRWISE_SYNERGY['E' - 'A']['R' - 'A'] = 2.8f;
    PAIRWISE_SYNERGY['R' - 'A']['E' - 'A'] = 2.8f;
    PAIRWISE_SYNERGY['E' - 'A']['D' - 'A'] = 2.2f;
    PAIRWISE_SYNERGY['D' - 'A']['E' - 'A'] = 2.2f;
    PAIRWISE_SYNERGY['E' - 'A']['S' - 'A'] = 2.5f;
    PAIRWISE_SYNERGY['S' - 'A']['E' - 'A'] = 2.5f;
    PAIRWISE_SYNERGY['S' - 'A']['T' - 'A'] = 2.2f;
    PAIRWISE_SYNERGY['T' - 'A']['S' - 'A'] = 2.2f;
    PAIRWISE_SYNERGY['R' - 'A']['T' - 'A'] = 2.0f;
    PAIRWISE_SYNERGY['T' - 'A']['R' - 'A'] = 2.0f;
    PAIRWISE_SYNERGY['C' - 'A']['H' - 'A'] = 2.5f;
    PAIRWISE_SYNERGY['H' - 'A']['C' - 'A'] = 2.5f;
    PAIRWISE_SYNERGY['S' - 'A']['H' - 'A'] = 2.2f;
    PAIRWISE_SYNERGY['H' - 'A']['S' - 'A'] = 2.2f;
    PAIRWISE_SYNERGY['T' - 'A']['H' - 'A'] = 2.0f;
    PAIRWISE_SYNERGY['H' - 'A']['T' - 'A'] = 2.0f;
    PAIRWISE_SYNERGY['P' - 'A']['H' - 'A'] = 1.8f;
    PAIRWISE_SYNERGY['H' - 'A']['P' - 'A'] = 1.8f;

    /* Toxic consonant clash pairs (severely clunky together) */
    PAIRWISE_SYNERGY['V' - 'A']['W' - 'A'] = -8.5f;
    PAIRWISE_SYNERGY['W' - 'A']['V' - 'A'] = -8.5f;
    PAIRWISE_SYNERGY['V' - 'A']['Q' - 'A'] = -9.0f;
    PAIRWISE_SYNERGY['Q' - 'A']['V' - 'A'] = -9.0f;
    PAIRWISE_SYNERGY['V' - 'A']['Z' - 'A'] = -6.5f;
    PAIRWISE_SYNERGY['Z' - 'A']['V' - 'A'] = -6.5f;
    PAIRWISE_SYNERGY['V' - 'A']['J' - 'A'] = -7.0f;
    PAIRWISE_SYNERGY['J' - 'A']['V' - 'A'] = -7.0f;
    PAIRWISE_SYNERGY['W' - 'A']['Q' - 'A'] = -8.0f;
    PAIRWISE_SYNERGY['Q' - 'A']['W' - 'A'] = -8.0f;
    PAIRWISE_SYNERGY['J' - 'A']['Q' - 'A'] = -9.0f;
    PAIRWISE_SYNERGY['Q' - 'A']['J' - 'A'] = -9.0f;
    PAIRWISE_SYNERGY['Q' - 'A']['Z' - 'A'] = -10.0f;
    PAIRWISE_SYNERGY['Z' - 'A']['Q' - 'A'] = -10.0f;
    PAIRWISE_SYNERGY['K' - 'A']['Q' - 'A'] = -7.0f;
    PAIRWISE_SYNERGY['Q' - 'A']['K' - 'A'] = -7.0f;
    PAIRWISE_SYNERGY['B' - 'A']['V' - 'A'] = -4.5f;
    PAIRWISE_SYNERGY['V' - 'A']['B' - 'A'] = -4.5f;
    PAIRWISE_SYNERGY['F' - 'A']['V' - 'A'] = -6.0f;
    PAIRWISE_SYNERGY['V' - 'A']['F' - 'A'] = -6.0f;
    PAIRWISE_SYNERGY['F' - 'A']['W' - 'A'] = -5.5f;
    PAIRWISE_SYNERGY['W' - 'A']['F' - 'A'] = -5.5f;

    /* Duplicate penalties (Diagonal) */
    PAIRWISE_SYNERGY['A' - 'A']['A' - 'A'] = -3.0f;
    PAIRWISE_SYNERGY['B' - 'A']['B' - 'A'] = -4.5f;
    PAIRWISE_SYNERGY['C' - 'A']['C' - 'A'] = -4.5f;
    PAIRWISE_SYNERGY['D' - 'A']['D' - 'A'] = -3.0f;
    PAIRWISE_SYNERGY['E' - 'A']['E' - 'A'] = -1.2f;
    PAIRWISE_SYNERGY['F' - 'A']['F' - 'A'] = -6.0f;
    PAIRWISE_SYNERGY['G' - 'A']['G' - 'A'] = -4.5f;
    PAIRWISE_SYNERGY['H' - 'A']['H' - 'A'] = -4.0f;
    PAIRWISE_SYNERGY['I' - 'A']['I' - 'A'] = -4.5f;
    PAIRWISE_SYNERGY['J' - 'A']['J' - 'A'] = -8.0f;
    PAIRWISE_SYNERGY['K' - 'A']['K' - 'A'] = -7.0f;
    PAIRWISE_SYNERGY['L' - 'A']['L' - 'A'] = -2.5f;
    PAIRWISE_SYNERGY['M' - 'A']['M' - 'A'] = -3.5f;
    PAIRWISE_SYNERGY['N' - 'A']['N' - 'A'] = -2.0f;
    PAIRWISE_SYNERGY['O' - 'A']['O' - 'A'] = -4.5f;
    PAIRWISE_SYNERGY['P' - 'A']['P' - 'A'] = -4.0f;
    PAIRWISE_SYNERGY['Q' - 'A']['Q' - 'A'] = -12.0f;
    PAIRWISE_SYNERGY['R' - 'A']['R' - 'A'] = -2.0f;
    PAIRWISE_SYNERGY['S' - 'A']['S' - 'A'] = -1.5f; /* Second S is still good, just less unique */
    PAIRWISE_SYNERGY['T' - 'A']['T' - 'A'] = -2.0f;
    PAIRWISE_SYNERGY['U' - 'A']['U' - 'A'] = -6.0f;
    PAIRWISE_SYNERGY['V' - 'A']['V' - 'A'] = -12.0f; /* VV is devastating */
    PAIRWISE_SYNERGY['W' - 'A']['W' - 'A'] = -8.0f;
    PAIRWISE_SYNERGY['X' - 'A']['X' - 'A'] = -8.0f;
    PAIRWISE_SYNERGY['Y' - 'A']['Y' - 'A'] = -4.5f;
    PAIRWISE_SYNERGY['Z' - 'A']['Z' - 'A'] = -6.0f;

    synergy_matrix_initialized = true;
}

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
    int occupied = 0;
    int tw_exposed = 0;

    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (board->cells[r][c].tile != 0) {
                occupied++;
            } else if (board_get_multiplier(r, c) == MULT_TW && board_is_anchor(board, r, c)) {
                tw_exposed++;
            }
        }
    }

    if (occupied < 8) return OPENNESS_CLOSED;
    if (tw_exposed >= 3) return OPENNESS_WIDE;
    if (tw_exposed >= 1) return OPENNESS_EXPOSED;
    return OPENNESS_STANDARD;
}

static bool is_vowel_c(char c) {
    c = (char)toupper((unsigned char)c);
    return (c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U');
}

/*
 * Higher-Order Super-Leave & Stem Recognition:
 * Matches sorted leave substrings against top championship bingo stems.
 */
static float eval_super_leave_stems(const char *sorted_leave, int len) {
    if (len < 3 || len > 6) return 0.0f;

    /* 6-letter legendary bingo stems */
    if (len == 6) {
        if (strcmp(sorted_leave, "AEINST") == 0) return 30.5f; /* SATINE / TISANE */
        if (strcmp(sorted_leave, "AEINRT") == 0) return 29.5f; /* RETINA / RETAIN */
        if (strcmp(sorted_leave, "ADENRS") == 0) return 24.0f; /* SNARED */
        if (strcmp(sorted_leave, "AELRST") == 0) return 25.0f; /* TALERS */
        if (strcmp(sorted_leave, "AEORST") == 0) return 24.5f; /* AORTES */
        if (strcmp(sorted_leave, "ADEIRS") == 0) return 23.0f; /* RAISED */
        if (strcmp(sorted_leave, "EINRST") == 0) return 27.0f; /* INTERS */
        if (strcmp(sorted_leave, "AEIRST") == 0) return 26.5f; /* AISTER */
        return 0.0f;
    }

    /* 5-letter top bingo stems */
    if (len == 5) {
        if (strcmp(sorted_leave, "AERST") == 0) return 18.0f;
        if (strcmp(sorted_leave, "EINST") == 0) return 17.5f;
        if (strcmp(sorted_leave, "EINRS") == 0) return 17.0f;
        if (strcmp(sorted_leave, "AEIRT") == 0) return 16.5f;
        if (strcmp(sorted_leave, "AEIRS") == 0) return 16.5f;
        if (strcmp(sorted_leave, "AENST") == 0) return 16.0f;
        if (strcmp(sorted_leave, "AEINS") == 0) return 15.5f;
        if (strcmp(sorted_leave, "AELST") == 0) return 14.5f;
        if (strcmp(sorted_leave, "AELRT") == 0) return 14.5f;
        if (strcmp(sorted_leave, "ADERS") == 0) return 14.0f;
        if (strcmp(sorted_leave, "ELNST") == 0) return 14.0f;
        if (strcmp(sorted_leave, "AINST") == 0) return 14.5f;
        return 0.0f;
    }

    /* 4-letter top cores */
    if (len == 4) {
        if (strcmp(sorted_leave, "ERST") == 0) return 11.5f;
        if (strcmp(sorted_leave, "AERT") == 0) return 10.0f;
        if (strcmp(sorted_leave, "AERS") == 0) return 10.5f;
        if (strcmp(sorted_leave, "AEST") == 0) return 10.5f;
        if (strcmp(sorted_leave, "EINS") == 0) return 10.0f;
        if (strcmp(sorted_leave, "EIRS") == 0) return 10.5f;
        if (strcmp(sorted_leave, "ENST") == 0) return 10.0f;
        if (strcmp(sorted_leave, "INST") == 0) return 9.5f;
        if (strcmp(sorted_leave, "AIRS") == 0) return 9.5f;
        if (strcmp(sorted_leave, "ANST") == 0) return 9.0f;
        if (strcmp(sorted_leave, "ELST") == 0) return 8.5f;
        if (strcmp(sorted_leave, "DERS") == 0) return 8.0f;
        return 0.0f;
    }

    /* 3-letter top cores */
    if (len == 3) {
        if (strcmp(sorted_leave, "ERS") == 0) return 5.5f;
        if (strcmp(sorted_leave, "ING") == 0) return 4.5f;
        if (strcmp(sorted_leave, "EST") == 0) return 5.0f;
        if (strcmp(sorted_leave, "RST") == 0) return 5.0f;
        if (strcmp(sorted_leave, "AER") == 0) return 4.5f;
        if (strcmp(sorted_leave, "AES") == 0) return 4.5f;
        if (strcmp(sorted_leave, "ENT") == 0) return 4.0f;
        if (strcmp(sorted_leave, "INT") == 0) return 4.0f;
        if (strcmp(sorted_leave, "TIN") == 0) return 4.0f;
        return 0.0f;
    }

    return 0.0f;
}

float eval_rack_leave_bucketed(const char *leave, PhaseBucket phase) {
    if (!leave) return 0.0f;
    int len = (int)strlen(leave);
    if (len == 0) return 0.0f;

    if (!synergy_matrix_initialized) init_synergy_matrix();

    int p_idx = (int)phase;
    if (p_idx < 0) p_idx = 0;
    if (p_idx > 7) p_idx = 7;

    float equity = 0.0f;
    int counts[27] = {0};
    int vowels = 0, consonants = 0, blanks = 0;
    int high_point_tiles = 0; /* F, H, V, W, Y, K, J, X, Q, Z */

    char sorted_leave[RACK_CAPACITY + 1];
    int slen = 0;

    /* 1. Accumulator A: Single Tile Values */
    for (int i = 0; i < len; i++) {
        char c = (char)toupper((unsigned char)leave[i]);
        if (c == '?') {
            counts[26]++;
            blanks++;
            equity += PHASE_TILE_EQUITY[p_idx][26];
            sorted_leave[slen++] = '?';
        } else if (c >= 'A' && c <= 'Z') {
            int idx = c - 'A';
            counts[idx]++;
            equity += PHASE_TILE_EQUITY[p_idx][idx];
            sorted_leave[slen++] = c;

            if (is_vowel_c(c)) {
                vowels++;
            } else {
                consonants++;
            }

            /* Track high value tile concentration */
            int val = TILE_VALUES[idx];
            if (val >= 4) {
                high_point_tiles++;
            }
        }
    }
    sorted_leave[slen] = '\0';

    /* Sort leave for stem lookup */
    for (int i = 0; i < slen - 1; i++) {
        for (int j = i + 1; j < slen; j++) {
            if (sorted_leave[i] > sorted_leave[j]) {
                char tmp = sorted_leave[i];
                sorted_leave[i] = sorted_leave[j];
                sorted_leave[j] = tmp;
            }
        }
    }

    /* 2. Accumulator B: Pairwise Letter Synergy & Duplicate Kernel */
    for (int i = 0; i < len; i++) {
        char c1 = (char)toupper((unsigned char)leave[i]);
        int idx1 = (c1 == '?') ? 26 : (c1 - 'A');
        for (int j = i + 1; j < len; j++) {
            char c2 = (char)toupper((unsigned char)leave[j]);
            int idx2 = (c2 == '?') ? 26 : (c2 - 'A');
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

float eval_board_risk(const Board *board, const Move *move) {
    OpennessBucket open = eval_classify_openness(board);
    return eval_board_risk_bucketed(board, move, open);
}

DualAccumulator eval_move_accumulate(const Board *board, const Move *move, int bag_count) {
    DualAccumulator acc;
    acc.board_acc = 0.0f;
    acc.rack_acc = 0.0f;
    acc.net_equity = 0.0f;

    if (move->type != MOVE_PLAY) return acc;

    PhaseBucket phase = eval_classify_phase(bag_count);
    OpennessBucket openness = eval_classify_openness(board);

    /* Hand-tuned Grandmaster Heuristic: board risk + 8-phase rack leave tables */
    float board_risk = eval_board_risk_bucketed(board, move, openness);
    float rack_leave = eval_rack_leave_bucketed(move->leave, phase);
    float hand_equity = (float)move->score + board_risk + (0.85f * rack_leave);

    /* If NNUE is loaded, combine Hand-Tuned Heuristic and Deep NNUE into a hybrid ensemble */
    if (g_nnue.is_loaded) {
        float nnue_val = nnue_evaluate_move(&g_nnue, board, move, NULL, bag_count);
        float hybrid = (0.50f * hand_equity) + (0.50f * nnue_val);
        acc.board_acc = (float)move->score + (0.50f * board_risk);
        acc.rack_acc = hybrid - acc.board_acc;
        acc.net_equity = hybrid;
        return acc;
    }

    acc.board_acc = (float)move->score + board_risk;
    acc.rack_acc = rack_leave;
    acc.net_equity = hand_equity;
    return acc;
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
    return board_acc + (0.85f * rack_acc);
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
