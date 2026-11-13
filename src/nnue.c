#include "nnue.h"
#include "eval.h"
#include "simulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdalign.h>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

NNUEModel g_nnue = {0};
NNUEQuantizedModel g_nnue_quantized = {0};

/* Random float in [-limit, +limit] */
static float rand_uniform(float limit) {
    float r = (float)rand() / (float)RAND_MAX;
    return (r * 2.0f - 1.0f) * limit;
}

void nnue_init_random(NNUEModel *model) {
    memset(model, 0, sizeof(NNUEModel));

    float limit1 = sqrtf(6.0f / (float)(NNUE_INPUT_DIM + NNUE_L1_DIM));
    for (int i = 0; i < NNUE_L1_DIM * NNUE_INPUT_DIM; i++) {
        model->w1[i] = rand_uniform(limit1);
    }
    for (int i = 0; i < NNUE_L1_DIM; i++) {
        model->b1[i] = 0.05f;
    }

    float limit2 = sqrtf(6.0f / (float)(NNUE_L1_ACT_DIM + NNUE_L2_DIM));
    for (int i = 0; i < NNUE_L2_DIM * NNUE_L1_ACT_DIM; i++) {
        model->w2[i] = rand_uniform(limit2);
    }
    for (int i = 0; i < NNUE_L2_DIM; i++) {
        model->b2[i] = 0.05f;
    }

    float limit3 = sqrtf(6.0f / (float)(NNUE_L2_DIM + NNUE_L3_DIM));
    for (int i = 0; i < NNUE_L3_DIM * NNUE_L2_DIM; i++) {
        model->w3[i] = rand_uniform(limit3);
    }
    for (int i = 0; i < NNUE_L3_DIM; i++) {
        model->b3[i] = 0.05f;
    }

    float limit_head = sqrtf(6.0f / (float)(NNUE_L3_DIM + 1));
    for (int i = 0; i < NNUE_L3_DIM; i++) {
        model->w_spread[i] = rand_uniform(limit_head);
        model->w_win[i] = rand_uniform(limit_head);
    }
    model->b_spread = 0.0f;
    model->b_win = 0.0f;
    model->is_loaded = true;

    for (int i = 0; i < NNUE_INPUT_DIM; i++) {
        for (int o = 0; o < NNUE_L1_DIM; o++) {
            model->w1_by_feature[i][o] = model->w1[o * NNUE_INPUT_DIM + i];
        }
    }

    nnue_quantize(model, &g_nnue_quantized);
}

void nnue_quantize(const NNUEModel *src, NNUEQuantizedModel *dst) {
    if (!src || !src->is_loaded || !dst) return;
    memset(dst, 0, sizeof(NNUEQuantizedModel));

    /* Find max absolute weight for L1 */
    float max_w1 = 1e-6f;
    for (int i = 0; i < NNUE_L1_DIM * NNUE_INPUT_DIM; i++) {
        float a = fabsf(src->w1[i]);
        if (a > max_w1) max_w1 = a;
    }
    dst->scale1 = 127.0f / max_w1;

    for (int i = 0; i < NNUE_L1_DIM * NNUE_INPUT_DIM; i++) {
        int val = (int)roundf(src->w1[i] * dst->scale1);
        if (val > 127) val = 127;
        if (val < -127) val = -127;
        dst->w1[i] = (int8_t)val;
    }
    for (int i = 0; i < NNUE_L1_DIM; i++) {
        int val = (int)roundf(src->b1[i] * dst->scale1);
        if (val > 32767) val = 32767;
        if (val < -32767) val = -32767;
        dst->b1[i] = (int16_t)val;
    }

    for (int i = 0; i < NNUE_INPUT_DIM; i++) {
        for (int o = 0; o < NNUE_L1_DIM; o++) {
            dst->w1_by_feature[i][o] = dst->w1[o * NNUE_INPUT_DIM + i];
        }
    }

    /* Find max absolute weight for L2 (64 x 512) */
    float max_w2 = 1e-6f;
    for (int i = 0; i < NNUE_L2_DIM * NNUE_L1_ACT_DIM; i++) {
        float a = fabsf(src->w2[i]);
        if (a > max_w2) max_w2 = a;
    }
    dst->scale2 = 127.0f / max_w2;

    for (int i = 0; i < NNUE_L2_DIM * NNUE_L1_ACT_DIM; i++) {
        int val = (int)roundf(src->w2[i] * dst->scale2);
        if (val > 127) val = 127;
        if (val < -127) val = -127;
        dst->w2[i] = (int8_t)val;
    }
    for (int i = 0; i < NNUE_L2_DIM; i++) {
        int val = (int)roundf(src->b2[i] * dst->scale2);
        if (val > 32767) val = 32767;
        if (val < -32767) val = -32767;
        dst->b2[i] = (int16_t)val;
    }

    /* Find max absolute weight for L3 (32 x 64) */
    float max_w3 = 1e-6f;
    for (int i = 0; i < NNUE_L3_DIM * NNUE_L2_DIM; i++) {
        float a = fabsf(src->w3[i]);
        if (a > max_w3) max_w3 = a;
    }
    dst->scale3 = 127.0f / max_w3;

    for (int i = 0; i < NNUE_L3_DIM * NNUE_L2_DIM; i++) {
        int val = (int)roundf(src->w3[i] * dst->scale3);
        if (val > 127) val = 127;
        if (val < -127) val = -127;
        dst->w3[i] = (int8_t)val;
    }
    for (int i = 0; i < NNUE_L3_DIM; i++) {
        int val = (int)roundf(src->b3[i] * dst->scale3);
        if (val > 32767) val = 32767;
        if (val < -32767) val = -32767;
        dst->b3[i] = (int16_t)val;
    }

    /* Score and Win head weights (32 -> 1) */
    for (int i = 0; i < NNUE_L3_DIM; i++) {
        dst->w_spread[i] = (int16_t)roundf(src->w_spread[i] * 512.0f);
        dst->w_win[i] = (int16_t)roundf(src->w_win[i] * 512.0f);
    }
    dst->b_spread = (int16_t)roundf(src->b_spread * 512.0f);
    dst->b_win = (int16_t)roundf(src->b_win * 512.0f);
    dst->is_loaded = true;
}

bool nnue_save(const NNUEModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (!f) return false;

    uint32_t header[6] = {
        NNUE_MAGIC,
        NNUE_VERSION,
        NNUE_INPUT_DIM,
        NNUE_L1_DIM,
        NNUE_L2_DIM,
        NNUE_L3_DIM
    };

    if (fwrite(header, sizeof(uint32_t), 6, f) != 6) {
        fclose(f);
        return false;
    }

    fwrite(model->w1, sizeof(float), NNUE_L1_DIM * NNUE_INPUT_DIM, f);
    fwrite(model->b1, sizeof(float), NNUE_L1_DIM, f);
    fwrite(model->w2, sizeof(float), NNUE_L2_DIM * NNUE_L1_ACT_DIM, f);
    fwrite(model->b2, sizeof(float), NNUE_L2_DIM, f);
    fwrite(model->w3, sizeof(float), NNUE_L3_DIM * NNUE_L2_DIM, f);
    fwrite(model->b3, sizeof(float), NNUE_L3_DIM, f);
    fwrite(model->w_spread, sizeof(float), NNUE_L3_DIM, f);
    fwrite(&model->b_spread, sizeof(float), 1, f);
    fwrite(model->w_win, sizeof(float), NNUE_L3_DIM, f);
    fwrite(&model->b_win, sizeof(float), 1, f);

    fclose(f);
    return true;
}

bool nnue_load(NNUEModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return false;

    uint32_t header[6];
    if (fread(header, sizeof(uint32_t), 6, f) != 6) {
        fclose(f);
        return false;
    }

    if (header[0] != NNUE_MAGIC || header[1] != NNUE_VERSION ||
        header[2] != NNUE_INPUT_DIM || header[3] != NNUE_L1_DIM ||
        header[4] != NNUE_L2_DIM || header[5] != NNUE_L3_DIM) {
        fclose(f);
        return false;
    }

    size_t r1 = fread(model->w1, sizeof(float), NNUE_L1_DIM * NNUE_INPUT_DIM, f);
    size_t r2 = fread(model->b1, sizeof(float), NNUE_L1_DIM, f);
    size_t r3 = fread(model->w2, sizeof(float), NNUE_L2_DIM * NNUE_L1_ACT_DIM, f);
    size_t r4 = fread(model->b2, sizeof(float), NNUE_L2_DIM, f);
    size_t r5 = fread(model->w3, sizeof(float), NNUE_L3_DIM * NNUE_L2_DIM, f);
    size_t r6 = fread(model->b3, sizeof(float), NNUE_L3_DIM, f);
    size_t r7 = fread(model->w_spread, sizeof(float), NNUE_L3_DIM, f);
    size_t r8 = fread(&model->b_spread, sizeof(float), 1, f);
    size_t r9 = fread(model->w_win, sizeof(float), NNUE_L3_DIM, f);
    size_t r10 = fread(&model->b_win, sizeof(float), 1, f);

    fclose(f);

    if (r1 != NNUE_L1_DIM * NNUE_INPUT_DIM || r2 != NNUE_L1_DIM ||
        r3 != NNUE_L2_DIM * NNUE_L1_ACT_DIM || r4 != NNUE_L2_DIM ||
        r5 != NNUE_L3_DIM * NNUE_L2_DIM || r6 != NNUE_L3_DIM ||
        r7 != NNUE_L3_DIM || r8 != 1 || r9 != NNUE_L3_DIM || r10 != 1) {
        return false;
    }

    model->is_loaded = true;
    for (int i = 0; i < NNUE_INPUT_DIM; i++) {
        for (int o = 0; o < NNUE_L1_DIM; o++) {
            model->w1_by_feature[i][o] = model->w1[o * NNUE_INPUT_DIM + i];
        }
    }
    nnue_quantize(model, &g_nnue_quantized);
    return true;
}

static inline bool is_vowel_char(char c) {
    c = (char)toupper((unsigned char)c);
    return (c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U');
}

void nnue_extract_features_with_score(const Board *board, const Move *move, const Rack *rack,
                                      int bag_count, int score_diff, float *out_features) {
    memset(out_features, 0, NNUE_INPUT_DIM * sizeof(float));
    if (!board) return;

    int leave_counts[27] = {0};
    int leave_len = 0;
    int blanks = 0, high_placed = 0;
    int vowels = 0, consonants = 0, high_points = 0, quad_points = 0;
    int duplicate_count = 0, distinct_letters = 0;
    int total_leave_points = 0;
    float tw_cnt = 0.0f, dw_cnt = 0.0f, tl_cnt = 0.0f, dl_cnt = 0.0f;
    int blanks_placed = 0;
    float center_dist = 0.0f;
    bool perimeter_touch = false;
    bool ideal_vc = false;
    int clumsy = 0;

    if (move) {
        /* 1. Move Placement Features (0..15: 16 features) */
        out_features[0] = (float)move->score / 50.0f;
        out_features[1] = (float)move->num_placed / 7.0f;
        out_features[2] = (move->num_placed == 7) ? 1.0f : 0.0f; /* Bingo */

    for (int p = 0; p < move->num_placed; p++) {
        int r = move->placed[p].row;
        int c = move->placed[p].col;
        MultiplierType m = board_get_multiplier(r, c);
        if (m == MULT_TW) tw_cnt += 1.0f;
        else if (m == MULT_DW) dw_cnt += 1.0f;
        else if (m == MULT_TL) tl_cnt += 1.0f;
        else if (m == MULT_DL) dl_cnt += 1.0f;

        if (move->placed[p].is_blank) blanks_placed++;
        char let = move->placed[p].letter;
        if (let >= 'A' && let <= 'Z' && TILE_VALUES[let - 'A'] >= 4) high_placed++;

        float dist = (float)(abs(r - 7) + abs(c - 7));
        if (dist > center_dist) center_dist = dist;

        if (r == 0 || r == 14 || c == 0 || c == 14) perimeter_touch = true;
    }

    out_features[3] = tw_cnt;
    out_features[4] = dw_cnt;
    out_features[5] = tl_cnt;
    out_features[6] = dl_cnt;
    out_features[7] = (float)strlen(move->word) / 15.0f;
    out_features[8] = (move->dir == DIR_VERTICAL) ? 1.0f : 0.0f;
    out_features[9] = center_dist / 14.0f;
    out_features[10] = perimeter_touch ? 1.0f : 0.0f;
    out_features[11] = (float)blanks_placed;
    out_features[12] = (float)high_placed / 3.0f;
    out_features[13] = (float)move->score / (float)(move->num_placed > 0 ? move->num_placed : 1) / 20.0f;
    out_features[14] = (tw_cnt > 0.0f && dw_cnt > 0.0f) ? 1.0f : 0.0f;
    out_features[15] = (move->type == MOVE_EXCHANGE) ? 1.0f : 0.0f;

    /* 2. Leave Tile Counts (16..42 for 27 tiles: 0..25=A..Z, 26=?) */
    leave_len = (int)strlen(move->leave);

    for (int i = 0; i < leave_len; i++) {
        char c = move->leave[i];
        if (c == '?') {
            leave_counts[26]++;
            blanks++;
        } else if (c >= 'A' && c <= 'Z') {
            int idx = c - 'A';
            if (leave_counts[idx] == 0) distinct_letters++;
            else duplicate_count++;
            leave_counts[idx]++;

            if (is_vowel_char(c)) vowels++;
            else consonants++;

            int val = TILE_VALUES[idx];
            total_leave_points += val;
            if (val >= 4) high_points++;
            if (val >= 8) quad_points++;
        }
    }

    for (int i = 0; i < 27; i++) {
        out_features[16 + i] = (float)leave_counts[i];
    }

    /* 3. Pairwise Leave Interactions (43..420: 378 pairs) */
    int pair_idx = 0;
    for (int i = 0; i < 27; i++) {
        for (int j = i; j < 27; j++) {
            if (pair_idx < 378) {
                if (i == j) {
                    if (leave_counts[i] >= 2) out_features[43 + pair_idx] = 1.0f;
                } else {
                    if (leave_counts[i] >= 1 && leave_counts[j] >= 1) {
                        out_features[43 + pair_idx] = 1.0f;
                    }
                }
            }
            pair_idx++;
        }
    }

    /* 4. Tri-Tile Leave Synergies (421..484: 64 triplets) */
    static const char *triplets[64] = {
        "ING", "ERS", "EST", "IER", "IES", "ION", "IST", "ITY", "ANT", "ENT",
        "ATE", "IVE", "OUS", "FUL", "LES", "NES", "RED", "TED", "CON", "DIS",
        "MIS", "PRE", "PRO", "SUB", "RET", "TIS", "RIN", "SIN", "TIN", "AIN",
        "EIN", "OIN", "RES", "TES", "DES", "SES", "STA", "STR", "ADE", "AGE",
        "ALE", "ANE", "ARE", "ASE", "AVE", "EAR", "EAT", "END", "ERA", "ERR",
        "ETA", "IDE", "INE", "OAR", "ONE", "ORE", "ORT", "AIR", "AIS", "AIT",
        "ELS", "ELT", "ENS", "ENT"
    };

    for (int t = 0; t < 64; t++) {
        int req[27] = {0};
        const char *trip = triplets[t];
        for (int k = 0; k < 3; k++) req[trip[k] - 'A']++;
        bool ok = true;
        for (int k = 0; k < 26; k++) {
            if (req[k] > leave_counts[k]) { ok = false; break; }
        }
        if (ok) out_features[421 + t] = 1.0f;
    }

    /* 5. Macro Rack Composition & Dynamic Equity (485..504: 20 features) */
    out_features[485] = (float)vowels / 5.0f;
    out_features[486] = (float)consonants / 5.0f;
    out_features[487] = ((float)vowels - (float)consonants) / 5.0f;
    out_features[488] = (float)blanks;
    out_features[489] = (float)high_points / 3.0f;
    out_features[490] = (float)quad_points;
    out_features[491] = (float)duplicate_count / 3.0f;
    out_features[492] = (duplicate_count >= 2) ? 1.0f : 0.0f;
    out_features[493] = (float)distinct_letters / 7.0f;
    out_features[494] = (float)total_leave_points / 30.0f;
    float v_ratio = (vowels + consonants > 0) ? (float)vowels / (float)(vowels + consonants) : 0.5f;
    out_features[495] = v_ratio;
    ideal_vc = (vowels == 2 && consonants == 3) || (vowels == 3 && consonants == 2) ||
               (vowels == 2 && consonants == 2) || (vowels == 3 && consonants == 3);
    out_features[496] = ideal_vc ? 1.0f : 0.0f;
    out_features[497] = (leave_len >= 3 && vowels == 0) ? 1.0f : 0.0f;
    out_features[498] = (leave_len >= 3 && consonants == 0) ? 1.0f : 0.0f;
    out_features[499] = (leave_counts['Q' - 'A'] > 0 && leave_counts['U' - 'A'] == 0) ? 1.0f : 0.0f;
    float unplayable_penalty = (leave_counts['Q' - 'A'] * 10.0f + leave_counts['V' - 'A'] * 5.0f) / 25.0f;
    out_features[500] = unplayable_penalty;
    out_features[501] = (float)leave_counts['S' - 'A'];
    clumsy = leave_counts['V' - 'A'] + leave_counts['W' - 'A'] + leave_counts['Y' - 'A'];
    out_features[502] = (float)clumsy / 3.0f;
    int liquid = leave_counts['L' - 'A'] + leave_counts['R' - 'A'] + leave_counts['S' - 'A'] + leave_counts['T' - 'A'];
    out_features[503] = (float)liquid / 4.0f;
    out_features[504] = (float)leave_len / 7.0f;

    /* 6. Championship Bingo Stem Indicators (505..536: 32 stems) */
    static const char *stems[32] = {
        "AEINST", "AEINRT", "ADENRS", "AELRST", "AEORST", "AENRST",
        "EINST",  "EINRT",  "DENRS",  "ELRST",  "EORST",  "ENRST",
        "AEIN",   "AEIR",   "AEIS",   "AEIT",   "AELN",   "AELR",
        "AELS",   "AELT",   "AERS",   "AERT",   "EILR",   "EILS",
        "EILT",   "EINS",   "EINT",   "EIRS",   "EIRT",   "ENRS",
        "ENRT",   "ERST"
    };

        for (int s = 0; s < 32; s++) {
            bool match = true;
            for (const char *p = stems[s]; *p; p++) {
                if (leave_counts[*p - 'A'] < 1) { match = false; break; }
            }
            if (match) out_features[505 + s] = 1.0f;
        }
    }

    /* 7. Board Exposure & Hazard Features (537..584: 48 features) */
    int open_tw = bb_popcount_and(&board->anchors, &BB_TW);
    int open_dw = bb_popcount_and(&board->anchors, &BB_DW);
    int open_tl = bb_popcount_and(&board->anchors, &BB_TL);
    int open_dl = bb_popcount_and(&board->anchors, &BB_DL);
    int hot_anchors = open_tw + open_dw + open_tl + open_dl;
    int total_anchors = bb_popcount(&board->anchors);
    int total_occupied = bb_popcount(&board->occupied);
    int exposed_tw_vowels = 0;

    int nw_occ = 0, ne_occ = 0, sw_occ = 0, se_occ = 0;
    int perimeter_occ = 0, center_occ = 0;

    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (bb_test(&board->occupied, r, c)) {
                if (r < 7 && c < 7) nw_occ++;
                else if (r < 7 && c > 7) ne_occ++;
                else if (r > 7 && c < 7) sw_occ++;
                else if (r > 7 && c > 7) se_occ++;

                if (r == 0 || r == 14 || c == 0 || c == 14) perimeter_occ++;
                if (r >= 5 && r <= 9 && c >= 5 && c <= 9) center_occ++;
            }
        }
    }

    int floating_tiles = 0;
    if (move) {
        for (int p = 0; p < move->num_placed; p++) {
            int r = move->placed[p].row;
            int c = move->placed[p].col;
            int dr[4] = {-1, 1, 0, 0};
            int dc[4] = {0, 0, -1, 1};
            bool has_perp = false;
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d];
                int nc = c + dc[d];
                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    if (board->cells[nr][nc].tile != 0) has_perp = true;
                    if (is_vowel_char(move->placed[p].letter) &&
                        board->cells[nr][nc].tile == 0 && board_get_multiplier(nr, nc) == MULT_TW) {
                        exposed_tw_vowels++;
                    }
                }
            }
            if (!has_perp) floating_tiles++;
        }
    }

    out_features[537] = (float)open_tw / 8.0f;
    out_features[538] = (float)open_dw / 16.0f;
    out_features[539] = (float)open_tl / 12.0f;
    out_features[540] = (float)open_dl / 24.0f;
    out_features[541] = (float)exposed_tw_vowels;
    out_features[542] = (float)total_occupied / 100.0f;

    OpennessBucket openness = eval_classify_openness(board);
    if ((int)openness >= 0 && (int)openness < 4) out_features[543 + (int)openness] = 1.0f;

    out_features[547] = (float)nw_occ / 25.0f;
    out_features[548] = (float)ne_occ / 25.0f;
    out_features[549] = (float)sw_occ / 25.0f;
    out_features[550] = (float)se_occ / 25.0f;
    out_features[551] = (float)perimeter_occ / 56.0f;
    out_features[552] = (float)center_occ / 25.0f;
    out_features[553] = (float)(open_tw) / 4.0f;
    out_features[554] = (float)(open_dw) / 8.0f;
    out_features[555] = (float)hot_anchors / 10.0f;
    out_features[556] = (float)total_anchors / 30.0f;
    out_features[557] = (float)floating_tiles / 7.0f;
    out_features[558] = fabsf((float)(nw_occ + sw_occ) - (float)(ne_occ + se_occ)) / 50.0f;
    out_features[559] = fabsf((float)(nw_occ + ne_occ) - (float)(sw_occ + se_occ)) / 50.0f;
    out_features[560] = fabsf((float)nw_occ - (float)se_occ) / 25.0f;

    /* 8. Unseen Bag & Bayesian Opponent Modeling (585..648: 64 features) */
    out_features[585] = (float)bag_count / 100.0f;
    out_features[586] = (bag_count == 0) ? 1.0f : 0.0f;
    out_features[587] = (bag_count <= 10) ? 1.0f : 0.0f;
    out_features[588] = (bag_count <= 20) ? 1.0f : 0.0f;

    char unseen[100];
    int unseen_count = sim_get_unseen_tiles(board, rack ? rack : &(Rack){0}, unseen);
    int u_vowels = 0, u_consonants = 0, u_blanks = 0, u_s = 0, u_quad = 0;
    int u_counts[27] = {0};

    for (int i = 0; i < unseen_count; i++) {
        char t = unseen[i];
        if (t == '?') { u_blanks++; u_counts[26]++; }
        else if (t >= 'A' && t <= 'Z') {
            int idx = t - 'A';
            u_counts[idx]++;
            if (is_vowel_char(t)) u_vowels++;
            else u_consonants++;
            if (t == 'S') u_s++;
            if (TILE_VALUES[idx] >= 8) u_quad++;
        }
    }

    float u_denom = unseen_count > 0 ? (float)unseen_count : 1.0f;
    out_features[589] = (float)u_vowels / u_denom;
    out_features[590] = (float)u_consonants / u_denom;
    out_features[591] = (float)u_blanks / 2.0f;
    out_features[592] = (float)u_s / 4.0f;
    out_features[593] = (float)u_quad / 4.0f;

    for (int i = 0; i < 27; i++) {
        out_features[594 + i] = (float)u_counts[i] / u_denom;
    }

    /* Opponent hoarding indicators for key letters */
    out_features[622] = (u_counts['Q' - 'A'] > 0) ? 1.0f : 0.0f;
    out_features[623] = (u_counts['Z' - 'A'] > 0) ? 1.0f : 0.0f;
    out_features[624] = (u_counts['J' - 'A'] > 0) ? 1.0f : 0.0f;
    out_features[625] = (u_counts['X' - 'A'] > 0) ? 1.0f : 0.0f;
    out_features[626] = (u_counts['V' - 'A'] > 0) ? 1.0f : 0.0f;

    /* 9. Live Game Phase, Score Differential & Tournament Equity (649..700: 52 features) */
    PhaseBucket phase = eval_classify_phase(bag_count);
    int p_idx = (int)phase;
    if (p_idx >= 0 && p_idx < 8) out_features[649 + p_idx] = 1.0f;

    out_features[657] = tanhf((float)score_diff / 50.0f);
    out_features[658] = fminf(1.0f, fmaxf(-1.0f, (float)score_diff / 100.0f));
    out_features[659] = (score_diff >= 40) ? 1.0f : 0.0f;
    out_features[660] = (score_diff <= -40) ? 1.0f : 0.0f;
    out_features[661] = eval_win_probability(score_diff, bag_count);
    out_features[662] = eval_score_margin_risk_multiplier(score_diff);
    out_features[663] = (fabsf((float)score_diff) <= 15.0f) ? 1.0f : 0.0f;

    /* 10. Cross-Check Hooks, Leave-Board Synergy & Lane Dynamics (701..868) */
    if (move) {
        out_features[701] = (leave_counts['S' - 'A'] > 0 && total_anchors > 5) ? 1.0f : 0.0f;
        out_features[702] = (leave_counts['D' - 'A'] > 0 && total_anchors > 5) ? 1.0f : 0.0f;
        out_features[703] = (leave_counts['R' - 'A'] > 0 && leave_counts['E' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[704] = (blanks > 0 && move->score < 35) ? -1.0f : 0.0f; /* Blank squandering penalty */
        out_features[705] = (leave_counts['S' - 'A'] > 0 && move->score < 25) ? -0.5f : 0.0f; /* S squandering */
        out_features[706] = (high_placed > 0 && move->score >= 35) ? 1.0f : 0.0f; /* High-value tile monetization */
        out_features[707] = (move->num_placed == 7 && score_diff < 0) ? 1.0f : 0.0f; /* Deficit bingo comeback */
        out_features[708] = (leave_len == 0) ? 1.0f : 0.0f; /* Clean empty leave */

        /* 11. Extended 4-Tile Bingo Stems (709..772: 64 stems) */
        static const char *stems4[64] = {
            "ACER", "ACRS", "ADEL", "ADER", "ADES", "ADIS", "AELP", "AELS", "AELT", "AEMN",
            "AENR", "AENS", "AEPS", "AERS", "AEST", "AILS", "AILV", "AINR", "AINS", "AINT",
            "AIRS", "AIRT", "AIST", "ALMN", "ALNR", "ALNS", "ALOS", "ALRT", "ALST", "ANOR",
            "ANRS", "ANST", "AORT", "ARST", "CDEN", "CINT", "DEIN", "DEIR", "DEIS", "DEIT",
            "DELN", "DELR", "DELS", "DENR", "DENS", "DEOR", "DEOS", "DERT", "DEST", "EILN",
            "EILR", "EILS", "EILT", "EIMN", "EIMR", "EIMS", "EINP", "EINR", "EINS", "EINT",
            "EIPS", "EIPR", "EIRT", "EIST"
        };
        for (int s = 0; s < 64; s++) {
            bool ok = true;
            for (int k = 0; k < 4; k++) {
                if (leave_counts[stems4[s][k] - 'A'] < 1) { ok = false; break; }
            }
            if (ok) out_features[709 + s] = 1.0f;
        }

        /* 12. High-Equity 2-Tile Leave Synergies (773..820: 48 synergies) */
        static const char *pairs2[48] = {
            "ER", "IN", "ES", "RE", "AN", "EN", "TI", "ST", "ON", "OR",
            "AT", "TE", "AL", "AR", "IS", "IT", "ED", "NT", "RA", "TO",
            "ET", "SE", "NE", "LI", "LE", "RO", "ND", "RI", "NG", "DE",
            "ME", "SA", "EL", "TA", "SI", "TH", "LO", "DI", "CO", "MA",
            "CA", "LA", "NA", "PE", "PR", "TR", "PL", "CL"
        };
        for (int p = 0; p < 48; p++) {
            char c0 = pairs2[p][0];
            char c1 = pairs2[p][1];
            if (c0 == c1) {
                if (leave_counts[c0 - 'A'] >= 2) out_features[773 + p] = 1.0f;
            } else {
                if (leave_counts[c0 - 'A'] >= 1 && leave_counts[c1 - 'A'] >= 1) out_features[773 + p] = 1.0f;
            }
        }

        /* 13. Fine-Grained Duplicate Penalties & Rack Micro-Distributions (821..868: 48 features) */
        for (int c = 0; c < 26; c++) {
            if (leave_counts[c] >= 2) {
                out_features[821 + c] = (float)(leave_counts[c] - 1);
            }
        }
        bool trip_dup = false;
        for (int c = 0; c < 26; c++) {
            if (leave_counts[c] >= 3) { trip_dup = true; break; }
        }
        out_features[847] = trip_dup ? 1.0f : 0.0f;
        out_features[848] = (vowels == 0) ? 1.0f : 0.0f;
        out_features[849] = (vowels == 1) ? 1.0f : 0.0f;
        out_features[850] = (vowels == 2) ? 1.0f : 0.0f;
        out_features[851] = (vowels >= 3) ? 1.0f : 0.0f;
        out_features[852] = (consonants == 0) ? 1.0f : 0.0f;
        out_features[853] = (consonants == 1) ? 1.0f : 0.0f;
        out_features[854] = (consonants == 2) ? 1.0f : 0.0f;
        out_features[855] = (consonants >= 3) ? 1.0f : 0.0f;
        out_features[856] = (blanks > 0 && leave_counts['S' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[857] = (leave_counts['S' - 'A'] >= 2) ? 1.0f : 0.0f;
        out_features[858] = (blanks >= 2) ? 1.0f : 0.0f;
        out_features[859] = (leave_counts['Q' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[860] = (leave_counts['Z' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[861] = (leave_counts['J' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[862] = (leave_counts['X' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[863] = (leave_counts['K' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[864] = (leave_counts['V' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[865] = (leave_counts['W' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[866] = (leave_counts['Y' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[867] = (leave_counts['B' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[868] = (leave_counts['P' - 'A'] > 0) ? 1.0f : 0.0f;
    }

    /* 14. Static Board Lane Openness & Hook Exposure (869..878: 10 features) */
    int r0_occ = 0, r14_occ = 0, c0_occ = 0, c14_occ = 0, r7_occ = 0, c7_occ = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (bb_test(&board->occupied, 0, i)) r0_occ++;
        if (bb_test(&board->occupied, 14, i)) r14_occ++;
        if (bb_test(&board->occupied, i, 0)) c0_occ++;
        if (bb_test(&board->occupied, i, 14)) c14_occ++;
        if (bb_test(&board->occupied, 7, i)) r7_occ++;
        if (bb_test(&board->occupied, i, 7)) c7_occ++;
    }
    int nw_anchors = 0, ne_anchors = 0, sw_anchors = 0, se_anchors = 0;
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (bb_test(&board->anchors, r, c)) {
                if (r < 7 && c < 7) nw_anchors++;
                else if (r < 7 && c > 7) ne_anchors++;
                else if (r > 7 && c < 7) sw_anchors++;
                else if (r > 7 && c > 7) se_anchors++;
            }
        }
    }
    out_features[869] = (float)r0_occ / 15.0f;
    out_features[870] = (float)r14_occ / 15.0f;
    out_features[871] = (float)c0_occ / 15.0f;
    out_features[872] = (float)c14_occ / 15.0f;
    out_features[873] = (float)r7_occ / 15.0f;
    out_features[874] = (float)c7_occ / 15.0f;
    out_features[875] = (float)nw_anchors / 10.0f;
    out_features[876] = (float)ne_anchors / 10.0f;
    out_features[877] = (float)sw_anchors / 10.0f;
    out_features[878] = (float)se_anchors / 10.0f;

    /* 15. Static Pre-Endgame & Bag Countdown Dynamics (930..950: 21 features) */
    out_features[930] = (bag_count >= 1 && bag_count <= 7) ? 1.0f : 0.0f;
    out_features[931] = (bag_count >= 8 && bag_count <= 14) ? 1.0f : 0.0f;
    out_features[932] = (bag_count >= 15 && bag_count <= 21) ? 1.0f : 0.0f;
    out_features[933] = (bag_count == 0) ? 1.0f : 0.0f;
    out_features[934] = (float)u_counts['Q' - 'A'];
    out_features[935] = (float)u_counts['Z' - 'A'];
    out_features[936] = (float)u_counts['X' - 'A'];
    out_features[937] = (float)u_counts['J' - 'A'];
    out_features[938] = (float)u_s / 4.0f;
    out_features[939] = (float)u_blanks / 2.0f;
    out_features[940] = (float)u_counts['C' - 'A'] / 2.0f;
    out_features[941] = (float)u_counts['V' - 'A'] / 2.0f;
    out_features[942] = (u_vowels + u_consonants > 0) ? (float)u_vowels / (float)(u_vowels + u_consonants) : 0.5f;
    out_features[943] = (float)abs(u_vowels - u_consonants) / (u_denom > 0.0f ? u_denom : 1.0f);
    out_features[944] = (bag_count <= 14 && u_blanks > 0) ? 1.0f : 0.0f;
    out_features[945] = (bag_count <= 14 && u_s > 0) ? 1.0f : 0.0f;
    out_features[946] = (100.0f - (float)bag_count) / 100.0f;
    out_features[947] = (bag_count < 7) ? (float)(7 - bag_count) / 7.0f : 0.0f;
    out_features[948] = (bag_count < 14) ? (float)(14 - bag_count) / 14.0f : 0.0f;
    out_features[949] = (bag_count > 0 && bag_count <= 4) ? 1.0f : 0.0f;
    out_features[950] = (u_counts['Q' - 'A'] > 0 && u_counts['U' - 'A'] == 0) ? 1.0f : 0.0f;

    /* 16. Static Score Differential Margin Buckets (965..973: 9 features) */
    out_features[965] = (score_diff >= 80) ? 1.0f : 0.0f;
    out_features[966] = (score_diff >= 40 && score_diff < 80) ? 1.0f : 0.0f;
    out_features[967] = (score_diff >= 15 && score_diff < 40) ? 1.0f : 0.0f;
    out_features[968] = (score_diff >= -14 && score_diff < 15) ? 1.0f : 0.0f;
    out_features[969] = (score_diff >= -39 && score_diff < -14) ? 1.0f : 0.0f;
    out_features[970] = (score_diff >= -79 && score_diff < -39) ? 1.0f : 0.0f;
    out_features[971] = (score_diff < -79) ? 1.0f : 0.0f;
    out_features[972] = tanhf(((float)score_diff - 40.0f) / 30.0f);
    out_features[973] = (score_diff > 30 && bag_count < 25) ? 1.0f : 0.0f;

    /* 17. Static Tournament Game-Theoretic Terms (991..1023: 33 features) */
    out_features[991] = (bag_count == 0) ? 1.0f : (bag_count <= 14 ? (14.0f - (float)bag_count) / 14.0f : 0.0f);
    out_features[992] = ((bag_count + 14) % 2 == 0) ? 1.0f : 0.0f;
    out_features[993] = (bag_count % 2 == 1) ? 1.0f : 0.0f;
    out_features[994] = (float)(u_counts['Q'-'A'] + u_counts['Z'-'A'] + u_counts['J'-'A'] + u_counts['X'-'A']) / 4.0f;
    out_features[995] = (u_vowels + u_consonants > 0) ? fabsf((float)u_vowels / (float)(u_vowels + u_consonants) - 0.42f) : 0.0f;
    out_features[996] = (total_anchors > 0) ? (float)hot_anchors / (float)total_anchors : 0.0f;
    float sum_r = 0.0f, sum_c = 0.0f;
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (bb_test(&board->occupied, r, c)) {
                sum_r += (float)r;
                sum_c += (float)c;
            }
        }
    }
    float cm_r = (total_occupied > 0) ? (sum_r / (float)total_occupied) : 7.0f;
    float cm_c = (total_occupied > 0) ? (sum_c / (float)total_occupied) : 7.0f;
    out_features[997] = cm_r / 14.0f;
    out_features[998] = cm_c / 14.0f;
    out_features[999] = (fabsf(cm_r - 7.0f) + fabsf(cm_c - 7.0f)) / 14.0f;

    out_features[1000] = (float)nw_occ / 25.0f;
    out_features[1001] = (float)ne_occ / 25.0f;
    out_features[1002] = (float)sw_occ / 25.0f;
    out_features[1003] = (float)se_occ / 25.0f;
    out_features[1004] = (float)(nw_occ + se_occ) / 50.0f;
    out_features[1005] = (float)(ne_occ + sw_occ) / 50.0f;
    out_features[1006] = (float)perimeter_occ / 56.0f;
    out_features[1007] = (float)center_occ / 25.0f;

    for (int p = 0; p < 8; p++) {
        out_features[1008 + p] = (phase == (PhaseBucket)p) ? tanhf((float)score_diff / 50.0f) : 0.0f;
    }

    out_features[1016] = (float)phase / 7.0f * (float)openness / 3.0f;
    out_features[1017] = (float)bag_count / 100.0f * (float)total_occupied / 100.0f;
    out_features[1018] = (float)hot_anchors / 10.0f * (float)open_tw / 8.0f;
    out_features[1019] = (float)u_blanks / 2.0f * (float)u_s / 4.0f;
    out_features[1020] = (score_diff > 0 ? 1.0f : -1.0f) * ((100.0f - (float)bag_count) / 100.0f);
    out_features[1021] = tanhf((float)score_diff / 100.0f) * ((float)total_occupied / 100.0f);
    out_features[1022] = (open_tw == 0) ? 1.0f : 0.0f;
    out_features[1023] = (open_dw == 0) ? 1.0f : 0.0f;

    /* 18. Move Dynamic Lane, Power & Tactical Features (879..929, 951..964, 974..990) */
    if (move) {
        bool touch_r0 = false, touch_r14 = false, touch_c0 = false, touch_c14 = false;
        bool touch_r7 = false, touch_c7 = false;
        bool crosses_tw_lane = false, crosses_dw_lane = false;
        int exp_empty_tw = 0, exp_empty_dw = 0, exp_empty_tl = 0, exp_empty_dl = 0;
        int exp_tw_consonants = 0, exp_tw_vowels = 0;
        int exp_dw_vowels = 0, exp_dw_consonants = 0;
        int min_r = 15, max_r = -1, min_c = 15, max_c = -1;
        int placed_on_bonus = 0;

        for (int p = 0; p < move->num_placed; p++) {
            int r = move->placed[p].row;
            int c = move->placed[p].col;
            if (r == 0) touch_r0 = true;
            if (r == 14) touch_r14 = true;
            if (c == 0) touch_c0 = true;
            if (c == 14) touch_c14 = true;
            if (r == 7) touch_r7 = true;
            if (c == 7) touch_c7 = true;
            if (r == 0 || r == 7 || r == 14) crosses_tw_lane = true;
            if (c == 0 || c == 7 || c == 14) crosses_tw_lane = true;
            if (r == c || r + c == 14) crosses_dw_lane = true;

            if (r < min_r) min_r = r;
            if (r > max_r) max_r = r;
            if (c < min_c) min_c = c;
            if (c > max_c) max_c = c;

            MultiplierType m = board_get_multiplier(r, c);
            if (m != MULT_NONE) placed_on_bonus++;

            char ch = move->placed[p].letter;
            bool is_v = is_vowel_char(ch);

            int dr[4] = {-1, 1, 0, 0};
            int dc[4] = {0, 0, -1, 1};
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d];
                int nc = c + dc[d];
                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    if (board->cells[nr][nc].tile == 0) {
                        MultiplierType nm = board_get_multiplier(nr, nc);
                        if (nm == MULT_TW) {
                            exp_empty_tw++;
                            if (is_v) exp_tw_vowels++;
                            else exp_tw_consonants++;
                        } else if (nm == MULT_DW) {
                            exp_empty_dw++;
                            if (is_v) exp_dw_vowels++;
                            else exp_dw_consonants++;
                        } else if (nm == MULT_TL) {
                            exp_empty_tl++;
                        } else if (nm == MULT_DL) {
                            exp_empty_dl++;
                        }
                    }
                }
            }
        }

        out_features[879] = touch_r0 ? 1.0f : 0.0f;
        out_features[880] = touch_r14 ? 1.0f : 0.0f;
        out_features[881] = touch_c0 ? 1.0f : 0.0f;
        out_features[882] = touch_c14 ? 1.0f : 0.0f;
        out_features[883] = touch_r7 ? 1.0f : 0.0f;
        out_features[884] = touch_c7 ? 1.0f : 0.0f;
        out_features[885] = crosses_tw_lane ? 1.0f : 0.0f;
        out_features[886] = crosses_dw_lane ? 1.0f : 0.0f;
        out_features[887] = (float)exp_empty_tw / 4.0f;
        out_features[888] = (float)exp_empty_dw / 8.0f;
        out_features[889] = (float)exp_empty_tl / 6.0f;
        out_features[890] = (float)exp_empty_dl / 12.0f;
        out_features[891] = (float)exp_tw_consonants / 2.0f;
        out_features[892] = (float)exp_tw_vowels / 2.0f;
        out_features[893] = (float)exp_dw_vowels / 4.0f;
        out_features[894] = (float)exp_dw_consonants / 4.0f;
        out_features[895] = (min_r >= 0 && min_r <= 14) ? (float)min_r / 14.0f : 0.5f;
        out_features[896] = (min_c >= 0 && min_c <= 14) ? (float)min_c / 14.0f : 0.5f;
        int span = (max_r >= min_r && max_c >= min_c) ? (max_r - min_r + max_c - min_c + 1) : 0;
        out_features[897] = (float)span / 15.0f;
        out_features[898] = (move->num_placed == 1) ? 1.0f : 0.0f;
        out_features[899] = (move->col > 0 && board->cells[move->row][move->col - 1].tile != 0) ? 1.0f : 0.0f;
        out_features[900] = (move->dir == DIR_HORIZONTAL && move->col + (int)strlen(move->word) < BOARD_SIZE &&
                             board->cells[move->row][move->col + (int)strlen(move->word)].tile != 0) ? 1.0f : 0.0f;
        int num_cross = (move->num_placed > 1) ? (move->num_placed - 1) : 0;
        out_features[901] = (num_cross >= 2) ? 1.0f : 0.0f;
        out_features[902] = (float)num_cross / 5.0f;
        out_features[903] = (float)move->score / 50.0f;
        out_features[904] = (float)move->score / (float)(move->num_placed > 0 ? move->num_placed : 1) / 10.0f;
        out_features[905] = (move->score > 0) ? (float)placed_on_bonus / (float)(move->num_placed > 0 ? move->num_placed : 1) : 0.0f;
        out_features[906] = (placed_on_bonus >= 2) ? 1.0f : 0.0f;
        out_features[907] = (float)move->num_placed / 7.0f;
        out_features[908] = (float)exp_empty_tw;
        out_features[909] = (float)(exp_empty_tw - (tw_cnt > 0.0f ? 1 : 0));
        out_features[910] = (touch_r0 || touch_r14 || touch_c0 || touch_c14) ? 1.0f : 0.0f;
        out_features[911] = (exp_empty_tw + exp_empty_dw >= 2) ? 1.0f : 0.0f;
        out_features[912] = (exp_empty_tw + exp_empty_dw == 0) ? 1.0f : 0.0f;
        out_features[913] = (crosses_tw_lane && move->num_placed >= 4) ? 1.0f : 0.0f;
        out_features[914] = (center_dist <= 3.0f) ? 1.0f : 0.0f;
        out_features[915] = (center_dist >= 8.0f) ? 1.0f : 0.0f;

        bool touches_power = false;
        bool s_placed = false;
        bool q_placed = false, z_placed = false, x_placed = false, j_placed = false;
        bool high_on_bonus = false;
        for (int p = 0; p < move->num_placed; p++) {
            int r = move->placed[p].row;
            int c = move->placed[p].col;
            char let = move->placed[p].letter;
            if (let == 'S') s_placed = true;
            MultiplierType m = board_get_multiplier(r, c);
            bool is_b = (m != MULT_NONE);
            if (let == 'Q') { q_placed = true; if (is_b) high_on_bonus = true; }
            if (let == 'Z') { z_placed = true; if (is_b) high_on_bonus = true; }
            if (let == 'X') { x_placed = true; if (is_b) high_on_bonus = true; }
            if (let == 'J') { j_placed = true; if (is_b) high_on_bonus = true; }

            int dr[4] = {-1, 1, 0, 0};
            int dc[4] = {0, 0, -1, 1};
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d];
                int nc = c + dc[d];
                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    char t = board->cells[nr][nc].tile;
                    if (t == 'Q' || t == 'Z' || t == 'X' || t == 'J') touches_power = true;
                }
            }
        }
        out_features[916] = touches_power ? 1.0f : 0.0f;

        /* Power tile management (917..929) */
        out_features[917] = (blanks_placed > 0 && move->num_placed < 7) ? 1.0f : 0.0f;
        out_features[918] = (blanks_placed > 0 && move->num_placed == 7) ? 1.0f : 0.0f;
        out_features[919] = (blanks > 0) ? 1.0f : 0.0f;
        out_features[920] = (s_placed && move->num_placed < 7) ? 1.0f : 0.0f;
        out_features[921] = (s_placed && move->num_placed == 7) ? 1.0f : 0.0f;
        out_features[922] = (leave_counts['S' - 'A'] > 0) ? 1.0f : 0.0f;
        out_features[923] = (q_placed && high_on_bonus) ? 1.0f : 0.0f;
        out_features[924] = (z_placed && high_on_bonus) ? 1.0f : 0.0f;
        out_features[925] = (x_placed && high_on_bonus) ? 1.0f : 0.0f;
        out_features[926] = (j_placed && high_on_bonus) ? 1.0f : 0.0f;
        out_features[927] = (q_placed && move->score < 20) ? 1.0f : 0.0f;
        out_features[928] = (high_placed > 0 && move->score >= 35) ? 1.0f : 0.0f;
        out_features[929] = (clumsy > 0 && (leave_counts['Q'-'A'] > 0 || leave_counts['Z'-'A'] > 0)) ? 1.0f : 0.0f;

        /* Move tactical synergies & clean leaves (951..964) */
        bool perfect_leave = (strcmp(move->leave, "ER") == 0 || strcmp(move->leave, "IN") == 0 ||
                              strcmp(move->leave, "ST") == 0 || strcmp(move->leave, "RE") == 0 ||
                              strcmp(move->leave, "ES") == 0);
        out_features[951] = perfect_leave ? 1.0f : 0.0f;
        out_features[952] = (move->score >= 40 && move->num_placed <= 4) ? 1.0f : 0.0f;
        out_features[953] = (move->num_placed > 0 && (float)move->score / (float)move->num_placed >= 8.0f) ? 1.0f : 0.0f;
        out_features[954] = (move->num_placed == 1 && (q_placed || z_placed || x_placed || j_placed)) ? 1.0f : 0.0f;
        out_features[955] = (move->type == MOVE_EXCHANGE && move->num_placed >= 5) ? 1.0f : 0.0f;
        int placed_vowels = 0, placed_consonants = 0;
        for (int p = 0; p < move->num_placed; p++) {
            if (is_vowel_char(move->placed[p].letter)) placed_vowels++;
            else placed_consonants++;
        }
        out_features[956] = (placed_vowels >= 3 && ideal_vc) ? 1.0f : 0.0f;
        out_features[957] = (placed_consonants >= 4 && ideal_vc) ? 1.0f : 0.0f;
        out_features[958] = (move->num_placed == 1 && move->score >= 20) ? 1.0f : 0.0f;
        out_features[959] = (tw_cnt >= 2.0f) ? 1.0f : 0.0f;
        out_features[960] = (move->num_placed >= 3 && num_cross >= 2) ? 1.0f : 0.0f;
        out_features[961] = (num_cross >= 1 && move->score >= 30) ? 1.0f : 0.0f;
        out_features[962] = (leave_len == 0) ? 1.0f : 0.0f;
        out_features[963] = (leave_len == 1) ? 1.0f : 0.0f;
        out_features[964] = (leave_len == 2) ? 1.0f : 0.0f;

        /* Post-move dynamic margin & win probability (974..990) */
        int post_diff = score_diff + move->score;
        int post_bag = bag_count >= move->num_placed ? (bag_count - move->num_placed) : 0;
        out_features[974] = (post_diff >= 80) ? 1.0f : 0.0f;
        out_features[975] = (post_diff >= 40 && post_diff < 80) ? 1.0f : 0.0f;
        out_features[976] = (post_diff >= 15 && post_diff < 40) ? 1.0f : 0.0f;
        out_features[977] = (post_diff >= -14 && post_diff < 15) ? 1.0f : 0.0f;
        out_features[978] = (post_diff >= -39 && post_diff < -14) ? 1.0f : 0.0f;
        out_features[979] = (post_diff >= -79 && post_diff < -39) ? 1.0f : 0.0f;
        out_features[980] = (post_diff < -79) ? 1.0f : 0.0f;
        float post_win = eval_win_probability(post_diff, post_bag);
        float pre_win = eval_win_probability(score_diff, bag_count);
        out_features[981] = post_win;
        out_features[982] = post_win - pre_win;
        out_features[983] = (score_diff < 0 && post_diff > 0) ? 1.0f : 0.0f;
        out_features[984] = (post_diff >= score_diff + 30) ? 1.0f : 0.0f;
        out_features[985] = (score_diff < -50 && move->score >= 40) ? 1.0f : 0.0f;
        out_features[986] = (score_diff > 40 && move->num_placed <= 2) ? 1.0f : 0.0f;
        out_features[987] = (post_diff > 0) ? tanhf((float)post_diff / 50.0f) : 0.0f;
        out_features[988] = (post_diff < 0) ? tanhf((float)(-post_diff) / 50.0f) : 0.0f;
        out_features[989] = (move->num_placed == 7 && post_diff >= 60) ? 1.0f : 0.0f;
        out_features[990] = (move->num_placed == 7 && fabsf((float)score_diff) <= 20.0f) ? 1.0f : 0.0f;
    }
}

void nnue_extract_features(const Board *board, const Move *move, const Rack *rack,
                          int bag_count, float *out_features) {
    nnue_extract_features_with_score(board, move, rack, bag_count, 0, out_features);
}

#if defined(__AVX2__)
static inline float hsum128(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline float hsum256(__m256 v) {
    __m128 v_hi = _mm256_extractf128_ps(v, 1);
    __m128 v_lo = _mm256_castps256_ps128(v);
    return hsum128(_mm_add_ps(v_lo, v_hi));
}
#endif

/*
 * AVX-512 / AVX2 Vectorized Raw Dot Product + Bias (No Activation).
 * 4-way unrolled reduction accumulators eliminate 4-cycle loop-carried FMA dependencies.
 */
static inline void vectorized_linear_raw(const float *weights, const float *biases,
                                         const float *in, float *out,
                                         int out_dim, int in_dim) {
#if defined(__AVX512F__)
    for (int o = 0; o < out_dim; o++) {
        __m512 v_acc0 = _mm512_setzero_ps();
        __m512 v_acc1 = _mm512_setzero_ps();
        const float *w_row = &weights[o * in_dim];
        int i = 0;
        for (; i + 31 < in_dim; i += 32) {
            v_acc0 = _mm512_fmadd_ps(_mm512_load_ps(&w_row[i + 0]),  _mm512_load_ps(&in[i + 0]),  v_acc0);
            v_acc1 = _mm512_fmadd_ps(_mm512_load_ps(&w_row[i + 16]), _mm512_load_ps(&in[i + 16]), v_acc1);
        }
        for (; i < in_dim; i += 16) {
            v_acc0 = _mm512_fmadd_ps(_mm512_load_ps(&w_row[i]), _mm512_load_ps(&in[i]), v_acc0);
        }
        out[o] = _mm512_reduce_add_ps(_mm512_add_ps(v_acc0, v_acc1)) + biases[o];
    }
#elif defined(__AVX2__)
    for (int o = 0; o < out_dim; o++) {
        __m256 v_acc0 = _mm256_setzero_ps();
        __m256 v_acc1 = _mm256_setzero_ps();
        __m256 v_acc2 = _mm256_setzero_ps();
        __m256 v_acc3 = _mm256_setzero_ps();
        const float *w_row = &weights[o * in_dim];
        int i = 0;
        for (; i + 31 < in_dim; i += 32) {
            v_acc0 = _mm256_fmadd_ps(_mm256_load_ps(&w_row[i + 0]),  _mm256_load_ps(&in[i + 0]),  v_acc0);
            v_acc1 = _mm256_fmadd_ps(_mm256_load_ps(&w_row[i + 8]),  _mm256_load_ps(&in[i + 8]),  v_acc1);
            v_acc2 = _mm256_fmadd_ps(_mm256_load_ps(&w_row[i + 16]), _mm256_load_ps(&in[i + 16]), v_acc2);
            v_acc3 = _mm256_fmadd_ps(_mm256_load_ps(&w_row[i + 24]), _mm256_load_ps(&in[i + 24]), v_acc3);
        }
        for (; i < in_dim; i += 8) {
            v_acc0 = _mm256_fmadd_ps(_mm256_load_ps(&w_row[i]), _mm256_load_ps(&in[i]), v_acc0);
        }
        __m256 sum256 = _mm256_add_ps(_mm256_add_ps(v_acc0, v_acc1), _mm256_add_ps(v_acc2, v_acc3));
        out[o] = hsum256(sum256) + biases[o];
    }
#else
    for (int o = 0; o < out_dim; o++) {
        float sum = biases[o];
        const float *w_row = &weights[o * in_dim];
        for (int i = 0; i < in_dim; i++) {
            sum += w_row[i] * in[i];
        }
        out[o] = sum;
    }
#endif
}

/*
 * Dual activation function: Path A ClippedReLU + Path B SqrClippedReLU (SCReLU)
 * Input: in[NNUE_L1_DIM] (256 floats)
 * Output: out[NNUE_L1_ACT_DIM] (512 floats)
 *   out[0..255]     = clamp(in[i], 0.0f, 1.0f)
 *   out[256..511]   = clamp(in[i], 0.0f, 1.0f)^2
 */
static inline void vectorized_dual_screlu(const float *in, float *out) {
#if defined(__AVX512F__)
    __m512 v_zero = _mm512_setzero_ps();
    __m512 v_one = _mm512_set1_ps(1.0f);
    for (int i = 0; i < NNUE_L1_DIM; i += 32) {
        __m512 x0 = _mm512_load_ps(&in[i + 0]);
        __m512 c0 = _mm512_min_ps(_mm512_max_ps(x0, v_zero), v_one);
        _mm512_store_ps(&out[i + 0], c0);
        _mm512_store_ps(&out[NNUE_L1_DIM + i + 0], _mm512_mul_ps(c0, c0));

        __m512 x1 = _mm512_load_ps(&in[i + 16]);
        __m512 c1 = _mm512_min_ps(_mm512_max_ps(x1, v_zero), v_one);
        _mm512_store_ps(&out[i + 16], c1);
        _mm512_store_ps(&out[NNUE_L1_DIM + i + 16], _mm512_mul_ps(c1, c1));
    }
#elif defined(__AVX2__)
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_one = _mm256_set1_ps(1.0f);
    for (int i = 0; i < NNUE_L1_DIM; i += 32) {
        __m256 x0 = _mm256_load_ps(&in[i + 0]);
        __m256 c0 = _mm256_min_ps(_mm256_max_ps(x0, v_zero), v_one);
        _mm256_store_ps(&out[i + 0], c0);
        _mm256_store_ps(&out[NNUE_L1_DIM + i + 0], _mm256_mul_ps(c0, c0));

        __m256 x1 = _mm256_load_ps(&in[i + 8]);
        __m256 c1 = _mm256_min_ps(_mm256_max_ps(x1, v_zero), v_one);
        _mm256_store_ps(&out[i + 8], c1);
        _mm256_store_ps(&out[NNUE_L1_DIM + i + 8], _mm256_mul_ps(c1, c1));

        __m256 x2 = _mm256_load_ps(&in[i + 16]);
        __m256 c2 = _mm256_min_ps(_mm256_max_ps(x2, v_zero), v_one);
        _mm256_store_ps(&out[i + 16], c2);
        _mm256_store_ps(&out[NNUE_L1_DIM + i + 16], _mm256_mul_ps(c2, c2));

        __m256 x3 = _mm256_load_ps(&in[i + 24]);
        __m256 c3 = _mm256_min_ps(_mm256_max_ps(x3, v_zero), v_one);
        _mm256_store_ps(&out[i + 24], c3);
        _mm256_store_ps(&out[NNUE_L1_DIM + i + 24], _mm256_mul_ps(c3, c3));
    }
#else
    for (int i = 0; i < NNUE_L1_DIM; i++) {
        float c = fminf(fmaxf(in[i], 0.0f), 1.0f);
        out[i] = c;
        out[NNUE_L1_DIM + i] = c * c;
    }
#endif
}

/*
 * AVX-512 / AVX2 Vectorized Dot Product + Bias with ClippedReLU activation.
 * 4-way unrolled reduction accumulators eliminate 4-cycle loop-carried FMA dependencies.
 */
static inline void vectorized_linear_crelu(const float *weights, const float *biases,
                                           const float *in, float *out,
                                           int out_dim, int in_dim) {
#if defined(__AVX512F__)
    for (int o = 0; o < out_dim; o++) {
        __m512 v_acc0 = _mm512_setzero_ps();
        __m512 v_acc1 = _mm512_setzero_ps();
        const float *w_row = &weights[o * in_dim];

        int i = 0;
        for (; i + 31 < in_dim; i += 32) {
            v_acc0 = _mm512_fmadd_ps(_mm512_load_ps(&w_row[i + 0]),  _mm512_load_ps(&in[i + 0]),  v_acc0);
            v_acc1 = _mm512_fmadd_ps(_mm512_load_ps(&w_row[i + 16]), _mm512_load_ps(&in[i + 16]), v_acc1);
        }
        for (; i < in_dim; i += 16) {
            v_acc0 = _mm512_fmadd_ps(_mm512_load_ps(&w_row[i]), _mm512_load_ps(&in[i]), v_acc0);
        }

        float sum = _mm512_reduce_add_ps(_mm512_add_ps(v_acc0, v_acc1)) + biases[o];
        out[o] = fminf(fmaxf(sum, 0.0f), 1.0f);
    }
#elif defined(__AVX2__)
    for (int o = 0; o < out_dim; o++) {
        __m256 v_acc0 = _mm256_setzero_ps();
        __m256 v_acc1 = _mm256_setzero_ps();
        __m256 v_acc2 = _mm256_setzero_ps();
        __m256 v_acc3 = _mm256_setzero_ps();
        const float *w_row = &weights[o * in_dim];

        int i = 0;
        for (; i + 31 < in_dim; i += 32) {
            v_acc0 = _mm256_fmadd_ps(_mm256_load_ps(&w_row[i + 0]),  _mm256_load_ps(&in[i + 0]),  v_acc0);
            v_acc1 = _mm256_fmadd_ps(_mm256_load_ps(&w_row[i + 8]),  _mm256_load_ps(&in[i + 8]),  v_acc1);
            v_acc2 = _mm256_fmadd_ps(_mm256_load_ps(&w_row[i + 16]), _mm256_load_ps(&in[i + 16]), v_acc2);
            v_acc3 = _mm256_fmadd_ps(_mm256_load_ps(&w_row[i + 24]), _mm256_load_ps(&in[i + 24]), v_acc3);
        }
        for (; i < in_dim; i += 8) {
            v_acc0 = _mm256_fmadd_ps(_mm256_load_ps(&w_row[i]), _mm256_load_ps(&in[i]), v_acc0);
        }

        __m256 sum256 = _mm256_add_ps(_mm256_add_ps(v_acc0, v_acc1), _mm256_add_ps(v_acc2, v_acc3));
        float sum = hsum256(sum256) + biases[o];
        out[o] = fminf(fmaxf(sum, 0.0f), 1.0f);
    }
#else
    for (int o = 0; o < out_dim; o++) {
        float sum = biases[o];
        const float *w_row = &weights[o * in_dim];
        for (int i = 0; i < in_dim; i++) {
            sum += w_row[i] * in[i];
        }
        out[o] = fminf(fmaxf(sum, 0.0f), 1.0f);
    }
#endif
}

float nnue_forward(const NNUEModel *model, const float *features) {
    if (!model || !model->is_loaded) return 0.0f;

    /* Layer 1: 1024 -> 256 accumulator */
    alignas(64) float l1_acc[NNUE_L1_DIM];
    vectorized_linear_raw(model->w1, model->b1, features, l1_acc, NNUE_L1_DIM, NNUE_INPUT_DIM);

    /* Dual SCReLU activation: 256 -> 512 */
    alignas(64) float l1_act[NNUE_L1_ACT_DIM];
    vectorized_dual_screlu(l1_acc, l1_act);

    /* Layer 2: 512 -> 64 */
    alignas(64) float l2_act[NNUE_L2_DIM];
    vectorized_linear_crelu(model->w2, model->b2, l1_act, l2_act, NNUE_L2_DIM, NNUE_L1_ACT_DIM);

    /* Layer 3: 64 -> 32 */
    alignas(64) float l3_act[NNUE_L3_DIM];
    vectorized_linear_crelu(model->w3, model->b3, l2_act, l3_act, NNUE_L3_DIM, NNUE_L2_DIM);

    /* Spread Head: 32 -> 1 */
#if defined(__AVX512F__)
    __m512 v_acc = _mm512_setzero_ps();
    for (int i = 0; i < NNUE_L3_DIM; i += 16) {
        __m512 v_w = _mm512_loadu_ps(&model->w_spread[i]);
        __m512 v_x = _mm512_loadu_ps(&l3_act[i]);
        v_acc = _mm512_fmadd_ps(v_w, v_x, v_acc);
    }
    float raw_equity = _mm512_reduce_add_ps(v_acc) + model->b_spread;
#elif defined(__AVX2__)
    __m256 v_acc = _mm256_setzero_ps();
    for (int i = 0; i < NNUE_L3_DIM; i += 8) {
        __m256 v_w = _mm256_loadu_ps(&model->w_spread[i]);
        __m256 v_x = _mm256_loadu_ps(&l3_act[i]);
        v_acc = _mm256_fmadd_ps(v_w, v_x, v_acc);
    }
    float raw_equity = hsum256(v_acc) + model->b_spread;
#else
    float raw_equity = model->b_spread;
    for (int i = 0; i < NNUE_L3_DIM; i++) {
        raw_equity += model->w_spread[i] * l3_act[i];
    }
#endif

    return raw_equity * 50.0f;
}

float nnue_forward_quantized(const NNUEQuantizedModel *model, const float *features) {
    if (!model || !model->is_loaded) return 0.0f;

    alignas(64) int16_t l1_acc[NNUE_L1_DIM];
    alignas(64) uint8_t l1_act[NNUE_L1_ACT_DIM];

    for (int o = 0; o < NNUE_L1_DIM; o++) {
        int32_t sum = (int32_t)model->b1[o];
        const int8_t *w_row = &model->w1[o * NNUE_INPUT_DIM];
        for (int i = 0; i < NNUE_INPUT_DIM; i++) {
            if (features[i] > 0.0f) {
                sum += (int32_t)w_row[i] * (int32_t)roundf(features[i] * 64.0f);
            }
        }
        l1_acc[o] = (int16_t)(sum / 64);
        int v = l1_acc[o] > 0 ? (l1_acc[o] > 127 ? 127 : l1_acc[o]) : 0;
        l1_act[o] = (uint8_t)v;
        l1_act[NNUE_L1_DIM + o] = (uint8_t)((v * v) / 127);
    }

    alignas(64) uint8_t l2_act[NNUE_L2_DIM];
    for (int o = 0; o < NNUE_L2_DIM; o++) {
        int32_t sum = (int32_t)model->b2[o];
        const int8_t *w_row = &model->w2[o * NNUE_L1_ACT_DIM];

#if defined(__AVX2__)
        __m256i sum_v = _mm256_setzero_si256();
        __m256i ones = _mm256_set1_epi16(1);
        for (int i = 0; i < NNUE_L1_ACT_DIM; i += 32) {
            __m256i act_v = _mm256_load_si256((const __m256i*)&l1_act[i]);
            __m256i w_v = _mm256_loadu_si256((const __m256i*)&w_row[i]);
            __m256i prod = _mm256_maddubs_epi16(act_v, w_v);
            __m256i prod32 = _mm256_madd_epi16(prod, ones);
            sum_v = _mm256_add_epi32(sum_v, prod32);
        }
        __m128i lo = _mm256_castsi256_si128(sum_v);
        __m128i hi = _mm256_extracti128_si256(sum_v, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum += _mm_cvtsi128_si32(sum128);
#else
        for (int i = 0; i < NNUE_L1_ACT_DIM; i++) {
            sum += (int32_t)w_row[i] * (int32_t)l1_act[i];
        }
#endif
        int v = sum / 128;
        l2_act[o] = (uint8_t)(v > 0 ? (v > 127 ? 127 : v) : 0);
    }

    alignas(64) uint8_t l3_act[NNUE_L3_DIM];
    for (int o = 0; o < NNUE_L3_DIM; o++) {
        int32_t sum = (int32_t)model->b3[o];
        const int8_t *w_row = &model->w3[o * NNUE_L2_DIM];
#if defined(__AVX2__)
        __m256i sum_v = _mm256_setzero_si256();
        __m256i ones = _mm256_set1_epi16(1);
        for (int i = 0; i < NNUE_L2_DIM; i += 32) {
            __m256i act_v = _mm256_load_si256((const __m256i*)&l2_act[i]);
            __m256i w_v = _mm256_loadu_si256((const __m256i*)&w_row[i]);
            __m256i prod = _mm256_maddubs_epi16(act_v, w_v);
            __m256i prod32 = _mm256_madd_epi16(prod, ones);
            sum_v = _mm256_add_epi32(sum_v, prod32);
        }
        __m128i lo = _mm256_castsi256_si128(sum_v);
        __m128i hi = _mm256_extracti128_si256(sum_v, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum += _mm_cvtsi128_si32(sum128);
#else
        for (int i = 0; i < NNUE_L2_DIM; i++) {
            sum += (int32_t)w_row[i] * (int32_t)l2_act[i];
        }
#endif
        int v = sum / 128;
        l3_act[o] = (uint8_t)(v > 0 ? (v > 127 ? 127 : v) : 0);
    }

    int32_t out_acc = (int32_t)model->b_spread;
    for (int i = 0; i < NNUE_L3_DIM; i++) {
        out_acc += (int32_t)model->w_spread[i] * (int32_t)l3_act[i];
    }

    float raw = (float)out_acc / (512.0f * 128.0f);
    return raw * 50.0f;
}

NNUEResult nnue_forward_dual(const NNUEModel *model, const float *features) {
    NNUEResult res;
    res.net_spread = 0.0f;
    res.win_probability = 0.5f;
    if (!model || !model->is_loaded) return res;

    /* Layer 1: 1024 -> 256 accumulator */
    alignas(64) float l1_acc[NNUE_L1_DIM];
    vectorized_linear_raw(model->w1, model->b1, features, l1_acc, NNUE_L1_DIM, NNUE_INPUT_DIM);

    /* Dual SCReLU activation: 256 -> 512 */
    alignas(64) float l1_act[NNUE_L1_ACT_DIM];
    vectorized_dual_screlu(l1_acc, l1_act);

    /* Layer 2: 512 -> 64 */
    alignas(64) float l2_act[NNUE_L2_DIM];
    vectorized_linear_crelu(model->w2, model->b2, l1_act, l2_act, NNUE_L2_DIM, NNUE_L1_ACT_DIM);

    /* Layer 3: 64 -> 32 */
    alignas(64) float l3_act[NNUE_L3_DIM];
    vectorized_linear_crelu(model->w3, model->b3, l2_act, l3_act, NNUE_L3_DIM, NNUE_L2_DIM);

    /* Dual Heads: 32 -> 1 */
#if defined(__AVX512F__)
    __m512 v_acc1 = _mm512_setzero_ps();
    __m512 v_acc2 = _mm512_setzero_ps();

    for (int i = 0; i < NNUE_L3_DIM; i += 16) {
        __m512 v_x = _mm512_loadu_ps(&l3_act[i]);
        __m512 v_w1 = _mm512_loadu_ps(&model->w_spread[i]);
        __m512 v_w2 = _mm512_loadu_ps(&model->w_win[i]);
        v_acc1 = _mm512_fmadd_ps(v_w1, v_x, v_acc1);
        v_acc2 = _mm512_fmadd_ps(v_w2, v_x, v_acc2);
    }

    float raw_equity = _mm512_reduce_add_ps(v_acc1) + model->b_spread;
    res.net_spread = raw_equity * 50.0f;

    float raw_win = _mm512_reduce_add_ps(v_acc2) + model->b_win;
    res.win_probability = 1.0f / (1.0f + expf(-raw_win * 3.0f));
#elif defined(__AVX2__)
    __m256 v_acc1 = _mm256_setzero_ps();
    __m256 v_acc2 = _mm256_setzero_ps();

    for (int i = 0; i < NNUE_L3_DIM; i += 8) {
        __m256 v_x = _mm256_loadu_ps(&l3_act[i]);
        __m256 v_w1 = _mm256_loadu_ps(&model->w_spread[i]);
        __m256 v_w2 = _mm256_loadu_ps(&model->w_win[i]);
        v_acc1 = _mm256_fmadd_ps(v_w1, v_x, v_acc1);
        v_acc2 = _mm256_fmadd_ps(v_w2, v_x, v_acc2);
    }

    float raw_equity = hsum256(v_acc1) + model->b_spread;
    res.net_spread = raw_equity * 50.0f;

    float raw_win = hsum256(v_acc2) + model->b_win;
    res.win_probability = 1.0f / (1.0f + expf(-raw_win * 3.0f));
#else
    float raw_equity = model->b_spread;
    float raw_win = model->b_win;
    for (int i = 0; i < NNUE_L3_DIM; i++) {
        raw_equity += model->w_spread[i] * l3_act[i];
        raw_win += model->w_win[i] * l3_act[i];
    }
    res.net_spread = raw_equity * 50.0f;
    res.win_probability = 1.0f / (1.0f + expf(-raw_win * 3.0f));
#endif

    return res;
}

float nnue_evaluate_move_with_score(const NNUEModel *model, const Board *board, const Move *move,
                                    const Rack *rack, int bag_count, int score_diff) {
    if (!model || !model->is_loaded) {
        return eval_move_static_with_score(board, move, bag_count, score_diff);
    }

    alignas(64) float features[NNUE_INPUT_DIM];
    nnue_extract_features_with_score(board, move, rack, bag_count, score_diff, features);

    float nnue_pred = nnue_forward(model, features);
    return (float)move->score + nnue_pred;
}

float nnue_evaluate_move(const NNUEModel *model, const Board *board, const Move *move,
                         const Rack *rack, int bag_count) {
    return nnue_evaluate_move_with_score(model, board, move, rack, bag_count, 0);
}

/*
 * Incremental Accumulator Architecture (Stockfish HalfKP pattern)
 */

void nnue_accumulator_build_base(const NNUEModel *model, const Board *board, const Rack *rack,
                                 int bag_count, int score_diff, NNUEAccumulator *base_acc) {
    if (!model || !base_acc) return;
    alignas(64) float base_features[NNUE_INPUT_DIM];
    nnue_extract_features_with_score(board, NULL, rack, bag_count, score_diff, base_features);

#if defined(__AVX512F__)
    for (int o = 0; o < NNUE_L1_DIM; o += 16) {
        __m512 acc = _mm512_load_ps(&model->b1[o]);
        for (int i = 0; i < NNUE_INPUT_DIM; i++) {
            float val = base_features[i];
            if (val != 0.0f) {
                __m512 w = _mm512_load_ps(&model->w1_by_feature[i][o]);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(val), w, acc);
            }
        }
        _mm512_store_ps(&base_acc->vals[o], acc);
    }
#elif defined(__AVX2__)
    for (int o = 0; o < NNUE_L1_DIM; o += 8) {
        __m256 acc = _mm256_load_ps(&model->b1[o]);
        for (int i = 0; i < NNUE_INPUT_DIM; i++) {
            float val = base_features[i];
            if (val != 0.0f) {
                __m256 w = _mm256_load_ps(&model->w1_by_feature[i][o]);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(val), w, acc);
            }
        }
        _mm256_store_ps(&base_acc->vals[o], acc);
    }
#else
    for (int o = 0; o < NNUE_L1_DIM; o++) {
        float sum = model->b1[o];
        for (int i = 0; i < NNUE_INPUT_DIM; i++) {
            if (base_features[i] != 0.0f) {
                sum += base_features[i] * model->w1_by_feature[i][o];
            }
        }
        base_acc->vals[o] = sum;
    }
#endif
}

int nnue_extract_move_sparse_features(const Board *board, const Move *move,
                                      int bag_count, int score_diff, SparseFeature *out_sparse) {
    if (!move || !out_sparse || !board) return 0;

    alignas(64) float features[NNUE_INPUT_DIM] = {0};

    int leave_counts[27] = {0};
    int leave_len = 0;
    int blanks = 0, high_placed = 0;
    int vowels = 0, consonants = 0, high_points = 0, quad_points = 0;
    int duplicate_count = 0, distinct_letters = 0;
    int total_leave_points = 0;
    float tw_cnt = 0.0f, dw_cnt = 0.0f, tl_cnt = 0.0f, dl_cnt = 0.0f;
    int blanks_placed = 0;
    float center_dist = 0.0f;
    bool perimeter_touch = false;
    bool ideal_vc = false;
    int clumsy = 0;
    int exp_empty_tw = 0, exp_empty_dw = 0, exp_empty_tl = 0, exp_empty_dl = 0;
    int exp_tw_consonants = 0, exp_tw_vowels = 0;
    int exp_dw_vowels = 0, exp_dw_consonants = 0;
    int min_r = 15, max_r = -1, min_c = 15, max_c = -1;
    int placed_on_bonus = 0;
    bool touch_r0 = false, touch_r14 = false, touch_c0 = false, touch_c14 = false;
    bool touch_r7 = false, touch_c7 = false;
    bool crosses_tw_lane = false, crosses_dw_lane = false;
    bool touches_power = false;
    bool s_placed = false;
    bool q_placed = false, z_placed = false, x_placed = false, j_placed = false;
    bool high_on_bonus = false;
    int exposed_tw_vowels = 0;
    int floating_tiles = 0;

    /* 1. Placement Geometry & Board Intersections (0..15) */
    features[0] = (float)move->score / 50.0f;
    features[1] = (float)move->num_placed / 7.0f;
    features[2] = (move->num_placed == 7) ? 1.0f : 0.0f;

    for (int p = 0; p < move->num_placed; p++) {
        int r = move->placed[p].row;
        int c = move->placed[p].col;
        MultiplierType m = board_get_multiplier(r, c);
        if (m == MULT_TW) tw_cnt += 1.0f;
        else if (m == MULT_DW) dw_cnt += 1.0f;
        else if (m == MULT_TL) tl_cnt += 1.0f;
        else if (m == MULT_DL) dl_cnt += 1.0f;
        if (m != MULT_NONE) placed_on_bonus++;

        if (r == 0) touch_r0 = true;
        if (r == 14) touch_r14 = true;
        if (c == 0) touch_c0 = true;
        if (c == 14) touch_c14 = true;
        if (r == 7) touch_r7 = true;
        if (c == 7) touch_c7 = true;
        if (r == 0 || r == 7 || r == 14) crosses_tw_lane = true;
        if (c == 0 || c == 7 || c == 14) crosses_tw_lane = true;
        if (r == c || r + c == 14) crosses_dw_lane = true;

        if (r < min_r) min_r = r;
        if (r > max_r) max_r = r;
        if (c < min_c) min_c = c;
        if (c > max_c) max_c = c;

        if (move->placed[p].is_blank) blanks_placed++;
        char let = move->placed[p].letter;
        if (let == 'S') s_placed = true;
        bool is_b = (m != MULT_NONE);
        if (let == 'Q') { q_placed = true; if (is_b) high_on_bonus = true; }
        if (let == 'Z') { z_placed = true; if (is_b) high_on_bonus = true; }
        if (let == 'X') { x_placed = true; if (is_b) high_on_bonus = true; }
        if (let == 'J') { j_placed = true; if (is_b) high_on_bonus = true; }

        if (let >= 'A' && let <= 'Z' && TILE_VALUES[let - 'A'] >= 4) high_placed++;

        float dist = (float)(abs(r - 7) + abs(c - 7));
        if (dist > center_dist) center_dist = dist;
        if (r == 0 || r == 14 || c == 0 || c == 14) perimeter_touch = true;

        bool is_v = is_vowel_char(let);
        int dr[4] = {-1, 1, 0, 0};
        int dc[4] = {0, 0, -1, 1};
        bool has_perp = false;
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d];
            int nc = c + dc[d];
            if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                char t = board->cells[nr][nc].tile;
                if (t != 0) {
                    has_perp = true;
                    if (t == 'Q' || t == 'Z' || t == 'X' || t == 'J') touches_power = true;
                } else {
                    MultiplierType nm = board_get_multiplier(nr, nc);
                    if (nm == MULT_TW) {
                        exp_empty_tw++;
                        if (is_v) exp_tw_vowels++;
                        else exp_tw_consonants++;
                    } else if (nm == MULT_DW) {
                        exp_empty_dw++;
                        if (is_v) exp_dw_vowels++;
                        else exp_dw_consonants++;
                    } else if (nm == MULT_TL) {
                        exp_empty_tl++;
                    } else if (nm == MULT_DL) {
                        exp_empty_dl++;
                    }
                    if (is_v && nm == MULT_TW) {
                        exposed_tw_vowels++;
                    }
                }
            }
        }
        if (!has_perp) floating_tiles++;
    }

    features[3] = tw_cnt;
    features[4] = dw_cnt;
    features[5] = tl_cnt;
    features[6] = dl_cnt;
    features[7] = (float)strlen(move->word) / 15.0f;
    features[8] = (move->dir == DIR_VERTICAL) ? 1.0f : 0.0f;
    features[9] = center_dist / 14.0f;
    features[10] = perimeter_touch ? 1.0f : 0.0f;
    features[11] = (float)blanks_placed;
    features[12] = (float)high_placed / 3.0f;
    features[13] = (float)move->score / (float)(move->num_placed > 0 ? move->num_placed : 1) / 20.0f;
    features[14] = (tw_cnt > 0.0f && dw_cnt > 0.0f) ? 1.0f : 0.0f;
    features[15] = (move->type == MOVE_EXCHANGE) ? 1.0f : 0.0f;

    /* 2. Leave Tile Counts (16..42) */
    leave_len = (int)strlen(move->leave);
    for (int i = 0; i < leave_len; i++) {
        char c = move->leave[i];
        if (c == '?') {
            leave_counts[26]++;
            blanks++;
        } else if (c >= 'A' && c <= 'Z') {
            int idx = c - 'A';
            if (leave_counts[idx] == 0) distinct_letters++;
            else duplicate_count++;
            leave_counts[idx]++;

            if (is_vowel_char(c)) vowels++;
            else consonants++;

            int val = TILE_VALUES[idx];
            total_leave_points += val;
            if (val >= 4) high_points++;
            if (val >= 8) quad_points++;
        }
    }
    for (int i = 0; i < 27; i++) {
        features[16 + i] = (float)leave_counts[i];
    }

    /* 3. Pairwise Leave Interactions (43..420) */
    int pair_idx = 0;
    for (int i = 0; i < 27; i++) {
        for (int j = i; j < 27; j++) {
            if (pair_idx < 378) {
                if (i == j) {
                    if (leave_counts[i] >= 2) features[43 + pair_idx] = 1.0f;
                } else {
                    if (leave_counts[i] >= 1 && leave_counts[j] >= 1) {
                        features[43 + pair_idx] = 1.0f;
                    }
                }
            }
            pair_idx++;
        }
    }

    /* 4. Tri-Tile Leave Synergies (421..484) */
    static const char *triplets[64] = {
        "ING", "ERS", "EST", "IER", "IES", "ION", "IST", "ITY", "ANT", "ENT",
        "ATE", "IVE", "OUS", "FUL", "LES", "NES", "RED", "TED", "CON", "DIS",
        "MIS", "PRE", "PRO", "SUB", "RET", "TIS", "RIN", "SIN", "TIN", "AIN",
        "EIN", "OIN", "RES", "TES", "DES", "SES", "STA", "STR", "ADE", "AGE",
        "ALE", "ANE", "ARE", "ASE", "AVE", "EAR", "EAT", "END", "ERA", "ERR",
        "ETA", "IDE", "INE", "OAR", "ONE", "ORE", "ORT", "AIR", "AIS", "AIT",
        "ELS", "ELT", "ENS", "ENT"
    };
    if (leave_len >= 3) {
        for (int t = 0; t < 64; t++) {
            int req[27] = {0};
            const char *trip = triplets[t];
            for (int k = 0; k < 3; k++) req[trip[k] - 'A']++;
            bool ok = true;
            for (int k = 0; k < 26; k++) {
                if (req[k] > leave_counts[k]) { ok = false; break; }
            }
            if (ok) features[421 + t] = 1.0f;
        }
    }

    /* 5. Macro Rack Composition (485..504) */
    features[485] = (float)vowels / 5.0f;
    features[486] = (float)consonants / 5.0f;
    features[487] = ((float)vowels - (float)consonants) / 5.0f;
    features[488] = (float)blanks;
    features[489] = (float)high_points / 3.0f;
    features[490] = (float)quad_points;
    features[491] = (float)duplicate_count / 3.0f;
    features[492] = (duplicate_count >= 2) ? 1.0f : 0.0f;
    features[493] = (float)distinct_letters / 7.0f;
    features[494] = (float)total_leave_points / 30.0f;
    float v_ratio = (vowels + consonants > 0) ? (float)vowels / (float)(vowels + consonants) : 0.5f;
    features[495] = v_ratio;
    ideal_vc = (vowels == 2 && consonants == 3) || (vowels == 3 && consonants == 2) ||
               (vowels == 2 && consonants == 2) || (vowels == 3 && consonants == 3);
    features[496] = ideal_vc ? 1.0f : 0.0f;
    features[497] = (leave_len >= 3 && vowels == 0) ? 1.0f : 0.0f;
    features[498] = (leave_len >= 3 && consonants == 0) ? 1.0f : 0.0f;
    features[499] = (leave_counts['Q' - 'A'] > 0 && leave_counts['U' - 'A'] == 0) ? 1.0f : 0.0f;
    float unplayable_penalty = (leave_counts['Q' - 'A'] * 10.0f + leave_counts['V' - 'A'] * 5.0f) / 25.0f;
    features[500] = unplayable_penalty;
    features[501] = (float)leave_counts['S' - 'A'];
    clumsy = leave_counts['V' - 'A'] + leave_counts['W' - 'A'] + leave_counts['Y' - 'A'];
    features[502] = (float)clumsy / 3.0f;
    int liquid = leave_counts['L' - 'A'] + leave_counts['R' - 'A'] + leave_counts['S' - 'A'] + leave_counts['T' - 'A'];
    features[503] = (float)liquid / 4.0f;
    features[504] = (float)leave_len / 7.0f;

    /* 6. Championship Bingo Stems (505..536) */
    static const char *stems[32] = {
        "AEINST", "AEINRT", "ADENRS", "AELRST", "AEORST", "AENRST",
        "EINST",  "EINRT",  "DENRS",  "ELRST",  "EORST",  "ENRST",
        "AEIN",   "AEIR",   "AEIS",   "AEIT",   "AELN",   "AELR",
        "AELS",   "AELT",   "AERS",   "AERT",   "EILR",   "EILS",
        "EILT",   "EINS",   "EINT",   "EIRS",   "EIRT",   "ENRS",
        "ENRT",   "ERST"
    };
    if (leave_len >= 4) {
        for (int s = 0; s < 32; s++) {
            bool match = true;
            for (const char *p = stems[s]; *p; p++) {
                if (leave_counts[*p - 'A'] < 1) { match = false; break; }
            }
            if (match) features[505 + s] = 1.0f;
        }
    }

    /* 7. Exposed TW Vowels & Floating Tiles (541, 557) */
    features[541] = (float)exposed_tw_vowels;
    features[557] = (float)floating_tiles / 7.0f;

    /* 8. Board Synergy & Extended 4-Tile Stems (701..868) */
    int total_anchors = bb_popcount(&board->anchors);
    features[701] = (leave_counts['S' - 'A'] > 0 && total_anchors > 5) ? 1.0f : 0.0f;
    features[702] = (leave_counts['D' - 'A'] > 0 && total_anchors > 5) ? 1.0f : 0.0f;
    features[703] = (leave_counts['R' - 'A'] > 0 && leave_counts['E' - 'A'] > 0) ? 1.0f : 0.0f;
    features[704] = (blanks > 0 && move->score < 35) ? -1.0f : 0.0f;
    features[705] = (leave_counts['S' - 'A'] > 0 && move->score < 25) ? -0.5f : 0.0f;
    features[706] = (high_placed > 0 && move->score >= 35) ? 1.0f : 0.0f;
    features[707] = (move->num_placed == 7 && score_diff < 0) ? 1.0f : 0.0f;
    features[708] = (leave_len == 0) ? 1.0f : 0.0f;

    static const char *stems4[64] = {
        "ACER", "ACRS", "ADEL", "ADER", "ADES", "ADIS", "AELP", "AELS", "AELT", "AEMN",
        "AENR", "AENS", "AEPS", "AERS", "AEST", "AILS", "AILV", "AINR", "AINS", "AINT",
        "AIRS", "AIRT", "AIST", "ALMN", "ALNR", "ALNS", "ALOS", "ALRT", "ALST", "ANOR",
        "ANRS", "ANST", "AORT", "ARST", "CDEN", "CINT", "DEIN", "DEIR", "DEIS", "DEIT",
        "DELN", "DELR", "DELS", "DENR", "DENS", "DEOR", "DEOS", "DERT", "DEST", "EILN",
        "EILR", "EILS", "EILT", "EIMN", "EIMR", "EIMS", "EINP", "EINR", "EINS", "EINT",
        "EIPS", "EIPR", "EIRT", "EIST"
    };
    if (leave_len >= 4) {
        for (int s = 0; s < 64; s++) {
            bool ok = true;
            for (int k = 0; k < 4; k++) {
                if (leave_counts[stems4[s][k] - 'A'] < 1) { ok = false; break; }
            }
            if (ok) features[709 + s] = 1.0f;
        }
    }

    static const char *pairs2[48] = {
        "ER", "IN", "ES", "RE", "AN", "EN", "TI", "ST", "ON", "OR",
        "AT", "TE", "AL", "AR", "IS", "IT", "ED", "NT", "RA", "TO",
        "ET", "SE", "NE", "LI", "LE", "RO", "ND", "RI", "NG", "DE",
        "ME", "SA", "EL", "TA", "SI", "TH", "LO", "DI", "CO", "MA",
        "CA", "LA", "NA", "PE", "PR", "TR", "PL", "CL"
    };
    if (leave_len >= 2) {
        for (int p = 0; p < 48; p++) {
            char c0 = pairs2[p][0];
            char c1 = pairs2[p][1];
            if (c0 == c1) {
                if (leave_counts[c0 - 'A'] >= 2) features[773 + p] = 1.0f;
            } else {
                if (leave_counts[c0 - 'A'] >= 1 && leave_counts[c1 - 'A'] >= 1) features[773 + p] = 1.0f;
            }
        }
    }

    for (int c = 0; c < 26; c++) {
        if (leave_counts[c] >= 2) {
            features[821 + c] = (float)(leave_counts[c] - 1);
        }
    }
    bool trip_dup = false;
    for (int c = 0; c < 26; c++) {
        if (leave_counts[c] >= 3) { trip_dup = true; break; }
    }
    features[847] = trip_dup ? 1.0f : 0.0f;
    features[848] = (vowels == 0) ? 1.0f : 0.0f;
    features[849] = (vowels == 1) ? 1.0f : 0.0f;
    features[850] = (vowels == 2) ? 1.0f : 0.0f;
    features[851] = (vowels >= 3) ? 1.0f : 0.0f;
    features[852] = (consonants == 0) ? 1.0f : 0.0f;
    features[853] = (consonants == 1) ? 1.0f : 0.0f;
    features[854] = (consonants == 2) ? 1.0f : 0.0f;
    features[855] = (consonants >= 3) ? 1.0f : 0.0f;
    features[856] = (blanks > 0 && leave_counts['S' - 'A'] > 0) ? 1.0f : 0.0f;
    features[857] = (leave_counts['S' - 'A'] >= 2) ? 1.0f : 0.0f;
    features[858] = (blanks >= 2) ? 1.0f : 0.0f;
    features[859] = (leave_counts['Q' - 'A'] > 0) ? 1.0f : 0.0f;
    features[860] = (leave_counts['Z' - 'A'] > 0) ? 1.0f : 0.0f;
    features[861] = (leave_counts['J' - 'A'] > 0) ? 1.0f : 0.0f;
    features[862] = (leave_counts['X' - 'A'] > 0) ? 1.0f : 0.0f;
    features[863] = (leave_counts['K' - 'A'] > 0) ? 1.0f : 0.0f;
    features[864] = (leave_counts['V' - 'A'] > 0) ? 1.0f : 0.0f;
    features[865] = (leave_counts['W' - 'A'] > 0) ? 1.0f : 0.0f;
    features[866] = (leave_counts['Y' - 'A'] > 0) ? 1.0f : 0.0f;
    features[867] = (leave_counts['B' - 'A'] > 0) ? 1.0f : 0.0f;
    features[868] = (leave_counts['P' - 'A'] > 0) ? 1.0f : 0.0f;

    /* 9. Power Tiles, Lanes & Tactical Geometry (879..929) */
    features[879] = touch_r0 ? 1.0f : 0.0f;
    features[880] = touch_r14 ? 1.0f : 0.0f;
    features[881] = touch_c0 ? 1.0f : 0.0f;
    features[882] = touch_c14 ? 1.0f : 0.0f;
    features[883] = touch_r7 ? 1.0f : 0.0f;
    features[884] = touch_c7 ? 1.0f : 0.0f;
    features[885] = crosses_tw_lane ? 1.0f : 0.0f;
    features[886] = crosses_dw_lane ? 1.0f : 0.0f;
    features[887] = (float)exp_empty_tw / 4.0f;
    features[888] = (float)exp_empty_dw / 8.0f;
    features[889] = (float)exp_empty_tl / 6.0f;
    features[890] = (float)exp_empty_dl / 12.0f;
    features[891] = (float)exp_tw_consonants / 2.0f;
    features[892] = (float)exp_tw_vowels / 2.0f;
    features[893] = (float)exp_dw_vowels / 4.0f;
    features[894] = (float)exp_dw_consonants / 4.0f;
    features[895] = (min_r >= 0 && min_r <= 14) ? (float)min_r / 14.0f : 0.5f;
    features[896] = (min_c >= 0 && min_c <= 14) ? (float)min_c / 14.0f : 0.5f;
    int span = (max_r >= min_r && max_c >= min_c) ? (max_r - min_r + max_c - min_c + 1) : 0;
    features[897] = (float)span / 15.0f;
    features[898] = (move->num_placed == 1) ? 1.0f : 0.0f;
    features[899] = (move->col > 0 && board->cells[move->row][move->col - 1].tile != 0) ? 1.0f : 0.0f;
    features[900] = (move->dir == DIR_HORIZONTAL && move->col + (int)strlen(move->word) < BOARD_SIZE &&
                     board->cells[move->row][move->col + (int)strlen(move->word)].tile != 0) ? 1.0f : 0.0f;
    int num_cross = (move->num_placed > 1) ? (move->num_placed - 1) : 0;
    features[901] = (num_cross >= 2) ? 1.0f : 0.0f;
    features[902] = (float)num_cross / 5.0f;
    features[903] = (float)move->score / 50.0f;
    features[904] = (float)move->score / (float)(move->num_placed > 0 ? move->num_placed : 1) / 10.0f;
    features[905] = (move->score > 0) ? (float)placed_on_bonus / (float)(move->num_placed > 0 ? move->num_placed : 1) : 0.0f;
    features[906] = (placed_on_bonus >= 2) ? 1.0f : 0.0f;
    features[907] = (float)move->num_placed / 7.0f;
    features[908] = (float)exp_empty_tw;
    features[909] = (float)(exp_empty_tw - (tw_cnt > 0.0f ? 1 : 0));
    features[910] = (touch_r0 || touch_r14 || touch_c0 || touch_c14) ? 1.0f : 0.0f;
    features[911] = (exp_empty_tw + exp_empty_dw >= 2) ? 1.0f : 0.0f;
    features[912] = (exp_empty_tw + exp_empty_dw == 0) ? 1.0f : 0.0f;
    features[913] = (crosses_tw_lane && move->num_placed >= 4) ? 1.0f : 0.0f;
    features[914] = (center_dist <= 3.0f) ? 1.0f : 0.0f;
    features[915] = (center_dist >= 8.0f) ? 1.0f : 0.0f;
    features[916] = touches_power ? 1.0f : 0.0f;
    features[917] = (blanks_placed > 0 && move->num_placed < 7) ? 1.0f : 0.0f;
    features[918] = (blanks_placed > 0 && move->num_placed == 7) ? 1.0f : 0.0f;
    features[919] = (blanks > 0) ? 1.0f : 0.0f;
    features[920] = (s_placed && move->num_placed < 7) ? 1.0f : 0.0f;
    features[921] = (s_placed && move->num_placed == 7) ? 1.0f : 0.0f;
    features[922] = (leave_counts['S' - 'A'] > 0) ? 1.0f : 0.0f;
    features[923] = (q_placed && high_on_bonus) ? 1.0f : 0.0f;
    features[924] = (z_placed && high_on_bonus) ? 1.0f : 0.0f;
    features[925] = (x_placed && high_on_bonus) ? 1.0f : 0.0f;
    features[926] = (j_placed && high_on_bonus) ? 1.0f : 0.0f;
    features[927] = (q_placed && move->score < 20) ? 1.0f : 0.0f;
    features[928] = (high_placed > 0 && move->score >= 35) ? 1.0f : 0.0f;
    features[929] = (clumsy > 0 && (leave_counts['Q'-'A'] > 0 || leave_counts['Z'-'A'] > 0)) ? 1.0f : 0.0f;

    /* 10. Tactical Synergies (951..964) */
    bool perfect_leave = (strcmp(move->leave, "ER") == 0 || strcmp(move->leave, "IN") == 0 ||
                          strcmp(move->leave, "ST") == 0 || strcmp(move->leave, "RE") == 0 ||
                          strcmp(move->leave, "ES") == 0);
    features[951] = perfect_leave ? 1.0f : 0.0f;
    features[952] = (move->score >= 40 && move->num_placed <= 4) ? 1.0f : 0.0f;
    features[953] = (move->num_placed > 0 && (float)move->score / (float)move->num_placed >= 8.0f) ? 1.0f : 0.0f;
    features[954] = (move->num_placed == 1 && (q_placed || z_placed || x_placed || j_placed)) ? 1.0f : 0.0f;
    features[955] = (move->type == MOVE_EXCHANGE && move->num_placed >= 5) ? 1.0f : 0.0f;
    int placed_vowels = 0, placed_consonants = 0;
    for (int p = 0; p < move->num_placed; p++) {
        if (is_vowel_char(move->placed[p].letter)) placed_vowels++;
        else placed_consonants++;
    }
    features[956] = (placed_vowels >= 3 && ideal_vc) ? 1.0f : 0.0f;
    features[957] = (placed_consonants >= 4 && ideal_vc) ? 1.0f : 0.0f;
    features[958] = (move->num_placed == 1 && move->score >= 20) ? 1.0f : 0.0f;
    features[959] = (tw_cnt >= 2.0f) ? 1.0f : 0.0f;
    features[960] = (move->num_placed >= 3 && num_cross >= 2) ? 1.0f : 0.0f;
    features[961] = (num_cross >= 1 && move->score >= 30) ? 1.0f : 0.0f;
    features[962] = (leave_len == 0) ? 1.0f : 0.0f;
    features[963] = (leave_len == 1) ? 1.0f : 0.0f;
    features[964] = (leave_len == 2) ? 1.0f : 0.0f;

    /* 11. Post-Move Dynamic Margin & Win Probability (974..990) */
    int post_diff = score_diff + move->score;
    int post_bag = bag_count >= move->num_placed ? (bag_count - move->num_placed) : 0;
    features[974] = (post_diff >= 80) ? 1.0f : 0.0f;
    features[975] = (post_diff >= 40 && post_diff < 80) ? 1.0f : 0.0f;
    features[976] = (post_diff >= 15 && post_diff < 40) ? 1.0f : 0.0f;
    features[977] = (post_diff >= -14 && post_diff < 15) ? 1.0f : 0.0f;
    features[978] = (post_diff >= -39 && post_diff < -14) ? 1.0f : 0.0f;
    features[979] = (post_diff >= -79 && post_diff < -39) ? 1.0f : 0.0f;
    features[980] = (post_diff < -79) ? 1.0f : 0.0f;
    float post_win = eval_win_probability(post_diff, post_bag);
    float pre_win = eval_win_probability(score_diff, bag_count);
    features[981] = post_win;
    features[982] = post_win - pre_win;
    features[983] = (score_diff < 0 && post_diff > 0) ? 1.0f : 0.0f;
    features[984] = (post_diff >= score_diff + 30) ? 1.0f : 0.0f;
    features[985] = (score_diff < -50 && move->score >= 40) ? 1.0f : 0.0f;
    features[986] = (score_diff > 40 && move->num_placed <= 2) ? 1.0f : 0.0f;
    features[987] = (post_diff > 0) ? tanhf((float)post_diff / 50.0f) : 0.0f;
    features[988] = (post_diff < 0) ? tanhf((float)(-post_diff) / 50.0f) : 0.0f;
    features[989] = (move->num_placed == 7 && post_diff >= 60) ? 1.0f : 0.0f;
    features[990] = (move->num_placed == 7 && fabsf((float)score_diff) <= 20.0f) ? 1.0f : 0.0f;

    /* Collect non-zero sparse delta features */
    int count = 0;
    for (int i = 0; i <= 536; i++) {
        if (features[i] != 0.0f) {
            out_sparse[count].index = (uint16_t)i;
            out_sparse[count].value = features[i];
            count++;
        }
    }
    if (features[541] != 0.0f) {
        out_sparse[count].index = 541;
        out_sparse[count].value = features[541];
        count++;
    }
    if (features[557] != 0.0f) {
        out_sparse[count].index = 557;
        out_sparse[count].value = features[557];
        count++;
    }
    for (int i = 701; i <= 868; i++) {
        if (features[i] != 0.0f) {
            out_sparse[count].index = (uint16_t)i;
            out_sparse[count].value = features[i];
            count++;
        }
    }
    for (int i = 879; i <= 929; i++) {
        if (features[i] != 0.0f) {
            out_sparse[count].index = (uint16_t)i;
            out_sparse[count].value = features[i];
            count++;
        }
    }
    for (int i = 951; i <= 964; i++) {
        if (features[i] != 0.0f) {
            out_sparse[count].index = (uint16_t)i;
            out_sparse[count].value = features[i];
            count++;
        }
    }
    for (int i = 974; i <= 990; i++) {
        if (features[i] != 0.0f) {
            out_sparse[count].index = (uint16_t)i;
            out_sparse[count].value = features[i];
            count++;
        }
    }
    return count;
}

void nnue_accumulator_apply_sparse(const NNUEModel *model, const NNUEAccumulator *base,
                                   const SparseFeature *features, int num_features,
                                   NNUEAccumulator *out_acc) {
#if defined(__AVX512F__)
    for (int o = 0; o < NNUE_L1_DIM; o += 64) {
        __m512 a0 = _mm512_load_ps(&base->vals[o + 0]);
        __m512 a1 = _mm512_load_ps(&base->vals[o + 16]);
        __m512 a2 = _mm512_load_ps(&base->vals[o + 32]);
        __m512 a3 = _mm512_load_ps(&base->vals[o + 48]);
        for (int f = 0; f < num_features; f++) {
            uint16_t idx = features[f].index;
            __m512 val = _mm512_set1_ps(features[f].value);
            a0 = _mm512_fmadd_ps(val, _mm512_load_ps(&model->w1_by_feature[idx][o + 0]), a0);
            a1 = _mm512_fmadd_ps(val, _mm512_load_ps(&model->w1_by_feature[idx][o + 16]), a1);
            a2 = _mm512_fmadd_ps(val, _mm512_load_ps(&model->w1_by_feature[idx][o + 32]), a2);
            a3 = _mm512_fmadd_ps(val, _mm512_load_ps(&model->w1_by_feature[idx][o + 48]), a3);
        }
        _mm512_store_ps(&out_acc->vals[o + 0], a0);
        _mm512_store_ps(&out_acc->vals[o + 16], a1);
        _mm512_store_ps(&out_acc->vals[o + 32], a2);
        _mm512_store_ps(&out_acc->vals[o + 48], a3);
    }
#elif defined(__AVX2__)
    for (int o = 0; o < NNUE_L1_DIM; o += 32) {
        __m256 a0 = _mm256_load_ps(&base->vals[o + 0]);
        __m256 a1 = _mm256_load_ps(&base->vals[o + 8]);
        __m256 a2 = _mm256_load_ps(&base->vals[o + 16]);
        __m256 a3 = _mm256_load_ps(&base->vals[o + 24]);
        for (int f = 0; f < num_features; f++) {
            uint16_t idx = features[f].index;
            __m256 val = _mm256_set1_ps(features[f].value);
            const float *w = &model->w1_by_feature[idx][o];
            a0 = _mm256_fmadd_ps(val, _mm256_load_ps(&w[0]),  a0);
            a1 = _mm256_fmadd_ps(val, _mm256_load_ps(&w[8]),  a1);
            a2 = _mm256_fmadd_ps(val, _mm256_load_ps(&w[16]), a2);
            a3 = _mm256_fmadd_ps(val, _mm256_load_ps(&w[24]), a3);
        }
        _mm256_store_ps(&out_acc->vals[o + 0],  a0);
        _mm256_store_ps(&out_acc->vals[o + 8],  a1);
        _mm256_store_ps(&out_acc->vals[o + 16], a2);
        _mm256_store_ps(&out_acc->vals[o + 24], a3);
    }
#else
    memcpy(out_acc->vals, base->vals, sizeof(base->vals));
    for (int f = 0; f < num_features; f++) {
        uint16_t idx = features[f].index;
        float val = features[f].value;
        const float *w = model->w1_by_feature[idx];
        for (int o = 0; o < NNUE_L1_DIM; o++) {
            out_acc->vals[o] += val * w[o];
        }
    }
#endif
}

float nnue_forward_accumulator(const NNUEModel *model, const NNUEAccumulator *acc) {
    if (!model || !model->is_loaded) return 0.0f;

    /* Dual SCReLU activation directly on accumulated pre-activation sum */
    alignas(64) float l1_act[NNUE_L1_ACT_DIM];
    vectorized_dual_screlu(acc->vals, l1_act);

    /* Layer 2: 512 -> 64 */
    alignas(64) float l2_act[NNUE_L2_DIM];
    vectorized_linear_crelu(model->w2, model->b2, l1_act, l2_act, NNUE_L2_DIM, NNUE_L1_ACT_DIM);

    /* Layer 3: 64 -> 32 */
    alignas(64) float l3_act[NNUE_L3_DIM];
    vectorized_linear_crelu(model->w3, model->b3, l2_act, l3_act, NNUE_L3_DIM, NNUE_L2_DIM);

    /* Spread Head: 32 -> 1 */
#if defined(__AVX512F__)
    __m512 v_acc = _mm512_setzero_ps();
    for (int i = 0; i < NNUE_L3_DIM; i += 16) {
        __m512 v_w = _mm512_load_ps(&model->w_spread[i]);
        __m512 v_x = _mm512_load_ps(&l3_act[i]);
        v_acc = _mm512_fmadd_ps(v_w, v_x, v_acc);
    }
    float raw_equity = _mm512_reduce_add_ps(v_acc) + model->b_spread;
#elif defined(__AVX2__)
    __m256 v_acc = _mm256_setzero_ps();
    for (int i = 0; i < NNUE_L3_DIM; i += 8) {
        __m256 v_w = _mm256_load_ps(&model->w_spread[i]);
        __m256 v_x = _mm256_load_ps(&l3_act[i]);
        v_acc = _mm256_fmadd_ps(v_w, v_x, v_acc);
    }
    float raw_equity = hsum256(v_acc) + model->b_spread;
#else
    float raw_equity = model->b_spread;
    for (int i = 0; i < NNUE_L3_DIM; i++) {
        raw_equity += model->w_spread[i] * l3_act[i];
    }
#endif

    return raw_equity * 50.0f;
}

NNUEResult nnue_forward_accumulator_dual(const NNUEModel *model, const NNUEAccumulator *acc) {
    NNUEResult res;
    res.net_spread = 0.0f;
    res.win_probability = 0.5f;
    if (!model || !model->is_loaded) return res;

    /* Dual SCReLU activation directly on accumulated pre-activation sum */
    alignas(64) float l1_act[NNUE_L1_ACT_DIM];
    vectorized_dual_screlu(acc->vals, l1_act);

    /* Layer 2: 512 -> 64 */
    alignas(64) float l2_act[NNUE_L2_DIM];
    vectorized_linear_crelu(model->w2, model->b2, l1_act, l2_act, NNUE_L2_DIM, NNUE_L1_ACT_DIM);

    /* Layer 3: 64 -> 32 */
    alignas(64) float l3_act[NNUE_L3_DIM];
    vectorized_linear_crelu(model->w3, model->b3, l2_act, l3_act, NNUE_L3_DIM, NNUE_L2_DIM);

    /* Dual Heads: 32 -> 1 */
#if defined(__AVX512F__)
    __m512 v_acc1 = _mm512_setzero_ps();
    __m512 v_acc2 = _mm512_setzero_ps();
    for (int i = 0; i < NNUE_L3_DIM; i += 16) {
        __m512 v_x = _mm512_load_ps(&l3_act[i]);
        __m512 v_w1 = _mm512_load_ps(&model->w_spread[i]);
        __m512 v_w2 = _mm512_load_ps(&model->w_win[i]);
        v_acc1 = _mm512_fmadd_ps(v_w1, v_x, v_acc1);
        v_acc2 = _mm512_fmadd_ps(v_w2, v_x, v_acc2);
    }
    float raw_equity = _mm512_reduce_add_ps(v_acc1) + model->b_spread;
    res.net_spread = raw_equity * 50.0f;
    float raw_win = _mm512_reduce_add_ps(v_acc2) + model->b_win;
    res.win_probability = 1.0f / (1.0f + expf(-raw_win * 3.0f));
#elif defined(__AVX2__)
    __m256 v_acc1 = _mm256_setzero_ps();
    __m256 v_acc2 = _mm256_setzero_ps();
    for (int i = 0; i < NNUE_L3_DIM; i += 8) {
        __m256 v_x = _mm256_load_ps(&l3_act[i]);
        __m256 v_w1 = _mm256_load_ps(&model->w_spread[i]);
        __m256 v_w2 = _mm256_load_ps(&model->w_win[i]);
        v_acc1 = _mm256_fmadd_ps(v_w1, v_x, v_acc1);
        v_acc2 = _mm256_fmadd_ps(v_w2, v_x, v_acc2);
    }
    float raw_equity = hsum256(v_acc1) + model->b_spread;
    res.net_spread = raw_equity * 50.0f;
    float raw_win = hsum256(v_acc2) + model->b_win;
    res.win_probability = 1.0f / (1.0f + expf(-raw_win * 3.0f));
#else
    float raw_equity = model->b_spread;
    float raw_win = model->b_win;
    for (int i = 0; i < NNUE_L3_DIM; i++) {
        raw_equity += model->w_spread[i] * l3_act[i];
        raw_win += model->w_win[i] * l3_act[i];
    }
    res.net_spread = raw_equity * 50.0f;
    res.win_probability = 1.0f / (1.0f + expf(-raw_win * 3.0f));
#endif

    return res;
}

float nnue_forward_move_incremental(const NNUEModel *model, const NNUEAccumulator *base_acc,
                                    const Board *board, const Move *move,
                                    int bag_count, int score_diff) {
    if (!model || !model->is_loaded) return 0.0f;
    SparseFeature sparse[256];
    int n = nnue_extract_move_sparse_features(board, move, bag_count, score_diff, sparse);
    NNUEAccumulator move_acc;
    nnue_accumulator_apply_sparse(model, base_acc, sparse, n, &move_acc);
    return nnue_forward_accumulator(model, &move_acc);
}

NNUEResult nnue_forward_move_incremental_dual(const NNUEModel *model, const NNUEAccumulator *base_acc,
                                              const Board *board, const Move *move,
                                              int bag_count, int score_diff) {
    NNUEResult res;
    res.net_spread = 0.0f;
    res.win_probability = 0.5f;
    if (!model || !model->is_loaded) return res;
    SparseFeature sparse[256];
    int n = nnue_extract_move_sparse_features(board, move, bag_count, score_diff, sparse);
    NNUEAccumulator move_acc;
    nnue_accumulator_apply_sparse(model, base_acc, sparse, n, &move_acc);
    return nnue_forward_accumulator_dual(model, &move_acc);
}
