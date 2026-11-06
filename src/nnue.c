#include "nnue.h"
#include "eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdalign.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

NNUEModel g_nnue = {0};

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

    float limit2 = sqrtf(6.0f / (float)(NNUE_L1_DIM + NNUE_L2_DIM));
    for (int i = 0; i < NNUE_L2_DIM * NNUE_L1_DIM; i++) {
        model->w2[i] = rand_uniform(limit2);
    }
    for (int i = 0; i < NNUE_L2_DIM; i++) {
        model->b2[i] = 0.05f;
    }

    float limit3 = sqrtf(6.0f / (float)(NNUE_L2_DIM + 1));
    for (int i = 0; i < NNUE_L2_DIM; i++) {
        model->w3[i] = rand_uniform(limit3);
        model->w_win[i] = rand_uniform(limit3);
    }
    model->b3 = 0.0f;
    model->b_win = 0.0f;
    model->is_loaded = true;
}

bool nnue_save(const NNUEModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (!f) return false;

    uint32_t header[5] = {
        NNUE_MAGIC,
        NNUE_VERSION,
        NNUE_INPUT_DIM,
        NNUE_L1_DIM,
        NNUE_L2_DIM
    };

    if (fwrite(header, sizeof(uint32_t), 5, f) != 5) {
        fclose(f);
        return false;
    }

    fwrite(model->w1, sizeof(float), NNUE_L1_DIM * NNUE_INPUT_DIM, f);
    fwrite(model->b1, sizeof(float), NNUE_L1_DIM, f);
    fwrite(model->w2, sizeof(float), NNUE_L2_DIM * NNUE_L1_DIM, f);
    fwrite(model->b2, sizeof(float), NNUE_L2_DIM, f);
    fwrite(model->w3, sizeof(float), NNUE_L2_DIM, f);
    fwrite(&model->b3, sizeof(float), 1, f);
    fwrite(model->w_win, sizeof(float), NNUE_L2_DIM, f);
    fwrite(&model->b_win, sizeof(float), 1, f);

    fclose(f);
    return true;
}

bool nnue_load(NNUEModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return false;

    uint32_t header[5];
    if (fread(header, sizeof(uint32_t), 5, f) != 5) {
        fclose(f);
        return false;
    }

    if (header[0] != NNUE_MAGIC || header[1] != NNUE_VERSION ||
        header[2] != NNUE_INPUT_DIM || header[3] != NNUE_L1_DIM || header[4] != NNUE_L2_DIM) {
        fclose(f);
        return false;
    }

    size_t r1 = fread(model->w1, sizeof(float), NNUE_L1_DIM * NNUE_INPUT_DIM, f);
    size_t r2 = fread(model->b1, sizeof(float), NNUE_L1_DIM, f);
    size_t r3 = fread(model->w2, sizeof(float), NNUE_L2_DIM * NNUE_L1_DIM, f);
    size_t r4 = fread(model->b2, sizeof(float), NNUE_L2_DIM, f);
    size_t r5 = fread(model->w3, sizeof(float), NNUE_L2_DIM, f);
    size_t r6 = fread(&model->b3, sizeof(float), 1, f);

    size_t r7 = fread(model->w_win, sizeof(float), NNUE_L2_DIM, f);
    size_t r8 = fread(&model->b_win, sizeof(float), 1, f);

    fclose(f);

    if (r1 != NNUE_L1_DIM * NNUE_INPUT_DIM || r2 != NNUE_L1_DIM ||
        r3 != NNUE_L2_DIM * NNUE_L1_DIM || r4 != NNUE_L2_DIM ||
        r5 != NNUE_L2_DIM || r6 != 1) {
        return false;
    }

    /* Fallback if file doesn't have dual head */
    if (r7 != NNUE_L2_DIM || r8 != 1) {
        for (int i = 0; i < NNUE_L2_DIM; i++) {
            model->w_win[i] = model->w3[i] * 1.5f;
        }
        model->b_win = model->b3;
    }

    model->is_loaded = true;
    return true;
}

static inline bool is_vowel_char(char c) {
    c = (char)toupper((unsigned char)c);
    return (c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U');
}

void nnue_extract_features(const Board *board, const Move *move, const Rack *rack,
                          int bag_count, float *out_features) {
    (void)rack;
    memset(out_features, 0, NNUE_INPUT_DIM * sizeof(float));

    if (!move) return;

    /* 1. Move Placement Features (0..6) */
    out_features[0] = (float)move->score / 50.0f;
    out_features[1] = (float)move->num_placed / 7.0f;
    out_features[2] = (move->num_placed == 7) ? 1.0f : 0.0f; /* Bingo */

    float tw_cnt = 0.0f, dw_cnt = 0.0f, tl_cnt = 0.0f, dl_cnt = 0.0f;
    for (int p = 0; p < move->num_placed; p++) {
        MultiplierType m = board_get_multiplier(move->placed[p].row, move->placed[p].col);
        if (m == MULT_TW) tw_cnt += 1.0f;
        else if (m == MULT_DW) dw_cnt += 1.0f;
        else if (m == MULT_TL) tl_cnt += 1.0f;
        else if (m == MULT_DL) dl_cnt += 1.0f;
    }
    out_features[3] = tw_cnt;
    out_features[4] = dw_cnt;
    out_features[5] = tl_cnt;
    out_features[6] = dl_cnt;

    /* 2. Leave Tile Counts (7..33 for 27 tiles: 0..25=A..Z, 26=?) */
    int leave_counts[27] = {0};
    int leave_len = (int)strlen(move->leave);
    int vowels = 0, consonants = 0, blanks = 0, high_points = 0;

    for (int i = 0; i < leave_len; i++) {
        char c = move->leave[i];
        if (c == '?') {
            leave_counts[26]++;
            blanks++;
        } else if (c >= 'A' && c <= 'Z') {
            int idx = c - 'A';
            leave_counts[idx]++;
            if (is_vowel_char(c)) vowels++;
            else consonants++;

            if (TILE_VALUES[idx] >= 4) high_points++;
        }
    }

    for (int i = 0; i < 27; i++) {
        out_features[7 + i] = (float)leave_counts[i];
    }

    /* 3. Pairwise Leave Interactions (34..411: 378 pairs) */
    int pair_idx = 0;
    for (int i = 0; i < 27; i++) {
        for (int j = i; j < 27; j++) {
            if (pair_idx < 378) {
                if (i == j) {
                    if (leave_counts[i] >= 2) out_features[34 + pair_idx] = 1.0f;
                } else {
                    if (leave_counts[i] >= 1 && leave_counts[j] >= 1) {
                        out_features[34 + pair_idx] = 1.0f;
                    }
                }
            }
            pair_idx++;
        }
    }

    /* 4. Macro Rack Composition (412..416) */
    out_features[412] = (float)vowels / 5.0f;
    out_features[413] = (float)consonants / 5.0f;
    out_features[414] = ((float)vowels - (float)consonants) / 5.0f;
    out_features[415] = (float)blanks;
    out_features[416] = (float)high_points / 3.0f;

    /* 5. Championship Bingo Stem Indicators (417..432) */
    static const char *stems[16] = {
        "AEINST", "AEINRT", "ADENRS", "AELRST",
        "AEORST", "AENRST", "EINST",  "EINRT",
        "DENRS",  "ELRST",  "EORST",  "ENRST",
        "AEIN",   "AEIR",   "AEIS",   "AEIT"
    };

    char sorted_leave[8];
    strncpy(sorted_leave, move->leave, 7);
    sorted_leave[leave_len < 7 ? leave_len : 7] = '\0';
    for (int i = 0; i < leave_len - 1; i++) {
        for (int j = i + 1; j < leave_len; j++) {
            if (sorted_leave[i] > sorted_leave[j]) {
                char tmp = sorted_leave[i];
                sorted_leave[i] = sorted_leave[j];
                sorted_leave[j] = tmp;
            }
        }
    }

    for (int s = 0; s < 16; s++) {
        if (strstr(sorted_leave, stems[s]) != NULL) {
            out_features[417 + s] = 1.0f;
        }
    }

    /* 6. Board Exposure & Hazard Features (433..436) */
    int open_tw = 0;
    int open_dw = 0;
    int exposed_tw_vowels = 0;
    int total_occupied = 0;

    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (board->cells[r][c].tile != 0) {
                total_occupied++;
            } else {
                MultiplierType m = board_get_multiplier(r, c);
                if (m == MULT_TW && board_is_anchor(board, r, c)) open_tw++;
                if (m == MULT_DW && board_is_anchor(board, r, c)) open_dw++;
            }
        }
    }

    for (int p = 0; p < move->num_placed; p++) {
        int r = move->placed[p].row;
        int c = move->placed[p].col;
        if (is_vowel_char(move->placed[p].letter)) {
            int dr[4] = {-1, 1, 0, 0};
            int dc[4] = {0, 0, -1, 1};
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d];
                int nc = c + dc[d];
                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    if (board->cells[nr][nc].tile == 0 && board_get_multiplier(nr, nc) == MULT_TW) {
                        exposed_tw_vowels++;
                    }
                }
            }
        }
    }

    out_features[433] = (float)open_tw / 8.0f;
    out_features[434] = (float)open_dw / 12.0f;
    out_features[435] = (float)exposed_tw_vowels;
    out_features[436] = (float)total_occupied / 100.0f;

    /* 7. Game Phase Buckets (437..444) */
    PhaseBucket phase = eval_classify_phase(bag_count);
    int p_idx = (int)phase;
    if (p_idx >= 0 && p_idx < 8) {
        out_features[437 + p_idx] = 1.0f;
    }

    /* 8. Bag Exhaustion Context (445..447) */
    out_features[445] = (float)bag_count / 100.0f;
    out_features[446] = (bag_count == 0) ? 1.0f : 0.0f;
    out_features[447] = (bag_count <= 14) ? 1.0f : 0.0f;
}

/*
 * AVX2 Vectorized Dot Product + Bias with ClippedReLU activation (with scalar fallback for portability).
 */
static inline void avx2_linear_crelu(const float *weights, const float *biases,
                                     const float *in, float *out,
                                     int out_dim, int in_dim) {
#if defined(__AVX2__)
    for (int o = 0; o < out_dim; o++) {
        __m256 v_acc = _mm256_setzero_ps();
        const float *w_row = &weights[o * in_dim];

        for (int i = 0; i < in_dim; i += 8) {
            __m256 v_w = _mm256_loadu_ps(&w_row[i]);
            __m256 v_x = _mm256_loadu_ps(&in[i]);
            v_acc = _mm256_fmadd_ps(v_w, v_x, v_acc);
        }

        /* Horizontal sum across 8 lanes */
        __m128 v_hi = _mm256_extractf128_ps(v_acc, 1);
        __m128 v_lo = _mm256_castps256_ps128(v_acc);
        __m128 v_sum128 = _mm_add_ps(v_lo, v_hi);
        v_sum128 = _mm_hadd_ps(v_sum128, v_sum128);
        v_sum128 = _mm_hadd_ps(v_sum128, v_sum128);

        float sum = _mm_cvtss_f32(v_sum128) + biases[o];

        /* ClippedReLU: clamp(sum, 0.0f, 1.0f) */
        sum = fminf(fmaxf(sum, 0.0f), 1.0f);
        out[o] = sum;
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

    /* Layer 1: 448 -> 128 */
    alignas(32) float l1_out[NNUE_L1_DIM];
    avx2_linear_crelu(model->w1, model->b1, features, l1_out, NNUE_L1_DIM, NNUE_INPUT_DIM);

    /* Layer 2: 128 -> 32 */
    alignas(32) float l2_out[NNUE_L2_DIM];
    avx2_linear_crelu(model->w2, model->b2, l1_out, l2_out, NNUE_L2_DIM, NNUE_L1_DIM);

    /* Layer 3: 32 -> 1 */
#if defined(__AVX2__)
    __m256 v_acc = _mm256_setzero_ps();
    for (int i = 0; i < NNUE_L2_DIM; i += 8) {
        __m256 v_w = _mm256_loadu_ps(&model->w3[i]);
        __m256 v_x = _mm256_loadu_ps(&l2_out[i]);
        v_acc = _mm256_fmadd_ps(v_w, v_x, v_acc);
    }

    __m128 v_hi = _mm256_extractf128_ps(v_acc, 1);
    __m128 v_lo = _mm256_castps256_ps128(v_acc);
    __m128 v_sum128 = _mm_add_ps(v_lo, v_hi);
    v_sum128 = _mm_hadd_ps(v_sum128, v_sum128);
    v_sum128 = _mm_hadd_ps(v_sum128, v_sum128);

    float raw_equity = _mm_cvtss_f32(v_sum128) + model->b3;
#else
    float raw_equity = model->b3;
    for (int i = 0; i < NNUE_L2_DIM; i++) {
        raw_equity += model->w3[i] * l2_out[i];
    }
#endif

    /* Scaled output to represent expected points margin (+/- 50 pts) */
    return raw_equity * 50.0f;
}

NNUEResult nnue_forward_dual(const NNUEModel *model, const float *features) {
    NNUEResult res;
    res.net_spread = 0.0f;
    res.win_probability = 0.5f;
    if (!model || !model->is_loaded) return res;

    /* Layer 1: 448 -> 128 */
    alignas(32) float l1_out[NNUE_L1_DIM];
    avx2_linear_crelu(model->w1, model->b1, features, l1_out, NNUE_L1_DIM, NNUE_INPUT_DIM);

    /* Layer 2: 128 -> 32 */
    alignas(32) float l2_out[NNUE_L2_DIM];
    avx2_linear_crelu(model->w2, model->b2, l1_out, l2_out, NNUE_L2_DIM, NNUE_L1_DIM);

    /* Layer 3 Dual Head: Score Head & Win-Probability Head */
#if defined(__AVX2__)
    __m256 v_acc1 = _mm256_setzero_ps();
    __m256 v_acc2 = _mm256_setzero_ps();

    for (int i = 0; i < NNUE_L2_DIM; i += 8) {
        __m256 v_x = _mm256_loadu_ps(&l2_out[i]);
        __m256 v_w1 = _mm256_loadu_ps(&model->w3[i]);
        __m256 v_w2 = _mm256_loadu_ps(&model->w_win[i]);
        v_acc1 = _mm256_fmadd_ps(v_w1, v_x, v_acc1);
        v_acc2 = _mm256_fmadd_ps(v_w2, v_x, v_acc2);
    }

    __m128 v_hi1 = _mm256_extractf128_ps(v_acc1, 1);
    __m128 v_lo1 = _mm256_castps256_ps128(v_acc1);
    __m128 v_sum1 = _mm_add_ps(v_lo1, v_hi1);
    v_sum1 = _mm_hadd_ps(v_sum1, v_sum1);
    v_sum1 = _mm_hadd_ps(v_sum1, v_sum1);
    float raw_equity = _mm_cvtss_f32(v_sum1) + model->b3;
    res.net_spread = raw_equity * 50.0f;

    __m128 v_hi2 = _mm256_extractf128_ps(v_acc2, 1);
    __m128 v_lo2 = _mm256_castps256_ps128(v_acc2);
    __m128 v_sum2 = _mm_add_ps(v_lo2, v_hi2);
    v_sum2 = _mm_hadd_ps(v_sum2, v_sum2);
    v_sum2 = _mm_hadd_ps(v_sum2, v_sum2);
    float raw_win = _mm_cvtss_f32(v_sum2) + model->b_win;
    res.win_probability = 1.0f / (1.0f + expf(-raw_win * 3.0f));
#else
    float raw_equity = model->b3;
    float raw_win = model->b_win;
    for (int i = 0; i < NNUE_L2_DIM; i++) {
        raw_equity += model->w3[i] * l2_out[i];
        raw_win += model->w_win[i] * l2_out[i];
    }
    res.net_spread = raw_equity * 50.0f;
    res.win_probability = 1.0f / (1.0f + expf(-raw_win * 3.0f));
#endif

    return res;
}

float nnue_evaluate_move(const NNUEModel *model, const Board *board, const Move *move,
                         const Rack *rack, int bag_count) {
    if (!model || !model->is_loaded) {
        return eval_move_static_bucketed(board, move, bag_count);
    }

    alignas(32) float features[NNUE_INPUT_DIM];
    nnue_extract_features(board, move, rack, bag_count, features);

    float nnue_pred = nnue_forward(model, features);

    /* Move immediate score is banked board score, NNUE predicts residual net equity */
    return (float)move->score + nnue_pred;
}
