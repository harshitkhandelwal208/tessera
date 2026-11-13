#ifndef NNUE_H
#define NNUE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>
#include "types.h"
#include "board.h"

#define NNUE_INPUT_DIM  1024
#define NNUE_L1_DIM     256
#define NNUE_L2_DIM     64
#define NNUE_MAGIC      0x4E4E5545 /* 'NNUE' in ASCII */
#define NNUE_VERSION    3

/*
 * Supercharged NNUE Architecture:
 * Input (1024) -> Linear (256) -> ClippedReLU -> Linear (64) -> ClippedReLU -> Dual-Head
 *
 * ClippedReLU(x) = clamp(x, 0.0f, 1.0f)
 */
typedef struct {
    alignas(64) float w1[NNUE_L1_DIM * NNUE_INPUT_DIM]; /* 256 x 1024 */
    alignas(64) float b1[NNUE_L1_DIM];                  /* 256 */
    alignas(64) float w2[NNUE_L2_DIM * NNUE_L1_DIM];    /* 64 x 256 */
    alignas(64) float b2[NNUE_L2_DIM];                  /* 64 */
    alignas(64) float w3[NNUE_L2_DIM];                  /* 1 x 64 (Score head) */
    float b3;                                           /* 1 */
    alignas(64) float w_win[NNUE_L2_DIM];               /* 1 x 64 (Win probability head) */
    float b_win;                                        /* 1 */
    bool is_loaded;
    /* Feature-major transposed W1 for ultra-fast incremental accumulation: w1_by_feature[feature_idx][neuron_idx] */
    alignas(64) float w1_by_feature[NNUE_INPUT_DIM][NNUE_L1_DIM];
} NNUEModel;

/* Quantized INT8 / INT16 representation for ultra-high throughput (>10M evals/sec) */
typedef struct {
    alignas(64) int8_t  w1[NNUE_L1_DIM * NNUE_INPUT_DIM];
    alignas(64) int16_t b1[NNUE_L1_DIM];
    alignas(64) int8_t  w2[NNUE_L2_DIM * NNUE_L1_DIM];
    alignas(64) int16_t b2[NNUE_L2_DIM];
    alignas(64) int16_t w3[NNUE_L2_DIM];
    int16_t b3;
    alignas(64) int16_t w_win[NNUE_L2_DIM];
    int16_t b_win;
    float scale1;
    float scale2;
    float scale3;
    bool is_loaded;
    alignas(64) int8_t  w1_by_feature[NNUE_INPUT_DIM][NNUE_L1_DIM];
} NNUEQuantizedModel;

typedef struct {
    float net_spread;       /* Expected net score margin in points */
    float win_probability;  /* P(Win) in [0.0, 1.0] */
} NNUEResult;

/* Incremental L1 Hidden Layer Accumulators (Stockfish HalfKP pattern) */
typedef struct {
    alignas(64) float vals[NNUE_L1_DIM];
} NNUEAccumulator;

typedef struct {
    alignas(64) int16_t vals[NNUE_L1_DIM];
} NNUEQuantizedAccumulator;

typedef struct {
    uint16_t index;
    float value;
} SparseFeature;

/* Global NNUE instances */
extern NNUEModel g_nnue;
extern NNUEQuantizedModel g_nnue_quantized;

/* Initialize NNUE weights (random Kaiming uniform or zero) */
void nnue_init_random(NNUEModel *model);

/* Quantize floating-point model to INT8/INT16 */
void nnue_quantize(const NNUEModel *src, NNUEQuantizedModel *dst);

/* Load pre-trained NNUE model from binary file */
bool nnue_load(NNUEModel *model, const char *filepath);

/* Save NNUE model to binary file */
bool nnue_save(const NNUEModel *model, const char *filepath);

/*
 * Extract 1024-dimensional feature vector from a board state, candidate move,
 * active rack, bag count, and score differential.
 */
void nnue_extract_features_with_score(const Board *board, const Move *move, const Rack *rack,
                                      int bag_count, int score_diff, float *out_features);
void nnue_extract_features(const Board *board, const Move *move, const Rack *rack,
                          int bag_count, float *out_features);

/*
 * AVX2/AVX-512 Vectorized Forward Inference:
 * Computes the expected equity of the given feature vector.
 */
float nnue_forward(const NNUEModel *model, const float *features);

/*
 * Quantized AVX2/AVX-512-VNNI Forward Inference (>10M evals/sec).
 */
float nnue_forward_quantized(const NNUEQuantizedModel *model, const float *features);

/*
 * Dual-Head Forward Inference:
 * Simultaneously computes expected net point margin and win probability.
 */
NNUEResult nnue_forward_dual(const NNUEModel *model, const float *features);

/*
 * Convenience function: extract features and evaluate in one fast call.
 */
float nnue_evaluate_move(const NNUEModel *model, const Board *board, const Move *move,
                         const Rack *rack, int bag_count);

/*
 * Convenience function: extract features with live score differential and evaluate in one fast call.
 */
float nnue_evaluate_move_with_score(const NNUEModel *model, const Board *board, const Move *move,
                                    const Rack *rack, int bag_count, int score_diff);

/*
 * Incremental Accumulator Functions (Stockfish HalfKP pattern)
 */
void nnue_accumulator_build_base(const NNUEModel *model, const Board *board, const Rack *rack,
                                 int bag_count, int score_diff, NNUEAccumulator *base_acc);

int nnue_extract_move_sparse_features(const Board *board, const Move *move,
                                      int bag_count, int score_diff, SparseFeature *out_sparse);

void nnue_accumulator_apply_sparse(const NNUEModel *model, const NNUEAccumulator *base,
                                   const SparseFeature *features, int num_features,
                                   NNUEAccumulator *out_acc);

float nnue_forward_accumulator(const NNUEModel *model, const NNUEAccumulator *acc);

NNUEResult nnue_forward_accumulator_dual(const NNUEModel *model, const NNUEAccumulator *acc);

float nnue_forward_move_incremental(const NNUEModel *model, const NNUEAccumulator *base_acc,
                                    const Board *board, const Move *move,
                                    int bag_count, int score_diff);

NNUEResult nnue_forward_move_incremental_dual(const NNUEModel *model, const NNUEAccumulator *base_acc,
                                              const Board *board, const Move *move,
                                              int bag_count, int score_diff);

#endif /* NNUE_H */
