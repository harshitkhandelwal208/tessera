#ifndef NNUE_H
#define NNUE_H

#include <stdint.h>
#include <stdbool.h>
#include "types.h"
#include "board.h"

#define NNUE_INPUT_DIM  448
#define NNUE_L1_DIM     128
#define NNUE_L2_DIM     32
#define NNUE_MAGIC      0x4E4E5545 /* 'NNUE' in ASCII */
#define NNUE_VERSION    1

/*
 * NNUE Architecture:
 * Input (448) -> Linear (128) -> ClippedReLU -> Linear (32) -> ClippedReLU -> Linear (1)
 *
 * ClippedReLU(x) = clamp(x, 0.0f, 1.0f)
 */
typedef struct {
    float w1[NNUE_L1_DIM * NNUE_INPUT_DIM]; /* 128 x 448 */
    float b1[NNUE_L1_DIM];                  /* 128 */
    float w2[NNUE_L2_DIM * NNUE_L1_DIM];    /* 32 x 128 */
    float b2[NNUE_L2_DIM];                  /* 32 */
    float w3[NNUE_L2_DIM];                  /* 1 x 32 */
    float b3;                               /* 1 */
    bool is_loaded;
} NNUEModel;

/* Global NNUE instance */
extern NNUEModel g_nnue;

/* Initialize NNUE weights (random Kaiming uniform or zero) */
void nnue_init_random(NNUEModel *model);

/* Load pre-trained NNUE model from binary file */
bool nnue_load(NNUEModel *model, const char *filepath);

/* Save NNUE model to binary file */
bool nnue_save(const NNUEModel *model, const char *filepath);

/*
 * Extract 448-dimensional feature vector from a board state, candidate move,
 * active rack, and bag count.
 */
void nnue_extract_features(const Board *board, const Move *move, const Rack *rack,
                          int bag_count, float *out_features);

/*
 * AVX2-Vectorized Forward Inference:
 * Computes the expected equity of the given feature vector.
 */
float nnue_forward(const NNUEModel *model, const float *features);

/*
 * Convenience function: extract features and evaluate in one fast call.
 */
float nnue_evaluate_move(const NNUEModel *model, const Board *board, const Move *move,
                         const Rack *rack, int bag_count);

#endif /* NNUE_H */
