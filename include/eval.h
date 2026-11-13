#ifndef EVAL_H
#define EVAL_H

#include "types.h"
#include "board.h"
#include "dict.h"
#include "nnue.h"

/* 8 Game-Phase Buckets (inspired by Reckless bucketing) */
typedef enum {
    PHASE_BUCKET_OPENING      = 7, /* 80-100 bag tiles */
    PHASE_BUCKET_EARLY        = 6, /* 60-79 bag tiles */
    PHASE_BUCKET_MIDGAME      = 5, /* 40-59 bag tiles */
    PHASE_BUCKET_LATE_MID     = 4, /* 20-39 bag tiles */
    PHASE_BUCKET_PRE_ENDGAME  = 3, /* 10-19 bag tiles */
    PHASE_BUCKET_EXHAUSTION   = 2, /* 1-9 bag tiles */
    PHASE_BUCKET_ENDGAME      = 1, /* 0 bag tiles */
    PHASE_BUCKET_PLAYOUT      = 0  /* 1-ply outplay */
} PhaseBucket;

/* 4 Board-Openness Output Buckets */
typedef enum {
    OPENNESS_CLOSED   = 0,
    OPENNESS_STANDARD = 1,
    OPENNESS_EXPOSED  = 2,
    OPENNESS_WIDE     = 3
} OpennessBucket;

/* Dual-Accumulator Evaluator State */
typedef struct {
    float board_acc;  /* Board accumulator: word score + premium capture + defensive threat */
    float rack_acc;   /* Rack leave accumulator: single tile + pairwise synergy + VC + stems */
    float net_equity; /* board_acc + rack_acc */
} DualAccumulator;

/* Bucket classifier helpers */
PhaseBucket    eval_classify_phase(int bag_count);
OpennessBucket eval_classify_openness(const Board *board);
const char*    eval_phase_name(PhaseBucket phase);
const char*    eval_openness_name(OpennessBucket openness);

/*
 * Dual-Accumulator Rack Leave Evaluation:
 * Evaluates the rack leave equity with 8 phase buckets, pairwise synergy kernels,
 * higher-order super-leave stem tables, and duplicate/toxic tile penalties.
 */
float eval_rack_leave_bucketed_with_score(const char *leave, PhaseBucket phase, int score_diff);
float eval_rack_leave_bucketed(const char *leave, PhaseBucket phase);
float eval_rack_leave(const char *leave);
float eval_move_heuristic_fast(const Board *board, const Move *move, int bag_count);

/*
 * Board Accumulator:
 * Evaluates premium capture, defensive risk, and floating vowel vulnerability.
 */
float eval_board_risk_bucketed(const Board *board, const Move *move, OpennessBucket openness);
float eval_board_risk_with_score(const Board *board, const Move *move, OpennessBucket openness, int score_diff);
float eval_board_risk(const Board *board, const Move *move);

/*
 * Tournament Win Probability and Score Margin Controls:
 */
float eval_win_probability(int score_diff, int bag_count);
float eval_score_margin_risk_multiplier(int score_diff);

/*
 * Dual-Accumulator Move Evaluation:
 * Computes the total static evaluation of a candidate move.
 */
DualAccumulator eval_move_accumulate_with_score(const Board *board, const Move *move, int bag_count, int score_diff);
DualAccumulator eval_move_accumulate_with_score_incremental(const Board *board, const Move *move,
                                                           const NNUEAccumulator *base_acc,
                                                           int bag_count, int score_diff);
DualAccumulator eval_move_accumulate(const Board *board, const Move *move, int bag_count);
float eval_move_static_with_score(const Board *board, const Move *move, int bag_count, int score_diff);
float eval_move_static_incremental(const Board *board, const Move *move,
                                  const NNUEAccumulator *base_acc,
                                  int bag_count, int score_diff);
float eval_move_static_bucketed(const Board *board, const Move *move, int bag_count);
float eval_move_static(const Board *board, const Move *move);

/*
 * Exact Sub-rack Exchange Optimizer:
 * Evaluates all 2^k subsets of the rack to find the optimal keep/discard split.
 */
bool eval_should_exchange(const Board *board, const Rack *rack, int bag_count,
                         float best_play_equity, Move *out_exchange_move);

/* Sort moves by static equity descending */
void eval_sort_moves(MoveList *list);

#endif /* EVAL_H */
