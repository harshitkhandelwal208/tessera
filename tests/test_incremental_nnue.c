#include "dict.h"
#include "board.h"
#include "movegen.h"
#include "eval.h"
#include "nnue.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>

int main(void) {
    printf("[TEST] Testing Stockfish-Style Incremental NNUE Accumulator...\n");

    Dawg *dawg = dawg_load_binary("data/dawg.bin");
    assert(dawg != NULL);

    bool loaded = nnue_load(&g_nnue, "data/nnue.bin");
    if (!loaded) {
        printf("[TEST] Generating random NNUE model for test...\n");
        nnue_init_random(&g_nnue);
    }

    Board board;
    board_init(&board);

    Rack rack;
    rack_init(&rack);
    rack_add_tile(&rack, 'R');
    rack_add_tile(&rack, 'E');
    rack_add_tile(&rack, 'T');
    rack_add_tile(&rack, 'A');
    rack_add_tile(&rack, 'I');
    rack_add_tile(&rack, 'N');
    rack_add_tile(&rack, 'S');

    MoveList moves = {0};
    movegen_generate_all(&board, dawg, &rack, &moves);
    printf("[TEST] Generated %d legal moves from opening position.\n", moves.count);
    assert(moves.count > 0);

    /* 1. Build Base Accumulator */
    clock_t t0 = clock();
    NNUEAccumulator base_acc;
    nnue_accumulator_build_base(&g_nnue, &board, &rack, 86, 0, &base_acc);
    clock_t t1 = clock();
    double base_ms = ((double)(t1 - t0) / CLOCKS_PER_SEC) * 1000.0;
    printf("[TEST] Built base accumulator in %.4f ms!\n", base_ms);

    /* 2. Benchmark Full Dense vs Incremental */
    clock_t t_dense_start = clock();
    float dense_sum = 0.0f;
    for (int i = 0; i < moves.count; i++) {
        alignas(64) float features[NNUE_INPUT_DIM];
        nnue_extract_features_with_score(&board, &moves.moves[i], &rack, 86, 0, features);
        dense_sum += nnue_forward(&g_nnue, features);
    }
    clock_t t_dense_end = clock();
    double dense_ms = ((double)(t_dense_end - t_dense_start) / CLOCKS_PER_SEC) * 1000.0;

    clock_t t_inc_start = clock();
    float inc_sum = 0.0f;
    for (int i = 0; i < moves.count; i++) {
        inc_sum += nnue_forward_move_incremental(&g_nnue, &base_acc, &board, &moves.moves[i], 86, 0);
    }
    clock_t t_inc_end = clock();
    double inc_ms = ((double)(t_inc_end - t_inc_start) / CLOCKS_PER_SEC) * 1000.0;

    printf("[BENCHMARK] %d move evaluations:\n", moves.count);
    printf("  Dense Forward Pass:       %.2f ms (%.0f evals/sec, sum=%.1f)\n",
           dense_ms, (double)moves.count / (dense_ms / 1000.0), dense_sum);
    printf("  Incremental Accumulator:  %.2f ms (%.0f evals/sec, sum=%.1f)\n",
           inc_ms, (double)moves.count / (inc_ms / 1000.0), inc_sum);
    printf("  Throughput Speedup:       %.2fx FASTER!\n", dense_ms / inc_ms);

    /* Check numerical equivalence */
    for (int i = 0; i < 50 && i < moves.count; i++) {
        alignas(64) float features[NNUE_INPUT_DIM];
        nnue_extract_features_with_score(&board, &moves.moves[i], &rack, 86, 0, features);
        float dense_val = nnue_forward(&g_nnue, features);
        float inc_val = nnue_forward_move_incremental(&g_nnue, &base_acc, &board, &moves.moves[i], 86, 0);
        float diff = fabsf(dense_val - inc_val);
        if (diff > 0.05f) {
            printf("Discrepancy at move %d (%s): dense=%.3f, inc=%.3f, diff=%.4f\n",
                   i, moves.moves[i].word, dense_val, inc_val, diff);
        }
        assert(diff < 0.5f);
    }

    move_list_destroy(&moves);
    dawg_free(dawg);
    printf("[TEST] Incremental Accumulator test PASSED flawlessly!\n");
    return 0;
}
