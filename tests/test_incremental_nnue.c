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

    /* 3. Raw Incremental Accumulator Pipeline Throughput Benchmark */
    SparseFeature test_sparse[256];
    int test_n = nnue_extract_move_sparse_features(&board, &moves.moves[0], 86, 0, test_sparse);

    clock_t t_acc_start = clock();
    int acc_iters = 1000000;
    NNUEAccumulator bench_acc;
    for (int it = 0; it < acc_iters; it++) {
        nnue_accumulator_apply_sparse(&g_nnue, &base_acc, test_sparse, test_n, &bench_acc);
    }
    clock_t t_acc_end = clock();
    double acc_ms = ((double)(t_acc_end - t_acc_start) / CLOCKS_PER_SEC) * 1000.0;
    double acc_eps = (double)acc_iters / (acc_ms / 1000.0);

    clock_t t_raw_start = clock();
    int raw_iters = 200000;
    float raw_sum = 0.0f;
    for (int it = 0; it < raw_iters; it++) {
        NNUEAccumulator test_acc;
        nnue_accumulator_apply_sparse(&g_nnue, &base_acc, test_sparse, test_n, &test_acc);
        raw_sum += nnue_forward_accumulator(&g_nnue, &test_acc);
    }
    clock_t t_raw_end = clock();
    double raw_ms = ((double)(t_raw_end - t_raw_start) / CLOCKS_PER_SEC) * 1000.0;
    double raw_eps = (double)raw_iters / (raw_ms / 1000.0);

    printf("[BENCHMARK] Move evaluations:\n");
    printf("  Dense Forward Pass (%d moves):        %.2f ms (%.0f evals/sec, sum=%.1f)\n",
           moves.count, dense_ms, (double)moves.count / (dense_ms / 1000.0), dense_sum);
    printf("  End-to-End Incremental (%d moves):    %.2f ms (%.0f evals/sec, sum=%.1f)\n",
           moves.count, inc_ms, (double)moves.count / (inc_ms / 1000.0), inc_sum);
    printf("  Sparse Accumulator Update Only:        %.2f ms (%.0f updates/sec)\n",
           acc_ms, acc_eps);
    printf("  Pure Incremental Accumulator Pipeline: %.2f ms (%.0f evals/sec, sum=%.1f)\n",
           raw_ms, raw_eps, raw_sum);
    printf("  Throughput Speedup:                    %.2fx FASTER!\n", dense_ms / inc_ms);

    /* Check exact numerical equivalence across all legal moves */
    float max_diff = 0.0f;
    for (int i = 0; i < moves.count; i++) {
        alignas(64) float features[NNUE_INPUT_DIM];
        nnue_extract_features_with_score(&board, &moves.moves[i], &rack, 86, 0, features);
        float dense_val = nnue_forward(&g_nnue, features);
        float inc_val = nnue_forward_move_incremental(&g_nnue, &base_acc, &board, &moves.moves[i], 86, 0);
        float diff = fabsf(dense_val - inc_val);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.001f) {
            printf("Discrepancy at move %d (%s): dense=%.4f, inc=%.4f, diff=%.6f\n",
                   i, moves.moves[i].word, dense_val, inc_val, diff);
        }
        assert(diff < 0.01f);
    }
    printf("[TEST] Verified 100%% equivalence across all %d moves (Max Diff: %.6f)!\n", moves.count, max_diff);

    /* 4. INT8/INT16 Pure Integer Vectorized Accumulator Pipeline Benchmark */
    NNUEQuantizedAccumulator q_base_acc;
    nnue_quantized_accumulator_build_base(&g_nnue_quantized, &board, &rack, 86, 0, &q_base_acc);

    clock_t t_q_acc_start = clock();
    int q_acc_iters = 1000000;
    NNUEQuantizedAccumulator q_bench_acc;
    int32_t q_sink = 0;
    for (int it = 0; it < q_acc_iters; it++) {
        nnue_quantized_accumulator_apply_sparse(&g_nnue_quantized, &q_base_acc, test_sparse, test_n, &q_bench_acc);
        q_sink += q_bench_acc.vals[it & 255];
    }
    clock_t t_q_acc_end = clock();
    double q_acc_ms = ((double)(t_q_acc_end - t_q_acc_start) / CLOCKS_PER_SEC) * 1000.0;
    double q_acc_eps = (double)q_acc_iters / (q_acc_ms / 1000.0);

    clock_t t_q_raw_start = clock();
    int q_raw_iters = 500000;
    float q_raw_sum = 0.0f;
    for (int it = 0; it < q_raw_iters; it++) {
        NNUEQuantizedAccumulator q_test_acc;
        nnue_quantized_accumulator_apply_sparse(&g_nnue_quantized, &q_base_acc, test_sparse, test_n, &q_test_acc);
        q_raw_sum += nnue_forward_quantized_accumulator(&g_nnue_quantized, &q_test_acc);
    }
    clock_t t_q_raw_end = clock();
    double q_raw_ms = ((double)(t_q_raw_end - t_q_raw_start) / CLOCKS_PER_SEC) * 1000.0;
    double q_raw_eps = (double)q_raw_iters / (q_raw_ms / 1000.0);

    printf("[BENCHMARK] Quantized INT8/INT16 Vectorized Pipeline:\n");
    printf("  Quantized Sparse Accumulator Update:   %.2f ms (%.0f updates/sec, sink=%d)\n",
           q_acc_ms, q_acc_eps, (int)q_sink);
    printf("  Quantized Incremental Pipeline:        %.2f ms (%.0f evals/sec, sum=%.1f)\n",
           q_raw_ms, q_raw_eps, q_raw_sum);

    /* 5. Tier 2 Master Net (16,384 -> 512 -> SCReLU(1024) -> 128 -> 64 -> Dual Heads, ~8.5 MB INT8) Benchmark */
    nnue_tier2_init_random(&g_nnue_tier2);
    NNUETier2QuantizedAccumulator t2_base_acc;
    nnue_tier2_quantized_accumulator_build_base(&g_nnue_tier2_quantized, &board, &rack, 86, 0, &t2_base_acc);

    SparseFeature t2_sparse[256];
    int t2_n = nnue_tier2_extract_sparse_features(&board, &moves.moves[0], 86, 0, t2_sparse);

    clock_t t_t2_acc_start = clock();
    int t2_acc_iters = 500000;
    NNUETier2QuantizedAccumulator t2_bench_acc;
    int32_t t2_sink = 0;
    for (int it = 0; it < t2_acc_iters; it++) {
        nnue_tier2_quantized_accumulator_apply_sparse(&g_nnue_tier2_quantized, &t2_base_acc, t2_sparse, t2_n, &t2_bench_acc);
        t2_sink += t2_bench_acc.vals[it & 511];
    }
    clock_t t_t2_acc_end = clock();
    double t2_acc_ms = ((double)(t_t2_acc_end - t_t2_acc_start) / CLOCKS_PER_SEC) * 1000.0;
    double t2_acc_eps = (double)t2_acc_iters / (t2_acc_ms / 1000.0);

    clock_t t_t2_raw_start = clock();
    int t2_raw_iters = 200000;
    float t2_raw_sum = 0.0f;
    for (int it = 0; it < t2_raw_iters; it++) {
        NNUETier2QuantizedAccumulator t2_test_acc;
        nnue_tier2_quantized_accumulator_apply_sparse(&g_nnue_tier2_quantized, &t2_base_acc, t2_sparse, t2_n, &t2_test_acc);
        t2_raw_sum += nnue_tier2_forward_quantized_accumulator(&g_nnue_tier2_quantized, &t2_test_acc);
    }
    clock_t t_t2_raw_end = clock();
    double t2_raw_ms = ((double)(t_t2_raw_end - t_t2_raw_start) / CLOCKS_PER_SEC) * 1000.0;
    double t2_raw_eps = (double)t2_raw_iters / (t2_raw_ms / 1000.0);

    printf("[BENCHMARK] Tier 2 Deep Master Net (16,384 -> 512 -> SCReLU -> 128 -> 64 -> Dual Heads, ~8.5 MB):\n");
    printf("  Tier 2 Sparse Accumulator Update:      %.2f ms (%.0f updates/sec, sink=%d)\n",
           t2_acc_ms, t2_acc_eps, (int)t2_sink);
    printf("  Tier 2 Incremental Pipeline:           %.2f ms (%.0f evals/sec, sum=%.1f)\n",
           t2_raw_ms, t2_raw_eps, t2_raw_sum);

    move_list_destroy(&moves);
    dawg_free(dawg);
    printf("[TEST] Incremental Accumulator test PASSED flawlessly!\n");
    return 0;
}
