#include "dict.h"
#include "board.h"
#include "movegen.h"
#include "eval.h"
#include "nnue.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define MAX_GAME_TURNS 60
#define GAMMA 0.92f

typedef struct {
    float features[NNUE_INPUT_DIM];
    float target; /* Normalized residual equity in [-1.0, +1.0] */
} TrainingSample;

typedef struct {
    Board board_before;
    Move move;
    Rack rack;
    int bag_count;
    int player; /* 0 = Player A, 1 = Player B */
    int score;
} TurnRecord;

static Move pick_selfplay_move(const Board *board, const Dawg *dawg, const Rack *rack,
                               int bag_count, unsigned int *seed) {
    MoveList moves = {0};
    movegen_generate_all(board, dawg, rack, &moves);

    if (moves.count == 0) {
        move_list_destroy(&moves);
        Move pass;
        memset(&pass, 0, sizeof(Move));
        pass.type = MOVE_PASS;
        return pass;
    }

    /* Evaluate candidate moves */
    for (int i = 0; i < moves.count; i++) {
        moves.moves[i].leave_equity = eval_move_static_bucketed(board, &moves.moves[i], bag_count);
    }
    eval_sort_moves(&moves);

    /* Epsilon-softmax exploration over top 6 candidate moves */
    int num_cands = moves.count < 6 ? moves.count : 6;
    Move chosen = moves.moves[0];

    if (num_cands > 1) {
        float r = (float)rand_r(seed) / (float)RAND_MAX;
        if (r >= 0.70f) {
            /* 30% temperature exploration over top candidates */
            float best_eq = moves.moves[0].leave_equity;
            float probs[6];
            float sum_p = 0.0f;
            for (int i = 0; i < num_cands; i++) {
                probs[i] = expf((moves.moves[i].leave_equity - best_eq) / 3.0f);
                sum_p += probs[i];
            }

            float pick_val = ((float)rand_r(seed) / (float)RAND_MAX) * sum_p;
            float accum = 0.0f;
            for (int i = 0; i < num_cands; i++) {
                accum += probs[i];
                if (accum >= pick_val) {
                    chosen = moves.moves[i];
                    break;
                }
            }
        }
    }

    move_list_destroy(&moves);
    return chosen;
}

int main(int argc, char *argv[]) {
    int num_games = 10000;
    const char *out_path = "data/train_data.bin";

    if (argc >= 2) num_games = atoi(argv[1]);
    if (argc >= 3) out_path = argv[2];

    printf("======================================================================\n");
    printf("   TESSERA HIGH-THROUGHPUT SELF-PLAY DATASET GENERATOR (OPENMP)       \n");
    printf("======================================================================\n");
    printf("Lexicon: SOWPODS | Target Games: %d | Output: %s\n", num_games, out_path);
    printf("Simulating self-play matches across %d CPU threads...\n\n", omp_get_max_threads());

    Dawg *dawg = dawg_load_binary("data/dawg.bin");
    if (!dawg) {
        fprintf(stderr, "Error loading data/dawg.bin\n");
        return 1;
    }

    if (nnue_load(&g_nnue, "data/nnue.bin")) {
        printf("[INFO] Loaded NNUE model from data/nnue.bin for neural self-play generation!\n");
    } else {
        printf("[INFO] No NNUE model found; falling back to 8-phase heuristic dual-accumulator.\n");
    }

    FILE *fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "Error opening output file %s\n", out_path);
        dawg_free(dawg);
        return 1;
    }

    clock_t start_time = clock();
    long total_samples = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int seed = (unsigned int)(1337 + tid * 10007 + time(NULL));

        #pragma omp for schedule(dynamic, 64) reduction(+:total_samples)
        for (int g = 0; g < num_games; g++) {
            Board board;
            board_init(&board);

            TileBag bag;
            bag_init(&bag);
            bag_shuffle(&bag);

            Rack racks[2];
            rack_init(&racks[0]);
            rack_init(&racks[1]);

            bag_draw_tiles(&bag, &racks[0], RACK_CAPACITY);
            bag_draw_tiles(&bag, &racks[1], RACK_CAPACITY);

            TurnRecord turns[MAX_GAME_TURNS];
            int turn_count = 0;
            int current_player = g % 2;
            int consecutive_passes = 0;

            while (consecutive_passes < 2 && turn_count < MAX_GAME_TURNS) {
                Rack *active_rack = &racks[current_player];
                TurnRecord *rec = &turns[turn_count];
                rec->board_before = board;
                rec->rack = *active_rack;
                rec->bag_count = bag.count;
                rec->player = current_player;

                Move m = pick_selfplay_move(&board, dawg, active_rack, bag.count, &seed);
                rec->move = m;
                rec->score = m.score;

                if (m.type == MOVE_PLAY) {
                    consecutive_passes = 0;
                    board_apply_move(&board, &m);

                    for (int p = 0; p < m.num_placed; p++) {
                        char t = m.placed[p].is_blank ? '?' : m.placed[p].letter;
                        rack_remove_tile(active_rack, t);
                    }
                    bag_draw_tiles(&bag, active_rack, m.num_placed);

                    turn_count++;
                    if (active_rack->count == 0 && bag.count == 0) break;
                } else {
                    consecutive_passes++;
                    turn_count++;
                }

                current_player = 1 - current_player;
            }

            /* Compute final game score and outcome for both players */
            int p0_score = 0, p1_score = 0;
            for (int t = 0; t < turn_count; t++) {
                if (turns[t].move.type == MOVE_PLAY) {
                    if (turns[t].player == 0) p0_score += turns[t].score;
                    else p1_score += turns[t].score;
                }
            }
            int p0_pen = 0, p1_pen = 0;
            for (int i = 0; i < racks[0].count; i++) {
                char c = racks[0].tiles[i];
                if (c >= 'A' && c <= 'Z') p0_pen += TILE_VALUES[c - 'A'];
            }
            for (int i = 0; i < racks[1].count; i++) {
                char c = racks[1].tiles[i];
                if (c >= 'A' && c <= 'Z') p1_pen += TILE_VALUES[c - 'A'];
            }
            if (racks[0].count == 0 && bag.count == 0) {
                p0_score += p1_pen;
                p1_score -= p1_pen;
            } else if (racks[1].count == 0 && bag.count == 0) {
                p1_score += p0_pen;
                p0_score -= p0_pen;
            } else {
                p0_score -= p0_pen;
                p1_score -= p1_pen;
            }
            float final_margin_p0 = (float)(p0_score - p1_score);

            /* Extract training samples based on empirical next-turn equity */
            TrainingSample local_samples[MAX_GAME_TURNS];
            int num_local = 0;

            for (int t = 0; t < turn_count; t++) {
                if (turns[t].move.type != MOVE_PLAY) continue;

                float active_next_score = 26.0f;
                float opp_next_score = 26.0f;

                /* Turn t+1 is the opponent's counterplay */
                if (t + 1 < turn_count && turns[t + 1].move.type == MOVE_PLAY) {
                    opp_next_score = (float)turns[t + 1].score;
                }
                /* Turn t+2 is the active player's realization of the leave */
                if (t + 2 < turn_count && turns[t + 2].move.type == MOVE_PLAY) {
                    active_next_score = (float)turns[t + 2].score;
                } else {
                    PhaseBucket pb = eval_classify_phase(turns[t].bag_count);
                    active_next_score = 26.0f + eval_rack_leave_bucketed(turns[t].move.leave, pb);
                }

                /*
                 * Residual Equity:
                 * (ActivePlayerNextTurnSurplus) - 0.4 * (OpponentCounterplaySurplus)
                 */
                float residual_equity = (active_next_score - 26.0f) - 0.40f * (opp_next_score - 26.0f);

                /* Normalized immediate target equity in [-1.0, +1.0] */
                float norm_immediate = fminf(1.0f, fmaxf(-1.0f, residual_equity / 50.0f));

                /* Normalized whole-game victory margin in [-1.0, +1.0] */
                float player_sign = (turns[t].player == 0) ? 1.0f : -1.0f;
                float norm_margin = fminf(1.0f, fmaxf(-1.0f, (player_sign * final_margin_p0) / 60.0f));

                /* Calibrated TD-blended target: sharp tactical surplus (85%) + gentle game margin (15%) */
                float blended_target = (0.85f * norm_immediate) + (0.15f * norm_margin);

                nnue_extract_features(&turns[t].board_before, &turns[t].move,
                                      &turns[t].rack, turns[t].bag_count,
                                      local_samples[num_local].features);
                local_samples[num_local].target = blended_target;
                num_local++;
            }

            #pragma omp critical
            {
                if (num_local > 0) {
                    fwrite(local_samples, sizeof(TrainingSample), num_local, fout);
                }
            }

            total_samples += num_local;
        }
    }

    fclose(fout);
    dawg_free(dawg);

    double elapsed_s = (double)(clock() - start_time) / (double)CLOCKS_PER_SEC;
    printf("\n>>> Self-play data generation complete!\n");
    printf("  Total Samples Generated: %ld\n", total_samples);
    printf("  Dataset File Size:        %.2f MB\n", (double)(total_samples * sizeof(TrainingSample)) / (1024.0 * 1024.0));
    printf("  Total Wall Time:          %.2f s (%.0f samples/sec)\n", elapsed_s, (double)total_samples / elapsed_s);
    printf("======================================================================\n");

    return 0;
}
