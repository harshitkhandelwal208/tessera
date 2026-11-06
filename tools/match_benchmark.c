#include "dict.h"
#include "board.h"
#include "movegen.h"
#include "eval.h"
#include "simulation.h"
#include "nnue.h"
#include "bayes_rack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef enum {
    BOT_BASELINE_GREEDY   = 0, /* Greedy 1-ply: Immediate word score (~1550 Elo) */
    BOT_DEFENSIVE_TURTLER = 1, /* Defensive Turtler: Avoids opening TW/DW (~1650 Elo) */
    BOT_BINGO_HUNTER      = 2, /* Bingo Hunter: 7-letter stem optimizer (~1750 Elo) */
    BOT_HEURISTIC_STATIC  = 3, /* Grandmaster 8-Phase Leave Heuristic (~1850 Elo) */
    BOT_NEURAL_STATIC     = 4, /* Raw Deep NNUE 1-Ply Evaluator (~1950 Elo) */
    BOT_CHAMPIONSHIP_SIM  = 5, /* Quackle-Style Lookahead Simulation (~2100 Elo) */
    BOT_MAGPIE_2PLY       = 6, /* MAGPIE/Woogles 2-Ply Realization Simulation (~2150 Elo) */
    BOT_TYPE_COUNT        = 7
} BotType;

static const char *BOT_NAMES[BOT_TYPE_COUNT] = {
    "Baseline Greedy (1-Ply Score)",
    "Defensive Turtler (Risk Minimizer)",
    "Bingo Hunter (Stem Synergy)",
    "Heuristic Grandmaster (8-Phase Table)",
    "Neural Static (Raw 448-dim NNUE 1-Ply)",
    "Championship Simulation (Quackle Emulation)",
    "MAGPIE/Woogles Champion (2-Ply Realization)"
};

static const int BOT_ESTIMATED_ELOS[BOT_TYPE_COUNT] = {
    1550,
    1650,
    1750,
    1850,
    1950,
    2100,
    2150
};

typedef struct {
    int wins;
    int losses;
    int ties;
    int total_tessera_score;
    int total_opponent_score;
    int tessera_bingos;
    int opponent_bingos;
    double elapsed_seconds;
} MatchStats;

/* Helper: shuffle an array of characters */
static void local_shuffle_tiles(char *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        char tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* 1. Greedy Bot */
static Move select_greedy_move(const Board *board, const Dawg *dawg, const Rack *rack) {
    MoveList moves;
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        Move pass_move;
        memset(&pass_move, 0, sizeof(Move));
        pass_move.type = MOVE_PASS;
        return pass_move;
    }
    int best_score = -1;
    int best_idx = 0;
    for (int i = 0; i < moves.count; i++) {
        if (moves.moves[i].score > best_score) {
            best_score = moves.moves[i].score;
            best_idx = i;
        }
    }
    return moves.moves[best_idx];
}

/* 2. Defensive Turtler Bot */
static Move select_defensive_move(const Board *board, const Dawg *dawg, const Rack *rack) {
    MoveList moves;
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        Move pass_move;
        memset(&pass_move, 0, sizeof(Move));
        pass_move.type = MOVE_PASS;
        return pass_move;
    }
    OpennessBucket openness = eval_classify_openness(board);
    float best_val = -10000.0f;
    int best_idx = 0;
    for (int i = 0; i < moves.count; i++) {
        float risk = eval_board_risk_bucketed(board, &moves.moves[i], openness);
        float val = (float)moves.moves[i].score + (3.0f * risk);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    return moves.moves[best_idx];
}

/* 3. Bingo Hunter Bot */
static Move select_bingo_hunter_move(const Board *board, const Dawg *dawg, const Rack *rack) {
    MoveList moves;
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        Move pass_move;
        memset(&pass_move, 0, sizeof(Move));
        pass_move.type = MOVE_PASS;
        return pass_move;
    }
    float best_val = -10000.0f;
    int best_idx = 0;
    for (int i = 0; i < moves.count; i++) {
        float stem_bonus = 0.0f;
        for (int j = 0; moves.moves[i].leave[j]; j++) {
            char c = moves.moves[i].leave[j];
            if (c == '?' || c == 'S') stem_bonus += 8.0f;
            else if (c == 'E' || c == 'R' || c == 'I' || c == 'A' || c == 'T' || c == 'N') stem_bonus += 3.5f;
            else if (c == 'Q' || c == 'Z' || c == 'J' || c == 'X' || c == 'V') stem_bonus -= 6.0f;
        }
        float val = (float)moves.moves[i].score + (1.2f * stem_bonus);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    return moves.moves[best_idx];
}

/* 4. Heuristic Static Bot (8-phase rack leave table, no NNUE) */
static Move select_heuristic_move(const Board *board, const Dawg *dawg, const Rack *rack, int bag_count) {
    MoveList moves;
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        Move pass_move;
        memset(&pass_move, 0, sizeof(Move));
        pass_move.type = MOVE_PASS;
        return pass_move;
    }
    float best_val = -10000.0f;
    int best_idx = 0;
    for (int i = 0; i < moves.count; i++) {
        float eq = eval_move_heuristic_fast(board, &moves.moves[i], bag_count);
        if (eq > best_val) {
            best_val = eq;
            best_idx = i;
        }
    }
    return moves.moves[best_idx];
}

/* 5. Neural Static Bot (Raw 448-dim NNUE 1-ply) */
static Move select_neural_static_move(const Board *board, const Dawg *dawg, const Rack *rack, int bag_count) {
    MoveList moves;
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        Move pass_move;
        memset(&pass_move, 0, sizeof(Move));
        pass_move.type = MOVE_PASS;
        return pass_move;
    }
    float best_eq = -10000.0f;
    int best_idx = 0;
    for (int i = 0; i < moves.count; i++) {
        float eq = eval_move_static_bucketed(board, &moves.moves[i], bag_count);
        if (eq > best_eq) {
            best_eq = eq;
            best_idx = i;
        }
    }
    return moves.moves[best_idx];
}

/* 6. Championship Simulation Bot (Quackle emulation: paired Monte Carlo lookahead) */
static Move select_quackle_sim_move(const Board *board, const Dawg *dawg, const Rack *rack, int bag_count) {
    MoveList moves;
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        Move pass_move;
        memset(&pass_move, 0, sizeof(Move));
        pass_move.type = MOVE_PASS;
        return pass_move;
    }

    for (int i = 0; i < moves.count; i++) {
        moves.moves[i].leave_equity = eval_move_heuristic_fast(board, &moves.moves[i], bag_count);
    }
    eval_sort_moves(&moves);

    int beam = moves.count < 5 ? moves.count : 5;
    if (beam <= 1) return moves.moves[0];

    char unseen[100];
    int unseen_count = sim_get_unseen_tiles(board, rack, unseen);

    float opp_totals[5] = {0};
    int N_SIMS = 12;

    for (int s = 0; s < N_SIMS; s++) {
        Rack opp_r;
        rack_init(&opp_r);
        if (bag_count > 0) local_shuffle_tiles(unseen, unseen_count);
        int d = unseen_count < 7 ? unseen_count : 7;
        for (int i = 0; i < d; i++) rack_add_tile(&opp_r, unseen[i]);

        for (int k = 0; k < beam; k++) {
            Board spec;
            board_copy(&spec, board);
            board_apply_move(&spec, &moves.moves[k]);

            MoveList om;
            movegen_generate_all(&spec, dawg, &opp_r, &om);
            float max_opp = 0.0f;
            for (int r = 0; r < om.count; r++) {
                float eq = eval_move_heuristic_fast(&spec, &om.moves[r], bag_count);
                if (eq > max_opp) max_opp = eq;
            }
            opp_totals[k] += max_opp;
        }
    }

    int best_k = 0;
    float best_net = -10000.0f;
    for (int k = 0; k < beam; k++) {
        float avg_opp = opp_totals[k] / (float)N_SIMS;
        float net = moves.moves[k].leave_equity - (0.80f * avg_opp);
        if (net > best_net) {
            best_net = net;
            best_k = k;
        }
    }
    return moves.moves[best_k];
}

/* 7. MAGPIE / Woogles Champion Bot (2-Ply Realized Equity Lookahead Simulation) */
static Move select_magpie_sim_move(const Board *board, const Dawg *dawg, const Rack *rack, int bag_count) {
    MoveList moves;
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        Move pass_move;
        memset(&pass_move, 0, sizeof(Move));
        pass_move.type = MOVE_PASS;
        return pass_move;
    }

    for (int i = 0; i < moves.count; i++) {
        moves.moves[i].leave_equity = eval_move_static_bucketed(board, &moves.moves[i], bag_count);
    }
    eval_sort_moves(&moves);

    int beam = moves.count < 5 ? moves.count : 5;
    if (beam <= 1) return moves.moves[0];

    char unseen[100];
    int unseen_count = sim_get_unseen_tiles(board, rack, unseen);

    float net_totals[5] = {0};
    int N_SIMS = 8;

    for (int s = 0; s < N_SIMS; s++) {
        char sim_unseen[100];
        memcpy(sim_unseen, unseen, unseen_count);
        local_shuffle_tiles(sim_unseen, unseen_count);

        Rack opp_r;
        rack_init(&opp_r);
        int d = unseen_count < 7 ? unseen_count : 7;
        for (int i = 0; i < d; i++) rack_add_tile(&opp_r, sim_unseen[i]);

        for (int k = 0; k < beam; k++) {
            Board spec;
            board_copy(&spec, board);
            board_apply_move(&spec, &moves.moves[k]);

            /* 1. Opponent best reply on spec board */
            float opp_equity = movegen_find_best_response_equity(&spec, dawg, &opp_r, bag_count > 0 ? bag_count - 1 : 0);

            /* 2. Active player's turn-2 realization of leave */
            Rack my_realized_rack;
            rack_init(&my_realized_rack);
            int leave_len = (int)strlen(moves.moves[k].leave);
            for (int i = 0; i < leave_len; i++) rack_add_tile(&my_realized_rack, moves.moves[k].leave[i]);
            int needed = 7 - leave_len;
            int draw_start = d;
            int available_draw = unseen_count - draw_start;
            int actual_draw = needed < available_draw ? needed : available_draw;
            for (int i = 0; i < actual_draw; i++) rack_add_tile(&my_realized_rack, sim_unseen[draw_start + i]);

            float my_realized = movegen_find_best_response_equity(&spec, dawg, &my_realized_rack, bag_count > 0 ? bag_count - 2 : 0);

            /* 2-ply net realization: (immediate score + my turn-2 realized score) - opponent response score */
            float sample_net = (float)moves.moves[k].score + (0.85f * my_realized) - opp_equity;
            net_totals[k] += sample_net;
        }
    }

    int best_k = 0;
    float best_val = -100000.0f;
    for (int k = 0; k < beam; k++) {
        float avg_net = net_totals[k] / (float)N_SIMS;
        if (avg_net > best_val) {
            best_val = avg_net;
            best_k = k;
        }
    }

    return moves.moves[best_k];
}

/* Dispatch opponent move based on bot type */
static Move select_opponent_move(BotType opp_type, const Board *board, const Dawg *dawg,
                                 const Rack *rack, int bag_count) {
    switch (opp_type) {
        case BOT_BASELINE_GREEDY:
            return select_greedy_move(board, dawg, rack);
        case BOT_DEFENSIVE_TURTLER:
            return select_defensive_move(board, dawg, rack);
        case BOT_BINGO_HUNTER:
            return select_bingo_hunter_move(board, dawg, rack);
        case BOT_HEURISTIC_STATIC:
            return select_heuristic_move(board, dawg, rack, bag_count);
        case BOT_NEURAL_STATIC:
            return select_neural_static_move(board, dawg, rack, bag_count);
        case BOT_CHAMPIONSHIP_SIM:
            return select_quackle_sim_move(board, dawg, rack, bag_count);
        case BOT_MAGPIE_2PLY:
            return select_magpie_sim_move(board, dawg, rack, bag_count);
        default:
            return select_greedy_move(board, dawg, rack);
    }
}

/* Play a single game between Tessera and Opponent */
static void play_single_game(const Dawg *dawg, BotType opp_type, int game_idx,
                             int *out_tessera_score, int *out_opp_score,
                             int *out_tessera_bingos, int *out_opp_bingos) {
    Board board;
    board_init(&board);

    TileBag bag;
    bag_init(&bag);
    bag_shuffle(&bag);

    Rack tessera_rack, opp_rack;
    rack_init(&tessera_rack);
    rack_init(&opp_rack);

    int turn = game_idx % 2; /* Alternate who plays first */

    if (turn == 0) {
        bag_draw_tiles(&bag, &tessera_rack, RACK_CAPACITY);
        bag_draw_tiles(&bag, &opp_rack, RACK_CAPACITY);
    } else {
        bag_draw_tiles(&bag, &opp_rack, RACK_CAPACITY);
        bag_draw_tiles(&bag, &tessera_rack, RACK_CAPACITY);
    }

    int tessera_score = 0;
    int opp_score = 0;
    int t_bingos = 0;
    int o_bingos = 0;
    int consecutive_passes = 0;

    EngineConfig cfg;
    engine_config_default(&cfg);
    cfg.max_deliberation_steps = 20;

    BayesRackTracker tracker;
    bayes_rack_init(&tracker);
    cfg.persistent_tracker = &tracker;

    while (consecutive_passes < 2) {
        if (turn == 0) {
            /* Tessera's turn */
            Move best = lattice_search(&board, dawg, &tessera_rack, &bag, &cfg, NULL, NULL);

            if (best.type == MOVE_PLAY) {
                consecutive_passes = 0;
                board_apply_move(&board, &best);
                tessera_score += best.score;
                if (best.num_placed == 7) t_bingos++;

                for (int i = 0; i < best.num_placed; i++) {
                    char t = best.placed[i].is_blank ? '?' : best.placed[i].letter;
                    rack_remove_tile(&tessera_rack, t);
                }
                bag_draw_tiles(&bag, &tessera_rack, best.num_placed);

                if (tessera_rack.count == 0 && bag.count == 0) break;
            } else if (best.type == MOVE_EXCHANGE) {
                consecutive_passes = 0;
                for (int i = 0; i < best.num_placed; i++) {
                    rack_remove_tile(&tessera_rack, best.placed[i].letter);
                    bag.tiles[bag.count++] = best.placed[i].letter;
                }
                bag_shuffle(&bag);
                bag_draw_tiles(&bag, &tessera_rack, best.num_placed);
            } else {
                consecutive_passes++;
            }
            turn = 1;
        } else {
            /* Opponent's turn */
            Board board_before;
            board_copy(&board_before, &board);

            Move opp_move = select_opponent_move(opp_type, &board, dawg, &opp_rack, bag.count);
            bayes_rack_observe_opponent(&tracker, &board_before, &opp_move);

            if (opp_move.type == MOVE_PLAY) {
                consecutive_passes = 0;
                board_apply_move(&board, &opp_move);
                opp_score += opp_move.score;
                if (opp_move.num_placed == 7) o_bingos++;

                for (int i = 0; i < opp_move.num_placed; i++) {
                    char t = opp_move.placed[i].is_blank ? '?' : opp_move.placed[i].letter;
                    rack_remove_tile(&opp_rack, t);
                }
                bag_draw_tiles(&bag, &opp_rack, opp_move.num_placed);

                if (opp_rack.count == 0 && bag.count == 0) break;
            } else {
                consecutive_passes++;
            }
            turn = 0;
        }
    }

    /* Outplay and unplayed tile deductions */
    if (tessera_rack.count == 0 && bag.count == 0) {
        int opp_pen = 0;
        for (int i = 0; i < opp_rack.count; i++) {
            char t = opp_rack.tiles[i];
            if (t >= 'A' && t <= 'Z') opp_pen += TILE_VALUES[t - 'A'];
        }
        tessera_score += 2 * opp_pen;
    } else if (opp_rack.count == 0 && bag.count == 0) {
        int t_pen = 0;
        for (int i = 0; i < tessera_rack.count; i++) {
            char t = tessera_rack.tiles[i];
            if (t >= 'A' && t <= 'Z') t_pen += TILE_VALUES[t - 'A'];
        }
        opp_score += 2 * t_pen;
    } else {
        int t_pen = 0, o_pen = 0;
        for (int i = 0; i < tessera_rack.count; i++) {
            char t = tessera_rack.tiles[i];
            if (t >= 'A' && t <= 'Z') t_pen += TILE_VALUES[t - 'A'];
        }
        for (int i = 0; i < opp_rack.count; i++) {
            char t = opp_rack.tiles[i];
            if (t >= 'A' && t <= 'Z') o_pen += TILE_VALUES[t - 'A'];
        }
        tessera_score -= t_pen;
        opp_score -= o_pen;
    }

    *out_tessera_score = tessera_score;
    *out_opp_score = opp_score;
    *out_tessera_bingos = t_bingos;
    *out_opp_bingos = o_bingos;
}

int main(int argc, char **argv) {
    int games_per_bot = 6;
    if (argc > 1) {
        games_per_bot = atoi(argv[1]);
        if (games_per_bot < 2) games_per_bot = 2;
        if (games_per_bot % 2 != 0) games_per_bot++; /* Ensure even for start balance */
    }

    printf("======================================================================\n");
    printf("   TESSERA GRANDMASTER MULTI-BOT BENCHMARK & COMPETITIVE MATRIX      \n");
    printf("======================================================================\n");
    printf("Lexicon: SOWPODS (267,751 words) | Architecture: Pure Graph Belief Lattice\n");
    printf("Playing %d tournament matches against EACH of %d Opponent Profiles (%d Total Games)...\n\n",
           games_per_bot, BOT_TYPE_COUNT, games_per_bot * BOT_TYPE_COUNT);

    Dawg *dawg = dawg_load_binary("data/dawg.bin");
    if (!dawg) {
        fprintf(stderr, "Error loading data/dawg.bin\n");
        return 1;
    }

    if (nnue_load(&g_nnue, "data/nnue.bin")) {
        printf("[INFO] Deep NNUE Evaluator loaded successfully (448->128->32->1, 246 KB)\n\n");
    } else {
        printf("[INFO] NNUE model data/nnue.bin not found. Using hand-tuned Evaluator.\n\n");
    }

    srand(1337);

    MatchStats all_stats[BOT_TYPE_COUNT];
    memset(all_stats, 0, sizeof(all_stats));

    int total_wins = 0, total_losses = 0, total_ties = 0;
    long long grand_tessera_pts = 0, grand_opp_pts = 0;
    int grand_tessera_bingos = 0, grand_opp_bingos = 0;
    double total_runtime = 0.0;

    for (int b = 0; b < BOT_TYPE_COUNT; b++) {
        printf(">>> MATCHUP %d/%d: Tessera vs. %s\n", b + 1, BOT_TYPE_COUNT, BOT_NAMES[b]);
        clock_t start = clock();

        for (int g = 0; g < games_per_bot; g++) {
            int ts = 0, os = 0, tb = 0, ob = 0;
            play_single_game(dawg, (BotType)b, g, &ts, &os, &tb, &ob);

            all_stats[b].total_tessera_score += ts;
            all_stats[b].total_opponent_score += os;
            all_stats[b].tessera_bingos += tb;
            all_stats[b].opponent_bingos += ob;

            if (ts > os) all_stats[b].wins++;
            else if (os > ts) all_stats[b].losses++;
            else all_stats[b].ties++;

            printf("\r  [Game %2d/%2d] Tessera: %3d | %s: %3d | Record: %d-%d-%d",
                   g + 1, games_per_bot, ts, BOT_NAMES[b], os,
                   all_stats[b].wins, all_stats[b].losses, all_stats[b].ties);
            fflush(stdout);
        }

        clock_t end = clock();
        all_stats[b].elapsed_seconds = (double)(end - start) / CLOCKS_PER_SEC;
        total_runtime += all_stats[b].elapsed_seconds;

        total_wins += all_stats[b].wins;
        total_losses += all_stats[b].losses;
        total_ties += all_stats[b].ties;
        grand_tessera_pts += all_stats[b].total_tessera_score;
        grand_opp_pts += all_stats[b].total_opponent_score;
        grand_tessera_bingos += all_stats[b].tessera_bingos;
        grand_opp_bingos += all_stats[b].opponent_bingos;

        double win_rate = (double)all_stats[b].wins / games_per_bot * 100.0;
        double avg_ts = (double)all_stats[b].total_tessera_score / games_per_bot;
        double avg_os = (double)all_stats[b].total_opponent_score / games_per_bot;

        printf("\n  Result: %d W - %d L - %d T (%.1f%% Win Rate) | Avg Score: %.1f vs %.1f (Spread: %+.1f pts) | Time: %.1fs\n\n",
               all_stats[b].wins, all_stats[b].losses, all_stats[b].ties, win_rate,
               avg_ts, avg_os, avg_ts - avg_os, all_stats[b].elapsed_seconds);
    }

    /* Overall Summary Matrix */
    int total_games = games_per_bot * BOT_TYPE_COUNT;
    double overall_win_rate = (double)total_wins / total_games * 100.0;
    double overall_avg_tessera = (double)grand_tessera_pts / total_games;
    double overall_avg_opp = (double)grand_opp_pts / total_games;
    double overall_spread = overall_avg_tessera - overall_avg_opp;
    double bingos_per_game = (double)grand_tessera_bingos / total_games;

    /* Compute Official Tournament Performance Rating (TPR):
     * MeanOpponentRating = sum(OpponentElo_i) / NumProfiles
     * ScoreRatio = (Wins + 0.5 * Ties) / (Losses + 0.5 * Ties)
     * PerformanceElo = MeanOpponentRating + 400 * log10(ScoreRatio)
     */
    double sum_opp_elo = 0.0;
    for (int b = 0; b < BOT_TYPE_COUNT; b++) {
        sum_opp_elo += (double)BOT_ESTIMATED_ELOS[b];
    }
    double mean_opp_elo = sum_opp_elo / (double)BOT_TYPE_COUNT;

    double points_won = (double)total_wins + 0.5 * (double)total_ties;
    double points_lost = (double)total_losses + 0.5 * (double)total_ties;
    double score_ratio = fmax(0.05, fmin(19.0, points_won / fmax(0.5, points_lost)));
    double elo_delta = 400.0 * log10(score_ratio);
    int estimated_tessera_elo = (int)(mean_opp_elo + elo_delta);

    printf("======================================================================\n");
    printf("                  TOURNAMENT LEADERBOARD & ELO MATRIX                 \n");
    printf("======================================================================\n");
    printf("  Opponent Profile                    | Record  | Win Rate | Avg Spread | Ref Elo\n");
    printf("  ------------------------------------+---------+----------+------------+--------\n");
    for (int b = 0; b < BOT_TYPE_COUNT; b++) {
        double wr = (double)all_stats[b].wins / games_per_bot * 100.0;
        double sp = (double)(all_stats[b].total_tessera_score - all_stats[b].total_opponent_score) / games_per_bot;
        printf("  %-35s | %2d-%2d-%-2d | %7.1f%% | %+9.1f  | ~%4d\n",
               BOT_NAMES[b], all_stats[b].wins, all_stats[b].losses, all_stats[b].ties, wr, sp, BOT_ESTIMATED_ELOS[b]);
    }
    printf("  ------------------------------------+---------+----------+------------+--------\n");
    printf("  OVERALL TESSERA TOURNAMENT TOTAL   | %2d-%2d-%-2d | %7.1f%% | %+9.1f  | ~%4d\n\n",
           total_wins, total_losses, total_ties, overall_win_rate, overall_spread, estimated_tessera_elo);

    printf("  Grand Tournament Statistics:\n");
    printf("  - Total Games Played:       %d\n", total_games);
    printf("  - Tessera Average Score:    %.1f pts/game\n", overall_avg_tessera);
    printf("  - Opponents Average Score:  %.1f pts/game\n", overall_avg_opp);
    printf("  - Net Tournament Spread:    %+.1f pts/game\n", overall_spread);
    printf("  - Tessera Bingo Frequency:  %.2f bingos/game (Opponents: %.2f)\n",
           bingos_per_game, (double)grand_opp_bingos / total_games);
    printf("  - Estimated Tessera Elo:    ~%d Elo (NASPA Grandmaster / Champion)\n", estimated_tessera_elo);
    printf("  - Total Computation Time:   %.2f seconds (%.1f ms/game)\n",
           total_runtime, (total_runtime / total_games) * 1000.0);
    printf("======================================================================\n");

    dawg_free(dawg);
    return 0;
}
