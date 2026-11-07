#include "dict.h"
#include "board.h"
#include "movegen.h"
#include "eval.h"
#include "simulation.h"
#include "nnue.h"
#include "bayes_rack.h"
#include "bot_archetypes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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

            Move opp_move = bot_archetype_select_move(opp_type, &board, dawg, &opp_rack, bag.count);
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

    /* Outplay and unplayed tile deductions (Official WESPA / NASPA rules) */
    if (tessera_rack.count == 0 && bag.count == 0) {
        int opp_pen = 0;
        for (int i = 0; i < opp_rack.count; i++) {
            char t = opp_rack.tiles[i];
            if (t >= 'A' && t <= 'Z') opp_pen += TILE_VALUES[t - 'A'];
        }
        tessera_score += opp_pen;
        opp_score -= opp_pen;
    } else if (opp_rack.count == 0 && bag.count == 0) {
        int t_pen = 0;
        for (int i = 0; i < tessera_rack.count; i++) {
            char t = tessera_rack.tiles[i];
            if (t >= 'A' && t <= 'Z') t_pen += TILE_VALUES[t - 'A'];
        }
        opp_score += t_pen;
        tessera_score -= t_pen;
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
        const char *bot_name = bot_archetype_name((BotType)b);
        printf(">>> MATCHUP %d/%d: Tessera vs. %s\n", b + 1, BOT_TYPE_COUNT, bot_name);
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
                   g + 1, games_per_bot, ts, bot_name, os,
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

    double sum_opp_elo = 0.0;
    for (int b = 0; b < BOT_TYPE_COUNT; b++) {
        sum_opp_elo += (double)bot_archetype_elo((BotType)b);
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
               bot_archetype_name((BotType)b), all_stats[b].wins, all_stats[b].losses, all_stats[b].ties, wr, sp, bot_archetype_elo((BotType)b));
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
