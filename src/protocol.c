#include "protocol.h"
#include "bot_archetypes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif
#include <ctype.h>
#include <time.h>

static void print_state_json(const GameState *game) {
    printf("{\"status\":\"ok\",\"bag_count\":%d,\"player_score\":%d,\"ai_score\":%d,\"turn\":%d,\"game_over\":%s,",
           game->bag.count, game->player_score, game->ai_score, game->turn,
           game->game_over ? "true" : "false");

    char p_rack[RACK_CAPACITY + 1], a_rack[RACK_CAPACITY + 1];
    rack_to_string(&game->player_rack, p_rack);
    rack_to_string(&game->ai_rack, a_rack);
    printf("\"player_rack\":\"%s\",\"ai_rack\":\"%s\",", p_rack, a_rack);

    printf("\"board\":[");
    for (int r = 0; r < BOARD_SIZE; r++) {
        printf("[");
        for (int c = 0; c < BOARD_SIZE; c++) {
            char t = game->board[r][c].tile;
            bool is_b = game->board[r][c].is_blank;
            MultiplierType m = game->board[r][c].mult;
            printf("{\"tile\":\"%s\",\"is_blank\":%s,\"mult\":%d}%s",
                   t != 0 ? (char[]){t, '\0'} : "",
                   is_b ? "true" : "false",
                   (int)m,
                   c < BOARD_SIZE - 1 ? "," : "");
        }
        printf("]%s", r < BOARD_SIZE - 1 ? "," : "");
    }
    printf("]}\n");
    fflush(stdout);
}

void protocol_run_server_loop(const Dawg *dawg) {
    GameState game;
    memset(&game, 0, sizeof(GameState));

    Board current_board;
    board_init(&current_board);

    bag_init(&game.bag);
    bag_shuffle(&game.bag);

    rack_init(&game.player_rack);
    rack_init(&game.ai_rack);
    bag_draw_tiles(&game.bag, &game.player_rack, RACK_CAPACITY);
    bag_draw_tiles(&game.bag, &game.ai_rack, RACK_CAPACITY);

    memcpy(game.board, current_board.cells, sizeof(game.board));
    game.turn = 0; /* Player starts */

    char line[1024];
    while (fgets(line, sizeof(line), stdin)) {
        /* Strip newline */
        int len = (int)strlen(line);
        while (len > 0 && (line[len - 1] == '\r' || line[len - 1] == '\n')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;

        char cmd[64];
        if (sscanf(line, "%63s", cmd) != 1) continue;

        if (strcmp(cmd, "NEW_GAME") == 0) {
            board_init(&current_board);
            memcpy(game.board, current_board.cells, sizeof(game.board));
            bag_init(&game.bag);
            bag_shuffle(&game.bag);
            rack_init(&game.player_rack);
            rack_init(&game.ai_rack);
            bag_draw_tiles(&game.bag, &game.player_rack, RACK_CAPACITY);
            bag_draw_tiles(&game.bag, &game.ai_rack, RACK_CAPACITY);
            game.player_score = 0;
            game.ai_score = 0;
            game.turn = 0;
            game.pass_count = 0;
            game.game_over = false;
            print_state_json(&game);
        } else if (strcmp(cmd, "GET_STATE") == 0) {
            print_state_json(&game);
        } else if (strcmp(cmd, "CHECK_WORD") == 0) {
            char word[64];
            if (sscanf(line, "%*s %63s", word) == 1) {
                bool valid = dawg_contains(dawg, word);
                printf("{\"status\":\"ok\",\"word\":\"%s\",\"valid\":%s}\n",
                       word, valid ? "true" : "false");
                fflush(stdout);
            }
        } else if (strcmp(cmd, "HINT") == 0) {
            MoveList moves;
            movegen_generate_all(&current_board, dawg, &game.player_rack, &moves);
            for (int i = 0; i < moves.count; i++) {
                moves.moves[i].leave_equity = eval_move_static_bucketed(&current_board, &moves.moves[i], game.bag.count);
            }
            eval_sort_moves(&moves);

            PhaseBucket pb = eval_classify_phase(game.bag.count);
            OpennessBucket ob = eval_classify_openness(&current_board);

            int count = moves.count < 5 ? moves.count : 5;
            printf("{\"status\":\"ok\",\"phase\":\"%s\",\"openness\":\"%s\",\"hints\":[",
                   eval_phase_name(pb), eval_openness_name(ob));
            for (int i = 0; i < count; i++) {
                Move *m = &moves.moves[i];
                printf("{\"word\":\"%s\",\"row\":%d,\"col\":%d,\"dir\":\"%s\",\"score\":%d,\"leave\":\"%s\",\"equity\":%.2f}%s",
                       m->word, m->row, m->col, m->dir == DIR_HORIZONTAL ? "H" : "V",
                       m->score, m->leave, m->leave_equity,
                       i < count - 1 ? "," : "");
            }
            printf("]}\n");
            fflush(stdout);
        } else if (strcmp(cmd, "OPPONENT_MOVE") == 0) {
            int bot_type_int = 6; /* Default to MAGPIE 2-Ply Realization */
            sscanf(line, "%*s %d", &bot_type_int);
            if (bot_type_int < 0 || bot_type_int >= BOT_TYPE_COUNT) {
                bot_type_int = 6;
            }
            BotType bot_type = (BotType)bot_type_int;

            if (game.game_over) {
                printf("{\"status\":\"error\",\"message\":\"Game is over\"}\n");
                fflush(stdout);
                continue;
            }

            clock_t start_clk = clock();
            Move best = bot_archetype_select_move(bot_type, &current_board, dawg, &game.player_rack, game.bag.count);
            double elapsed_ms = ((double)(clock() - start_clk) / CLOCKS_PER_SEC) * 1000.0;

            if (best.type == MOVE_PLAY) {
                board_apply_move(&current_board, &best);
                memcpy(game.board, current_board.cells, sizeof(game.board));
                game.player_score += best.score;

                for (int i = 0; i < best.num_placed; i++) {
                    char t = best.placed[i].is_blank ? '?' : best.placed[i].letter;
                    rack_remove_tile(&game.player_rack, t);
                }
                bag_draw_tiles(&game.bag, &game.player_rack, best.num_placed);
                game.pass_count = 0;
            } else if (best.type == MOVE_EXCHANGE) {
                for (int i = 0; i < best.num_placed; i++) {
                    rack_remove_tile(&game.player_rack, best.placed[i].letter);
                    game.bag.tiles[game.bag.count++] = best.placed[i].letter;
                }
                bag_shuffle(&game.bag);
                bag_draw_tiles(&game.bag, &game.player_rack, best.num_placed);
                game.pass_count = 0;
            } else {
                game.pass_count++;
            }

            if (game.pass_count >= 2 || (game.player_rack.count == 0 && game.bag.count == 0)) {
                game.game_over = true;
            }
            game.turn = 1; /* Now Tessera's turn */

            char p_rack[RACK_CAPACITY + 1], a_rack[RACK_CAPACITY + 1];
            rack_to_string(&game.player_rack, p_rack);
            rack_to_string(&game.ai_rack, a_rack);

            printf("{\"status\":\"ok\",\"time_ms\":%.2f,\"bot_type\":%d,\"bot_name\":\"%s\",\"bot_elo\":%d,\"move\":{\"type\":%d,\"word\":\"%s\",\"row\":%d,\"col\":%d,\"dir\":\"%s\",\"score\":%d,\"leave\":\"%s\",\"equity\":%.2f},\"player_score\":%d,\"ai_score\":%d,\"bag_count\":%d,\"game_over\":%s,\"player_rack\":\"%s\",\"ai_rack\":\"%s\"}\n",
                   elapsed_ms, (int)bot_type, bot_archetype_name(bot_type), bot_archetype_elo(bot_type),
                   best.type, best.word, best.row, best.col,
                   best.dir == DIR_HORIZONTAL ? "H" : "V",
                   best.score, best.leave, best.leave_equity,
                   game.player_score, game.ai_score, game.bag.count,
                   game.game_over ? "true" : "false", p_rack, a_rack);
            fflush(stdout);
        } else if (strcmp(cmd, "AI_MOVE") == 0) {
            if (game.game_over) {
                printf("{\"status\":\"error\",\"message\":\"Game is over\"}\n");
                fflush(stdout);
                continue;
            }

            PhaseBucket pb = eval_classify_phase(game.bag.count);
            OpennessBucket ob = eval_classify_openness(&current_board);
            bool is_endgame = (game.bag.count == 0);

            EngineConfig cfg;
            engine_config_default(&cfg);
            EngineMetrics metrics;

            MoveList moves;
            Move best = lattice_search(&current_board, dawg, &game.ai_rack, &game.bag, &cfg, &moves, &metrics);

            if (best.type == MOVE_PLAY) {
                board_apply_move(&current_board, &best);
                memcpy(game.board, current_board.cells, sizeof(game.board));
                game.ai_score += best.score;

                /* Remove placed tiles from AI rack */
                for (int i = 0; i < best.num_placed; i++) {
                    char t = best.placed[i].is_blank ? '?' : best.placed[i].letter;
                    rack_remove_tile(&game.ai_rack, t);
                }
                bag_draw_tiles(&game.bag, &game.ai_rack, best.num_placed);
                game.pass_count = 0;
            } else if (best.type == MOVE_EXCHANGE) {
                for (int i = 0; i < best.num_placed; i++) {
                    rack_remove_tile(&game.ai_rack, best.placed[i].letter);
                    game.bag.tiles[game.bag.count++] = best.placed[i].letter;
                }
                bag_shuffle(&game.bag);
                bag_draw_tiles(&game.bag, &game.ai_rack, best.num_placed);
                game.pass_count = 0;
            } else {
                game.pass_count++;
            }

            if (game.pass_count >= 2 || (game.ai_rack.count == 0 && game.bag.count == 0)) {
                game.game_over = true;
            }
            game.turn = 0;

            printf("{\"status\":\"ok\",\"time_ms\":%.2f,\"phase\":\"%s\",\"openness\":\"%s\",\"endgame\":%s,\"move\":{\"type\":%d,\"word\":\"%s\",\"row\":%d,\"col\":%d,\"dir\":\"%s\",\"score\":%d,\"leave\":\"%s\",\"equity\":%.2f},",
                   metrics.deliberation_ms, eval_phase_name(pb), eval_openness_name(ob),
                   is_endgame ? "true" : "false",
                   best.type, best.word, best.row, best.col,
                   best.dir == DIR_HORIZONTAL ? "H" : "V",
                   best.score, best.leave, best.sim_equity);

            char p_rack[RACK_CAPACITY + 1], a_rack[RACK_CAPACITY + 1];
            rack_to_string(&game.player_rack, p_rack);
            rack_to_string(&game.ai_rack, a_rack);
            printf("\"metrics\":{\"nodes\":%u,\"transpositions\":%u,\"settle_passes\":%u,\"confidence\":%.2f,\"palimpsest_bias\":%.2f},",
                   metrics.nodes_in_lattice, metrics.transpositions_merged, metrics.dirty_passes_settled,
                   metrics.court_confidence, metrics.palimpsest_max_bias);
            printf("\"player_score\":%d,\"ai_score\":%d,\"bag_count\":%d,\"game_over\":%s,\"player_rack\":\"%s\",\"ai_rack\":\"%s\"}\n",
                   game.player_score, game.ai_score, game.bag.count,
                   game.game_over ? "true" : "false", p_rack, a_rack);
            fflush(stdout);
        } else if (strcmp(cmd, "PLAY_MOVE") == 0) {
            /* Format: PLAY_MOVE row col dir word blank_mask */
            int r, c;
            char dir_str[4], word[64], blanks[64] = "";
            if (sscanf(line, "%*s %d %d %3s %63s %63s", &r, &c, dir_str, word, blanks) >= 4) {
                Direction dir = (toupper((unsigned char)dir_str[0]) == 'V') ? DIR_VERTICAL : DIR_HORIZONTAL;

                /* Validate with move generator */
                MoveList moves;
                movegen_generate_all(&current_board, dawg, &game.player_rack, &moves);

                int match_idx = -1;
                for (int i = 0; i < moves.count; i++) {
                    if (moves.moves[i].row == r && moves.moves[i].col == c &&
                        moves.moves[i].dir == dir &&
                        strcasecmp(moves.moves[i].word, word) == 0) {
                        match_idx = i;
                        break;
                    }
                }

                if (match_idx < 0) {
                    printf("{\"status\":\"error\",\"message\":\"Illegal move. Word not in dictionary or placement invalid.\"}\n");
                    fflush(stdout);
                    continue;
                }

                Move *m = &moves.moves[match_idx];
                board_apply_move(&current_board, m);
                memcpy(game.board, current_board.cells, sizeof(game.board));
                game.player_score += m->score;

                for (int i = 0; i < m->num_placed; i++) {
                    char t = m->placed[i].is_blank ? '?' : m->placed[i].letter;
                    rack_remove_tile(&game.player_rack, t);
                }
                bag_draw_tiles(&game.bag, &game.player_rack, m->num_placed);

                game.pass_count = 0;
                game.turn = 1;
                if (game.player_rack.count == 0 && game.bag.count == 0) {
                    game.game_over = true;
                }

                printf("{\"status\":\"ok\",\"score\":%d,\"player_score\":%d,\"ai_score\":%d,\"bag_count\":%d,\"game_over\":%s,",
                       m->score, game.player_score, game.ai_score, game.bag.count,
                       game.game_over ? "true" : "false");
                char p_rack[RACK_CAPACITY + 1];
                rack_to_string(&game.player_rack, p_rack);
                printf("\"player_rack\":\"%s\"}\n", p_rack);
                fflush(stdout);
            } else {
                printf("{\"status\":\"error\",\"message\":\"Malformed PLAY_MOVE command\"}\n");
                fflush(stdout);
            }
        } else if (strcmp(cmd, "PASS") == 0) {
            game.pass_count++;
            game.turn = 1;
            if (game.pass_count >= 2) game.game_over = true;
            printf("{\"status\":\"ok\",\"pass_count\":%d,\"game_over\":%s}\n",
                   game.pass_count, game.game_over ? "true" : "false");
            fflush(stdout);
        } else if (strcmp(cmd, "EXCHANGE") == 0) {
            char exchange_tiles[64];
            if (sscanf(line, "%*s %63s", exchange_tiles) == 1) {
                int ex_len = (int)strlen(exchange_tiles);
                if (game.bag.count < RACK_CAPACITY) {
                    printf("{\"status\":\"error\",\"message\":\"Cannot exchange when bag has fewer than 7 tiles\"}\n");
                    fflush(stdout);
                    continue;
                }
                for (int i = 0; i < ex_len; i++) {
                    char t = exchange_tiles[i];
                    if (rack_remove_tile(&game.player_rack, t)) {
                        game.bag.tiles[game.bag.count++] = t;
                    }
                }
                bag_shuffle(&game.bag);
                bag_draw_tiles(&game.bag, &game.player_rack, ex_len);
                game.pass_count = 0;
                game.turn = 1;
                char p_rack[RACK_CAPACITY + 1];
                rack_to_string(&game.player_rack, p_rack);
                printf("{\"status\":\"ok\",\"player_rack\":\"%s\"}\n", p_rack);
                fflush(stdout);
            }
        } else if (strcmp(cmd, "QUIT") == 0) {
            printf("{\"status\":\"ok\",\"message\":\"Goodbye\"}\n");
            fflush(stdout);
            break;
        } else {
            printf("{\"status\":\"error\",\"message\":\"Unknown command\"}\n");
            fflush(stdout);
        }
    }
}

void protocol_run_terminal_game(const Dawg *dawg, bool ai_vs_ai) {
    printf("============================================================\n");
    printf("       TESSERA — GRANDMASTER SCRABBLE ENGINE (NATIVE C)     \n");
    printf("============================================================\n\n");

    Board board;
    board_init(&board);

    TileBag bag;
    bag_init(&bag);
    bag_shuffle(&bag);

    Rack p1_rack, p2_rack;
    rack_init(&p1_rack);
    rack_init(&p2_rack);
    bag_draw_tiles(&bag, &p1_rack, RACK_CAPACITY);
    bag_draw_tiles(&bag, &p2_rack, RACK_CAPACITY);

    int p1_score = 0, p2_score = 0;
    int turn = 0;
    int pass_count = 0;

    EngineConfig cfg;
    engine_config_default(&cfg);

    while (pass_count < 2) {
        board_print(&board);
        printf("\nScore: P1 (AI-1): %d | P2 (%s): %d | Bag: %d tiles\n",
               p1_score, ai_vs_ai ? "AI-2" : "Human", p2_score, bag.count);

        Rack *cur_rack = (turn == 0) ? &p1_rack : &p2_rack;
        char rack_str[RACK_CAPACITY + 1];
        rack_to_string(cur_rack, rack_str);
        printf("Turn: %s | Rack: [%s]\n",
               turn == 0 ? "AI-1" : (ai_vs_ai ? "AI-2" : "Human"), rack_str);

        if (turn == 0 || ai_vs_ai) {
            printf("AI is thinking...\n");
            MoveList moves;
            clock_t start = clock();
            Move best = sim_find_best_move(&board, dawg, cur_rack, &bag, &cfg, &moves);
            clock_t end = clock();
            double ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

            if (best.type == MOVE_PLAY) {
                printf("AI plays: %s at (%d, %d) dir=%s for %d points! (%.1f ms)\n",
                       best.word, best.row + 1, best.col + 1,
                       best.dir == DIR_HORIZONTAL ? "Across" : "Down",
                       best.score, ms);
                board_apply_move(&board, &best);
                if (turn == 0) p1_score += best.score;
                else p2_score += best.score;

                for (int i = 0; i < best.num_placed; i++) {
                    char t = best.placed[i].is_blank ? '?' : best.placed[i].letter;
                    rack_remove_tile(cur_rack, t);
                }
                bag_draw_tiles(&bag, cur_rack, best.num_placed);
                pass_count = 0;
            } else {
                printf("AI passes or exchanges!\n");
                pass_count++;
            }
        } else {
            /* Human input */
            printf("Enter move (e.g. 'PLAY 8 8 H CAT' or 'PASS' or 'HINT'): ");
            char user_line[128];
            if (!fgets(user_line, sizeof(user_line), stdin)) break;

            if (strncmp(user_line, "HINT", 4) == 0) {
                MoveList moves;
                movegen_generate_all(&board, dawg, cur_rack, &moves);
                for (int i = 0; i < moves.count; i++) {
                    moves.moves[i].leave_equity = eval_move_static(&board, &moves.moves[i]);
                }
                eval_sort_moves(&moves);
                printf("\nTop 3 Hints:\n");
                for (int i = 0; i < (moves.count < 3 ? moves.count : 3); i++) {
                    printf("  %d) %s at (%d, %d) %s -> %d pts (leave: %s)\n",
                           i + 1, moves.moves[i].word, moves.moves[i].row + 1,
                           moves.moves[i].col + 1,
                           moves.moves[i].dir == DIR_HORIZONTAL ? "H" : "V",
                           moves.moves[i].score, moves.moves[i].leave);
                }
                continue;
            } else if (strncmp(user_line, "PASS", 4) == 0) {
                pass_count++;
            } else {
                int r, c;
                char d_str[4], w[64];
                if (sscanf(user_line, "PLAY %d %d %3s %63s", &r, &c, d_str, w) >= 4) {
                    MoveList moves;
                    movegen_generate_all(&board, dawg, cur_rack, &moves);
                    Direction d = (toupper((unsigned char)d_str[0]) == 'V') ? DIR_VERTICAL : DIR_HORIZONTAL;
                    int match = -1;
                    for (int i = 0; i < moves.count; i++) {
                        if (moves.moves[i].row == r - 1 && moves.moves[i].col == c - 1 &&
                            moves.moves[i].dir == d && strcasecmp(moves.moves[i].word, w) == 0) {
                            match = i;
                            break;
                        }
                    }
                    if (match >= 0) {
                        Move *m = &moves.moves[match];
                        board_apply_move(&board, m);
                        p2_score += m->score;
                        for (int i = 0; i < m->num_placed; i++) {
                            char t = m->placed[i].is_blank ? '?' : m->placed[i].letter;
                            rack_remove_tile(cur_rack, t);
                        }
                        bag_draw_tiles(&bag, cur_rack, m->num_placed);
                        pass_count = 0;
                    } else {
                        printf("Invalid move! Try again.\n");
                        continue;
                    }
                } else {
                    printf("Could not parse command.\n");
                    continue;
                }
            }
        }

        if (cur_rack->count == 0 && bag.count == 0) {
            printf("\nPlayer played out all tiles! Game over!\n");
            break;
        }

        turn = 1 - turn;
    }

    printf("\n================ GAME OVER ================\n");
    printf("Final Score: AI-1: %d | Opponent: %d\n", p1_score, p2_score);
    if (p1_score > p2_score) printf("AI-1 Wins!\n");
    else if (p2_score > p1_score) printf("Opponent Wins!\n");
    else printf("It's a Tie!\n");
}

void protocol_run_benchmark(const Dawg *dawg) {
    printf("[BENCHMARK] Running Tessera Performance Suite...\n");

    Board board;
    board_init(&board);

    Rack racks[5];
    rack_init(&racks[0]);
    strcpy(racks[0].tiles, "RETAINS"); racks[0].count = 7;
    rack_init(&racks[1]);
    strcpy(racks[1].tiles, "QUIZZES"); racks[1].count = 7;
    rack_init(&racks[2]);
    strcpy(racks[2].tiles, "AEINST?"); racks[2].count = 7;
    rack_init(&racks[3]);
    strcpy(racks[3].tiles, "BCDFGHJ"); racks[3].count = 7;
    rack_init(&racks[4]);
    strcpy(racks[4].tiles, "AEIOU??"); racks[4].count = 7;

    TileBag bag;
    bag_init(&bag);

    EngineConfig cfg;
    engine_config_default(&cfg);

    int total_moves_gen = 0;
    clock_t start = clock();

    for (int rep = 0; rep < 5; rep++) {
        for (int i = 0; i < 5; i++) {
            MoveList ml;
            total_moves_gen += movegen_generate_all(&board, dawg, &racks[i], &ml);
        }
    }

    clock_t end = clock();
    double elapsed_s = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Move Generation: %d moves generated across 25 calls in %.3f seconds (%.0f moves/sec)\n",
           total_moves_gen, elapsed_s, (double)total_moves_gen / elapsed_s);

    printf("Lattice Belief-Settling Benchmark...\n");
    start = clock();
    MoveList ml;
    EngineConfig scfg;
    engine_config_default(&scfg);
    scfg.max_deliberation_steps = 50;
    EngineMetrics smetrics;
    Move m = lattice_search(&board, dawg, &racks[0], &bag, &scfg, &ml, &smetrics);
    end = clock();
    double sim_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("Decision resolved in %.2f ms!\n", sim_ms);
    printf("  Best Play: %s, Score: %d, Settled Q-value: %.2f\n", m.word, m.score, m.sim_equity);
    printf("  Lattice Nodes: %u, Transpositions Merged: %u\n", smetrics.nodes_in_lattice, smetrics.transpositions_merged);
    printf("  Dirty Passes Settled: %u, Frontier Leaves Bought: %u\n", smetrics.dirty_passes_settled, smetrics.frontier_leaves_bought);
    printf("  Root Court Confidence: %.1f%%, Palimpsest Max Bias: %.2f pts\n",
           smetrics.court_confidence * 100.0f, smetrics.palimpsest_max_bias);
    printf("[BENCHMARK] Complete.\n");
}
