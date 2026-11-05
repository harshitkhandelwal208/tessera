#include "simulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void sim_config_default(SimConfig *config) {
    config->num_candidates = 12;
    config->num_simulations = 10;
    config->use_endgame_solver = true;
}

int sim_get_unseen_tiles(const Board *board, const Rack *player_rack, char unseen[100]) {
    /* Count total tiles from standard distribution */
    int counts[27];
    for (int i = 0; i < 27; i++) {
        counts[i] = TILE_BAG_DISTRIBUTION[i];
    }

    /* Subtract tiles on board */
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            char t = board->cells[r][c].tile;
            if (t != 0) {
                if (board->cells[r][c].is_blank) {
                    counts[26]--;
                } else if (t >= 'A' && t <= 'Z') {
                    counts[t - 'A']--;
                }
            }
        }
    }

    /* Subtract player's rack */
    for (int i = 0; i < player_rack->count; i++) {
        char t = player_rack->tiles[i];
        if (t == '?') {
            counts[26]--;
        } else if (t >= 'A' && t <= 'Z') {
            counts[t - 'A']--;
        }
    }

    /* Collect unseen tiles */
    int unseen_count = 0;
    for (int i = 0; i < 26; i++) {
        for (int k = 0; k < counts[i]; k++) {
            unseen[unseen_count++] = (char)('A' + i);
        }
    }
    for (int k = 0; k < counts[26]; k++) {
        unseen[unseen_count++] = '?';
    }

    return unseen_count;
}

static void shuffle_tiles(char *tiles, int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        char tmp = tiles[i];
        tiles[i] = tiles[j];
        tiles[j] = tmp;
    }
}

Move sim_find_best_move(const Board *board, const Dawg *dawg, const Rack *rack,
                        const TileBag *bag, const SimConfig *config,
                        MoveList *all_moves) {
    /* Generate all legal moves */
    movegen_generate_all(board, dawg, rack, all_moves);

    /* If no moves possible, pass or exchange */
    if (all_moves->count == 0) {
        Move ex_move;
        if (eval_should_exchange(board, rack, bag ? bag->count : 0, 0.0f, &ex_move)) {
            return ex_move;
        }
        Move pass_move;
        memset(&pass_move, 0, sizeof(Move));
        pass_move.type = MOVE_PASS;
        return pass_move;
    }

    /* Compute static equity for all candidate moves */
    for (int i = 0; i < all_moves->count; i++) {
        all_moves->moves[i].leave_equity = eval_move_static(board, &all_moves->moves[i]);
        all_moves->moves[i].sim_equity = all_moves->moves[i].leave_equity;
    }

    /* Sort all moves by static equity descending */
    eval_sort_moves(all_moves);

    /* Check if exchanging is better than the best play */
    Move ex_move;
    if (eval_should_exchange(board, rack, bag ? bag->count : 0,
                            all_moves->moves[0].leave_equity, &ex_move)) {
        return ex_move;
    }

    int num_candidates = config ? config->num_candidates : 12;
    if (num_candidates > all_moves->count) {
        num_candidates = all_moves->count;
    }

    int num_sims = config ? config->num_simulations : 10;
    if (num_sims <= 0 || num_candidates <= 1) {
        return all_moves->moves[0];
    }

    /* Gather unseen tiles */
    char unseen[100];
    int unseen_count = sim_get_unseen_tiles(board, rack, unseen);

    /* --- ENDGAME SOLVER (Bag is empty) --- */
    if (bag && bag->count == 0 && config && config->use_endgame_solver) {
        /* Opponent's rack is known with 100% exactness */
        Rack opp_exact_rack;
        rack_init(&opp_exact_rack);
        int opp_rack_val = 0;
        for (int i = 0; i < unseen_count && i < RACK_CAPACITY; i++) {
            rack_add_tile(&opp_exact_rack, unseen[i]);
            if (unseen[i] != '?') {
                opp_rack_val += TILE_VALUES[unseen[i] - 'A'];
            }
        }

        int best_candidate_idx = 0;
        float best_endgame_net = -10000.0f;

        for (int c = 0; c < num_candidates; c++) {
            Move *cand = &all_moves->moves[c];

            /* If candidate plays out completely */
            if (cand->num_placed == rack->count) {
                /* Game ends! We get 2x opponent remaining tile value */
                float net = (float)cand->score + (2.0f * (float)opp_rack_val);
                cand->sim_equity = net;
                if (net > best_endgame_net) {
                    best_endgame_net = net;
                    best_candidate_idx = c;
                }
                continue;
            }

            /* Opponent response on speculative board */
            Board spec_board;
            board_copy(&spec_board, board);
            board_apply_move(&spec_board, cand);

            MoveList opp_moves;
            movegen_generate_all(&spec_board, dawg, &opp_exact_rack, &opp_moves);

            int max_opp_score = 0;
            for (int j = 0; j < opp_moves.count; j++) {
                if (opp_moves.moves[j].score > max_opp_score) {
                    max_opp_score = opp_moves.moves[j].score;
                }
            }

            float net = (float)cand->score - (float)max_opp_score;
            cand->sim_equity = net;
            if (net > best_endgame_net) {
                best_endgame_net = net;
                best_candidate_idx = c;
            }
        }

        return all_moves->moves[best_candidate_idx];
    }

    /* --- MIDGAME MONTE CARLO SIMULATION --- */
    if (unseen_count == 0) {
        return all_moves->moves[0];
    }

    float opp_response_sums[MAX_CANDIDATES] = {0.0f};

    for (int s = 0; s < num_sims; s++) {
        shuffle_tiles(unseen, unseen_count);

        Rack sim_opp_rack;
        rack_init(&sim_opp_rack);
        int draw_count = unseen_count < RACK_CAPACITY ? unseen_count : RACK_CAPACITY;
        for (int d = 0; d < draw_count; d++) {
            rack_add_tile(&sim_opp_rack, unseen[d]);
        }

        for (int c = 0; c < num_candidates; c++) {
            Move *cand = &all_moves->moves[c];

            Board spec_board;
            board_copy(&spec_board, board);
            board_apply_move(&spec_board, cand);

            MoveList opp_moves;
            movegen_generate_all(&spec_board, dawg, &sim_opp_rack, &opp_moves);

            float best_opp_equity = 0.0f;
            for (int j = 0; j < opp_moves.count; j++) {
                float opp_eq = (float)opp_moves.moves[j].score +
                               eval_rack_leave(opp_moves.moves[j].leave);
                if (opp_eq > best_opp_equity) {
                    best_opp_equity = opp_eq;
                }
            }

            opp_response_sums[c] += best_opp_equity;
        }
    }

    /* Compute final net equity and find champion move */
    int champion_idx = 0;
    float champion_equity = -100000.0f;

    for (int c = 0; c < num_candidates; c++) {
        float avg_opp = opp_response_sums[c] / (float)num_sims;
        /* Net Equity = Static Equity - 0.85 * Avg Opponent Best Response */
        all_moves->moves[c].sim_equity = all_moves->moves[c].leave_equity - (0.85f * avg_opp);

        if (all_moves->moves[c].sim_equity > champion_equity) {
            champion_equity = all_moves->moves[c].sim_equity;
            champion_idx = c;
        }
    }

    return all_moves->moves[champion_idx];
}
