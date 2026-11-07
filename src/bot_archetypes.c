#include "bot_archetypes.h"
#include "movegen.h"
#include "eval.h"
#include "simulation.h"
#include "nnue.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *BOT_NAMES[BOT_TYPE_COUNT] = {
    "Baseline Greedy (1-Ply)",
    "Defensive Turtler",
    "Bingo Hunter",
    "Heuristic Grandmaster",
    "Neural Static (448-dim NNUE)",
    "Quackle Simulation",
    "MAGPIE (2-Ply Realization)"
};

static const char *BOT_DESCRIPTIONS[BOT_TYPE_COUNT] = {
    "Selects highest immediate word score (Greedy 1-Ply).",
    "Minimizes premium hazard exposure and board openness.",
    "Maximizes 7-letter bingo leave stem synergy bonuses.",
    "Grandmaster 8-phase rack leave table with pairwise harmony.",
    "Direct forward pass through the 448-dim NNUE neural network.",
    "Paired Monte Carlo lookahead simulation (Quackle emulation).",
    "2-Ply paired realization against simulated opponent refills."
};

static const int BOT_ELOS[BOT_TYPE_COUNT] = {
    1550,
    1650,
    1750,
    1850,
    1950,
    2100,
    2150
};

const char *bot_archetype_name(BotType type) {
    if (type >= 0 && type < BOT_TYPE_COUNT) return BOT_NAMES[type];
    return "MAGPIE (2-Ply Realization)";
}

const char *bot_archetype_desc(BotType type) {
    if (type >= 0 && type < BOT_TYPE_COUNT) return BOT_DESCRIPTIONS[type];
    return "2-Ply Realization Simulation";
}

int bot_archetype_elo(BotType type) {
    if (type >= 0 && type < BOT_TYPE_COUNT) return BOT_ELOS[type];
    return 2150;
}

static void bot_shuffle_tiles(char *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        char tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* 1. Greedy Bot */
static Move select_greedy_move(const Board *board, const Dawg *dawg, const Rack *rack) {
    MoveList moves = {0};
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        move_list_destroy(&moves);
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
    Move result = moves.moves[best_idx];
    move_list_destroy(&moves);
    return result;
}

/* 2. Defensive Turtler Bot */
static Move select_defensive_move(const Board *board, const Dawg *dawg, const Rack *rack) {
    MoveList moves = {0};
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        move_list_destroy(&moves);
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
    Move result = moves.moves[best_idx];
    move_list_destroy(&moves);
    return result;
}

/* 3. Bingo Hunter Bot */
static Move select_bingo_hunter_move(const Board *board, const Dawg *dawg, const Rack *rack) {
    MoveList moves = {0};
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        move_list_destroy(&moves);
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
    Move result = moves.moves[best_idx];
    move_list_destroy(&moves);
    return result;
}

/* 4. Heuristic Grandmaster Bot (8-phase rack leave table) */
static Move select_heuristic_move(const Board *board, const Dawg *dawg, const Rack *rack, int bag_count) {
    MoveList moves = {0};
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        move_list_destroy(&moves);
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
    Move result = moves.moves[best_idx];
    move_list_destroy(&moves);
    return result;
}

/* 5. Neural Static Bot (Raw 448-dim NNUE 1-ply) */
static Move select_neural_static_move(const Board *board, const Dawg *dawg, const Rack *rack, int bag_count) {
    MoveList moves = {0};
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        move_list_destroy(&moves);
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
    Move result = moves.moves[best_idx];
    move_list_destroy(&moves);
    return result;
}

/* 6. Championship Simulation Bot (Quackle emulation: paired Monte Carlo lookahead) */
static Move select_quackle_sim_move(const Board *board, const Dawg *dawg, const Rack *rack, int bag_count) {
    MoveList moves = {0};
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        move_list_destroy(&moves);
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
    if (beam <= 1) {
        Move result = moves.moves[0];
        move_list_destroy(&moves);
        return result;
    }

    char unseen[100];
    int unseen_count = sim_get_unseen_tiles(board, rack, unseen);

    float opp_totals[5] = {0};
    int N_SIMS = 10;
    MoveList om = {0};

    for (int s = 0; s < N_SIMS; s++) {
        Rack opp_r;
        rack_init(&opp_r);
        if (bag_count > 0) bot_shuffle_tiles(unseen, unseen_count);
        int d = unseen_count < 7 ? unseen_count : 7;
        for (int i = 0; i < d; i++) rack_add_tile(&opp_r, unseen[i]);

        for (int k = 0; k < beam; k++) {
            Board spec;
            board_copy(&spec, board);
            board_apply_move(&spec, &moves.moves[k]);

            movegen_generate_all(&spec, dawg, &opp_r, &om);
            float max_opp = 0.0f;
            for (int r = 0; r < om.count; r++) {
                float eq = eval_move_heuristic_fast(&spec, &om.moves[r], bag_count);
                if (eq > max_opp) max_opp = eq;
            }
            opp_totals[k] += max_opp;
        }
    }
    move_list_destroy(&om);

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
    Move result = moves.moves[best_k];
    move_list_destroy(&moves);
    return result;
}

/* 7. MAGPIE / Woogles Champion Bot (2-Ply Realized Equity Lookahead Simulation) */
static Move select_magpie_sim_move(const Board *board, const Dawg *dawg, const Rack *rack, int bag_count) {
    MoveList moves = {0};
    movegen_generate_all(board, dawg, rack, &moves);
    if (moves.count == 0) {
        move_list_destroy(&moves);
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
    if (beam <= 1) {
        Move result = moves.moves[0];
        move_list_destroy(&moves);
        return result;
    }

    char unseen[100];
    int unseen_count = sim_get_unseen_tiles(board, rack, unseen);

    float net_totals[5] = {0};
    int N_SIMS = 8;

    for (int s = 0; s < N_SIMS; s++) {
        char sim_unseen[100];
        memcpy(sim_unseen, unseen, unseen_count);
        bot_shuffle_tiles(sim_unseen, unseen_count);

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
            for (int j = 0; moves.moves[k].leave[j]; j++) {
                rack_add_tile(&my_realized_rack, moves.moves[k].leave[j]);
            }
            int refill = 7 - my_realized_rack.count;
            int avail = (unseen_count > d) ? (unseen_count - d) : 0;
            int draw_n = refill < avail ? refill : avail;
            for (int r = 0; r < draw_n; r++) {
                rack_add_tile(&my_realized_rack, sim_unseen[d + r]);
            }

            float my_realized_equity = movegen_find_best_response_equity(&spec, dawg, &my_realized_rack, bag_count > 0 ? bag_count - 1 : 0);

            float net_turn = (float)moves.moves[k].score - (0.75f * opp_equity) + (0.45f * my_realized_equity);
            net_totals[k] += net_turn;
        }
    }

    int best_k = 0;
    float best_net = -10000.0f;
    for (int k = 0; k < beam; k++) {
        float avg_net = net_totals[k] / (float)N_SIMS;
        if (avg_net > best_net) {
            best_net = avg_net;
            best_k = k;
        }
    }
    Move result = moves.moves[best_k];
    move_list_destroy(&moves);
    return result;
}

Move bot_archetype_select_move(BotType type, const Board *board, const Dawg *dawg, const Rack *rack, int bag_count) {
    switch (type) {
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
        default:
            return select_magpie_sim_move(board, dawg, rack, bag_count);
    }
}
