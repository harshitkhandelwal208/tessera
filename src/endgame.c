#include "endgame.h"
#include "movegen.h"
#include "zobrist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ENDGAME_TT_SIZE 32768
#define ENDGAME_INF 100000

typedef enum {
    TT_EXACT = 0,
    TT_LOWER = 1,
    TT_UPPER = 2
} TTFlag;

typedef struct {
    uint64_t key;
    int depth;
    int value;
    TTFlag flag;
    Move best_move;
} EndgameTTEntry;

static EndgameTTEntry g_endgame_tt[ENDGAME_TT_SIZE];
static bool g_tt_initialized = false;

static void endgame_tt_clear(void) {
    memset(g_endgame_tt, 0, sizeof(g_endgame_tt));
    g_tt_initialized = true;
}

static uint64_t compute_endgame_key(const Board *board, const Rack *r1, const Rack *r2, int turn) {
    uint64_t k = zobrist_compute_board(board, turn);
    for (int i = 0; i < r1->count; i++) {
        k ^= (uint64_t)r1->tiles[i] * 1000003UL;
    }
    for (int i = 0; i < r2->count; i++) {
        k ^= (uint64_t)r2->tiles[i] * 3000017UL;
    }
    return k;
}

static int rack_total_points(const Rack *rack) {
    int sum = 0;
    for (int i = 0; i < rack->count; i++) {
        char t = rack->tiles[i];
        if (t >= 'A' && t <= 'Z') sum += TILE_VALUES[t - 'A'];
    }
    return sum;
}

/*
 * Negamax formulation of perfect-information Scrabble endgame on a position-keyed DAG.
 * Returns the net margin from the perspective of the player to move.
 */
static int negamax_dag(const Board *board, const Dawg *dawg,
                       const Rack *my_rack, const Rack *opp_rack,
                       int alpha, int beta, int depth, int consecutive_passes,
                       Move *out_best) {
    int my_rack_val = rack_total_points(my_rack);
    int opp_rack_val = rack_total_points(opp_rack);

    /* Terminal: Both players passed consecutively */
    if (consecutive_passes >= 2) {
        return -my_rack_val + opp_rack_val;
    }

    /* Terminal: My rack is empty (I went out previously or opponent went out) */
    if (my_rack->count == 0) {
        return 2 * opp_rack_val;
    }
    if (opp_rack->count == 0) {
        return -2 * my_rack_val;
    }

    if (depth >= 6) {
        /* Bound depth horizon: heuristic margin */
        return -my_rack_val + opp_rack_val;
    }

    uint64_t key = compute_endgame_key(board, my_rack, opp_rack, depth % 2);
    uint32_t tt_idx = (uint32_t)(key & (ENDGAME_TT_SIZE - 1));
    EndgameTTEntry *entry = &g_endgame_tt[tt_idx];

    if (entry->key == key && entry->depth >= (6 - depth)) {
        if (entry->flag == TT_EXACT) {
            if (out_best && depth == 0) *out_best = entry->best_move;
            return entry->value;
        } else if (entry->flag == TT_LOWER && entry->value >= beta) {
            return entry->value;
        } else if (entry->flag == TT_UPPER && entry->value <= alpha) {
            return entry->value;
        }
    }

    MoveList moves;
    movegen_generate_all(board, dawg, my_rack, &moves);

    if (moves.count == 0) {
        /* Must pass */
        int score = -negamax_dag(board, dawg, opp_rack, my_rack, -beta, -alpha,
                                 depth + 1, consecutive_passes + 1, NULL);
        if (out_best && depth == 0) {
            memset(out_best, 0, sizeof(Move));
            out_best->type = MOVE_PASS;
        }
        return score;
    }

    /* Simple insertion sort by score descending for move ordering */
    for (int i = 0; i < moves.count - 1; i++) {
        for (int j = i + 1; j < moves.count; j++) {
            if (moves.moves[j].score > moves.moves[i].score) {
                Move tmp = moves.moves[i];
                moves.moves[i] = moves.moves[j];
                moves.moves[j] = tmp;
            }
        }
    }

    int best_score = -ENDGAME_INF;
    Move best_move = moves.moves[0];
    int orig_alpha = alpha;

    /* Search beam: inspect up to 16 best candidate plays in endgame */
    int beam = moves.count < 16 ? moves.count : 16;

    for (int i = 0; i < beam; i++) {
        Move *m = &moves.moves[i];

        /* If this move empties my rack, game immediately terminates! */
        if (m->num_placed == my_rack->count) {
            int game_out_score = m->score + (2 * opp_rack_val);
            if (game_out_score > best_score) {
                best_score = game_out_score;
                best_move = *m;
            }
            if (game_out_score > alpha) alpha = game_out_score;
            if (alpha >= beta) break;
            continue;
        }

        /* Create resulting child board and updated rack */
        Board child_board;
        board_copy(&child_board, board);
        board_apply_move(&child_board, m);

        Rack new_my_rack;
        memcpy(&new_my_rack, my_rack, sizeof(Rack));
        for (int p = 0; p < m->num_placed; p++) {
            char t = m->placed[p].is_blank ? '?' : m->placed[p].letter;
            rack_remove_tile(&new_my_rack, t);
        }

        int score = m->score - negamax_dag(&child_board, dawg, opp_rack, &new_my_rack,
                                           -beta, -alpha, depth + 1, 0, NULL);

        if (score > best_score) {
            best_score = score;
            best_move = *m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break; /* Beta cutoff */
    }

    /* Transposition DAG entry update */
    entry->key = key;
    entry->depth = 6 - depth;
    entry->value = best_score;
    entry->best_move = best_move;
    if (best_score <= orig_alpha) {
        entry->flag = TT_UPPER;
    } else if (best_score >= beta) {
        entry->flag = TT_LOWER;
    } else {
        entry->flag = TT_EXACT;
    }

    if (out_best && depth == 0) {
        *out_best = best_move;
    }

    return best_score;
}

bool endgame_solve_exact(const Board *board, const Dawg *dawg,
                         const Rack *my_rack, const Rack *opp_rack,
                         Move *out_best_move, int *out_net_margin) {
    if (!g_tt_initialized) endgame_tt_clear();
    else {
        /* Fast clear every 20 games */
        static int solve_count = 0;
        if (++solve_count > 20) {
            endgame_tt_clear();
            solve_count = 0;
        }
    }

    Move best;
    memset(&best, 0, sizeof(Move));
    int margin = negamax_dag(board, dawg, my_rack, opp_rack,
                             -ENDGAME_INF, ENDGAME_INF, 0, 0, &best);

    if (out_best_move) *out_best_move = best;
    if (out_net_margin) *out_net_margin = margin;
    return true;
}
