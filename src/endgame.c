#include "endgame.h"
#include "movegen.h"
#include "zobrist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>

#define ENDGAME_TT_CLUSTERS    8192 /* 8,192 * 64 bytes = 512 KB (L2 cache friendly) */
#define ENDGAME_TT_CLUSTER_SIZE 4
#define ENDGAME_INF 100000
#define MAX_ENDGAME_DEPTH 14

typedef enum {
    TT_EXACT = 0,
    TT_LOWER = 1,
    TT_UPPER = 2
} TTFlag;

/* 16-byte packed Transposition Table entry with Zobrist XOR verification */
typedef struct {
    uint64_t key_xor_data;
    int16_t  value;
    int8_t   depth;
    uint8_t  flag;
    uint32_t generation;
} EndgameTTEntry;

/* Exactly 64-byte cluster (1 CPU cacheline) containing 4 entries */
typedef struct {
    alignas(64) EndgameTTEntry entries[ENDGAME_TT_CLUSTER_SIZE];
} EndgameTTCluster;

typedef struct {
    EndgameTTCluster tt[ENDGAME_TT_CLUSTERS];
    bool initialized;
    MoveList moves_pool[MAX_ENDGAME_DEPTH + 2];
    int solve_count;
    uint32_t generation;
    Move root_best_move;
} EndgameThreadContext;

static _Thread_local EndgameThreadContext g_endgame_ctx;

static inline uint64_t endgame_pack_data(int16_t value, int8_t depth, uint8_t flag, uint32_t generation) {
    return ((uint64_t)(uint16_t)value) |
           (((uint64_t)(uint8_t)depth) << 16) |
           (((uint64_t)flag) << 24) |
           (((uint64_t)generation) << 32);
}

static void endgame_tt_clear(void) {
    memset(g_endgame_ctx.tt, 0, sizeof(g_endgame_ctx.tt));
    g_endgame_ctx.initialized = true;
    g_endgame_ctx.generation = 1;
}

static int rack_total_points(const Rack *rack) {
    int sum = 0;
    for (int i = 0; i < rack->count; i++) {
        char t = rack->tiles[i];
        if (t >= 'A' && t <= 'Z') sum += TILE_VALUES[t - 'A'];
    }
    return sum;
}

typedef struct {
    int32_t eff_score;
    int32_t idx;
} EndgameMoveOrder;

static int compare_move_order(const void *a, const void *b) {
    const EndgameMoveOrder *oa = (const EndgameMoveOrder *)a;
    const EndgameMoveOrder *ob = (const EndgameMoveOrder *)b;
    return ob->eff_score - oa->eff_score;
}

static inline void sort_endgame_moves(EndgameMoveOrder *order, int n) {
    if (n <= 1) return;
    if (n <= 32) {
        for (int i = 1; i < n; i++) {
            EndgameMoveOrder key = order[i];
            int j = i - 1;
            while (j >= 0 && order[j].eff_score < key.eff_score) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }
    } else {
        qsort(order, n, sizeof(EndgameMoveOrder), compare_move_order);
    }
}

/*
 * Negamax formulation of perfect-information Scrabble endgame on a position-keyed DAG.
 * Returns the net margin from the perspective of the player to move.
 */
static int negamax_dag(const Board *board, const Dawg *dawg,
                       const Rack *my_rack, const Rack *opp_rack,
                       uint64_t board_hash,
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

    if (depth >= MAX_ENDGAME_DEPTH) {
        /* Bound depth horizon: heuristic margin */
        return -my_rack_val + opp_rack_val;
    }

    uint64_t key = board_hash ^ zobrist_compute_rack(my_rack, 0) ^ zobrist_compute_rack(opp_rack, 1);
    uint32_t cluster_idx = (uint32_t)(key & (ENDGAME_TT_CLUSTERS - 1));
    EndgameTTCluster *cl = &g_endgame_ctx.tt[cluster_idx];
    EndgameTTEntry *victim = &cl->entries[0];
    int lowest_depth = 999;

    for (int i = 0; i < ENDGAME_TT_CLUSTER_SIZE; i++) {
        EndgameTTEntry *e = &cl->entries[i];
        if (e->generation == 0) {
            victim = e;
            break;
        }
        uint64_t data = endgame_pack_data(e->value, e->depth, e->flag, e->generation);
        if ((e->key_xor_data ^ data) == key) {
            if (e->depth >= (MAX_ENDGAME_DEPTH - depth)) {
                if (e->flag == TT_EXACT) {
                    if (out_best && depth == 0) *out_best = g_endgame_ctx.root_best_move;
                    return e->value;
                } else if (e->flag == TT_LOWER && e->value >= beta) {
                    return e->value;
                } else if (e->flag == TT_UPPER && e->value <= alpha) {
                    return e->value;
                }
            }
            victim = e;
            break;
        }

        int prio = (e->generation != g_endgame_ctx.generation ? 0 : 100) + e->depth;
        if (prio < lowest_depth) {
            lowest_depth = prio;
            victim = e;
        }
    }

    MoveList *moves = &g_endgame_ctx.moves_pool[depth];
    move_list_init(moves);
    movegen_generate_all(board, dawg, my_rack, moves);

    if (moves->count == 0) {
        /* Must pass */
        int score = -negamax_dag(board, dawg, opp_rack, my_rack,
                                 zobrist_toggle_turn(board_hash),
                                 -beta, -alpha,
                                 depth + 1, consecutive_passes + 1, NULL);
        if (out_best && depth == 0) {
            memset(out_best, 0, sizeof(Move));
            out_best->type = MOVE_PASS;
        }
        return score;
    }

    /* Fast indexed move ordering: prioritize moves that empty the rack (instant outplay), then by score */
    EndgameMoveOrder order[moves->count];
    for (int i = 0; i < moves->count; i++) {
        order[i].idx = i;
        int eff = moves->moves[i].score;
        if (moves->moves[i].num_placed == my_rack->count) {
            eff += (2 * opp_rack_val); /* Boost for outplay */
        }
        order[i].eff_score = eff;
    }
    sort_endgame_moves(order, moves->count);

    int best_score = -ENDGAME_INF;
    Move best_move = moves->moves[order[0].idx];
    int orig_alpha = alpha;

    /* Dynamic search beam: thorough at root/low depth, focused at deeper plies */
    int beam = moves->count;
    int max_beam = (depth == 0) ? 64 : (depth <= 2 ? 40 : 24);
    if (beam > max_beam) beam = max_beam;

    bool has_evaluated_outplay = false;

    for (int i = 0; i < beam; i++) {
        Move *m = &moves->moves[order[i].idx];

        /* If this move empties my rack, game immediately terminates! */
        if (m->num_placed == my_rack->count) {
            if (!has_evaluated_outplay) {
                has_evaluated_outplay = true;
                int game_out_score = order[i].eff_score;
                if (game_out_score > best_score) {
                    best_score = game_out_score;
                    best_move = *m;
                }
                if (game_out_score > alpha) alpha = game_out_score;
                if (alpha >= beta) break;
            }
            continue;
        }

        /* Create resulting child board and updated rack */
        Board child_board;
        board_copy(&child_board, board);
        board_apply_move(&child_board, m);

        Rack new_my_rack;
        new_my_rack.count = my_rack->count - m->num_placed;
        memcpy(new_my_rack.tiles, m->leave, new_my_rack.count);
        new_my_rack.tiles[new_my_rack.count] = '\0';

        uint64_t child_board_hash = zobrist_toggle_turn(zobrist_update_move(board_hash, m));

        int score = m->score - negamax_dag(&child_board, dawg, opp_rack, &new_my_rack,
                                           child_board_hash,
                                           -beta, -alpha, depth + 1, 0, NULL);

        if (score > best_score) {
            best_score = score;
            best_move = *m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break; /* Beta cutoff */
    }

    /* Transposition DAG entry update with Zobrist XOR verification */
    uint8_t flag = TT_EXACT;
    if (best_score <= orig_alpha) {
        flag = TT_UPPER;
    } else if (best_score >= beta) {
        flag = TT_LOWER;
    }

    int8_t entry_depth = (int8_t)(MAX_ENDGAME_DEPTH - depth);
    victim->value = (int16_t)best_score;
    victim->depth = entry_depth;
    victim->flag = flag;
    victim->generation = g_endgame_ctx.generation;
    victim->key_xor_data = key ^ endgame_pack_data((int16_t)best_score, entry_depth, flag, g_endgame_ctx.generation);

    if (depth == 0) {
        g_endgame_ctx.root_best_move = best_move;
        if (out_best) *out_best = best_move;
    }

    return best_score;
}

bool endgame_solve_exact(const Board *board, const Dawg *dawg,
                         const Rack *my_rack, const Rack *opp_rack,
                         Move *out_best_move, int *out_net_margin) {
    if (!g_endgame_ctx.initialized) {
        endgame_tt_clear();
    } else {
        /* Fast clear every 20 games */
        if (++g_endgame_ctx.solve_count > 20) {
            endgame_tt_clear();
            g_endgame_ctx.solve_count = 0;
        }
    }

    Move best;
    memset(&best, 0, sizeof(Move));
    uint64_t root_board_hash = zobrist_compute_board(board, 0);
    int margin = negamax_dag(board, dawg, my_rack, opp_rack,
                             root_board_hash,
                             -ENDGAME_INF, ENDGAME_INF, 0, 0, &best);

    if (out_best_move) *out_best_move = best;
    if (out_net_margin) *out_net_margin = margin;
    return true;
}

void endgame_cleanup(void) {
    for (int d = 0; d < MAX_ENDGAME_DEPTH + 2; d++) {
        move_list_destroy(&g_endgame_ctx.moves_pool[d]);
    }
    endgame_tt_clear();
}
