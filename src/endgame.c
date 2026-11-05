#include "endgame.h"
#include "zobrist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TT_SIZE 65536
#define TT_MASK (TT_SIZE - 1)

typedef enum {
    TT_EXACT = 0,
    TT_LOWER = 1,
    TT_UPPER = 2
} TTFlag;

typedef struct {
    uint64_t key;
    int depth;
    int score;
    TTFlag flag;
    Move best_move;
} TTEntry;

static TTEntry g_endgame_tt[TT_SIZE];
static bool g_tt_initialized = false;

static void init_endgame_tt(void) {
    if (g_tt_initialized) return;
    memset(g_endgame_tt, 0, sizeof(g_endgame_tt));
    g_tt_initialized = true;
}

static uint64_t compute_endgame_hash(const Board *board, const Rack *active, const Rack *idle) {
    uint64_t h = zobrist_compute_board(board, 0);

    /* Mix active rack */
    for (int i = 0; i < active->count; i++) {
        uint64_t t = (unsigned char)active->tiles[i];
        h ^= (t + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    }
    h = (h << 31) | (h >> 33);

    /* Mix idle rack */
    for (int i = 0; i < idle->count; i++) {
        uint64_t t = (unsigned char)idle->tiles[i];
        h ^= (t + 0x517cc1b727220a95ULL + (h << 6) + (h >> 2));
    }
    return h;
}

static int compute_rack_point_value(const Rack *rack) {
    int val = 0;
    for (int i = 0; i < rack->count; i++) {
        char t = rack->tiles[i];
        if (t >= 'A' && t <= 'Z') {
            val += TILE_VALUES[t - 'A'];
        }
    }
    return val;
}

static int minimax_endgame(const Board *board, const Dawg *dawg,
                           const Rack *active_rack, const Rack *idle_rack,
                           int depth, int alpha, int beta, uint32_t *nodes) {
    (*nodes)++;

    /* 1. Terminal: active player has 0 tiles -> outplay bonus */
    if (active_rack->count == 0) {
        int idle_val = compute_rack_point_value(idle_rack);
        return 2 * idle_val;
    }

    /* 2. Horizon reached: static evaluation */
    if (depth <= 0) {
        /*
         * Net point spread from unplayed tiles:
         * (- active_points) - (- idle_points) = idle_points - active_points
         */
        return compute_rack_point_value(idle_rack) - compute_rack_point_value(active_rack);
    }

    /* 3. Probe Transposition Table */
    uint64_t key = compute_endgame_hash(board, active_rack, idle_rack);
    uint32_t tt_idx = (uint32_t)(key & TT_MASK);
    TTEntry *tte = &g_endgame_tt[tt_idx];

    if (tte->key == key && tte->depth >= depth) {
        if (tte->flag == TT_EXACT) return tte->score;
        if (tte->flag == TT_LOWER && tte->score > alpha) alpha = tte->score;
        if (tte->flag == TT_UPPER && tte->score < beta)  beta = tte->score;
        if (alpha >= beta) return tte->score;
    }

    MoveList moves;
    movegen_generate_all(board, dawg, active_rack, &moves);

    /* If no moves possible: active player passes */
    if (moves.count == 0) {
        /* Check if opponent also has no moves -> double pass game over */
        MoveList idle_moves;
        movegen_generate_all(board, dawg, idle_rack, &idle_moves);
        if (idle_moves.count == 0) {
            /* Both players passed: subtract remaining rack values */
            return compute_rack_point_value(idle_rack) - compute_rack_point_value(active_rack);
        }
        /* Active player passes; opponent turn */
        return -minimax_endgame(board, dawg, idle_rack, active_rack, depth - 1, -beta, -alpha, nodes);
    }

    /* Move Ordering: Outplays first, then TT move, then descending score */
    for (int i = 0; i < moves.count - 1; i++) {
        for (int j = i + 1; j < moves.count; j++) {
            bool j_out = (moves.moves[j].num_placed == active_rack->count);
            bool i_out = (moves.moves[i].num_placed == active_rack->count);

            bool swap = false;
            if (j_out && !i_out) {
                swap = true;
            } else if (!j_out && i_out) {
                swap = false;
            } else if (moves.moves[j].score > moves.moves[i].score) {
                swap = true;
            }

            if (swap) {
                Move tmp = moves.moves[i];
                moves.moves[i] = moves.moves[j];
                moves.moves[j] = tmp;
            }
        }
    }

    int best_val = -100000;
    Move best_m = moves.moves[0];
    int original_alpha = alpha;
    int limit = moves.count < 20 ? moves.count : 20;

    for (int i = 0; i < limit; i++) {
        const Move *m = &moves.moves[i];

        /* If this move plays out all remaining active tiles: instant terminal outplay */
        if (m->num_placed == active_rack->count) {
            int idle_val = compute_rack_point_value(idle_rack);
            int spread = m->score + (2 * idle_val);
            if (spread > best_val) {
                best_val = spread;
                best_m = *m;
            }
            if (best_val > alpha) alpha = best_val;
            if (alpha >= beta) break;
            continue;
        }

        /* Speculative board execution */
        Board spec_board;
        board_copy(&spec_board, board);
        board_apply_move(&spec_board, m);

        Rack next_active_rack;
        memcpy(&next_active_rack, active_rack, sizeof(Rack));
        for (int p = 0; p < m->num_placed; p++) {
            char needed = m->placed[p].is_blank ? '?' : m->placed[p].letter;
            rack_remove_tile(&next_active_rack, needed);
        }

        int score = m->score - minimax_endgame(&spec_board, dawg, idle_rack, &next_active_rack,
                                               depth - 1, -beta, -alpha, nodes);

        if (score > best_val) {
            best_val = score;
            best_m = *m;
        }
        if (best_val > alpha) alpha = best_val;
        if (alpha >= beta) break;
    }

    /* Store in TT */
    tte->key = key;
    tte->depth = depth;
    tte->score = best_val;
    tte->best_move = best_m;
    if (best_val <= original_alpha) {
        tte->flag = TT_UPPER;
    } else if (best_val >= beta) {
        tte->flag = TT_LOWER;
    } else {
        tte->flag = TT_EXACT;
    }

    return best_val;
}

EndgameResult endgame_solve(const Board *board, const Dawg *dawg,
                            const Rack *my_rack, const Rack *opp_rack,
                            int max_depth) {
    init_endgame_tt();

    EndgameResult res;
    memset(&res, 0, sizeof(EndgameResult));
    res.is_proven = true;
    res.final_spread = -100000;

    MoveList root_moves;
    movegen_generate_all(board, dawg, my_rack, &root_moves);

    if (root_moves.count == 0) {
        res.best_move.type = MOVE_PASS;
        return res;
    }

    /* Initial ordering: outplays first, then descending by score */
    for (int i = 0; i < root_moves.count - 1; i++) {
        for (int j = i + 1; j < root_moves.count; j++) {
            bool j_out = (root_moves.moves[j].num_placed == my_rack->count);
            bool i_out = (root_moves.moves[i].num_placed == my_rack->count);
            bool swap = false;
            if (j_out && !i_out) swap = true;
            else if (!j_out && i_out) swap = false;
            else if (root_moves.moves[j].score > root_moves.moves[i].score) swap = true;

            if (swap) {
                Move tmp = root_moves.moves[i];
                root_moves.moves[i] = root_moves.moves[j];
                root_moves.moves[j] = tmp;
            }
        }
    }

    int opp_initial_val = compute_rack_point_value(opp_rack);

    /* Dynamic target depth based on remaining tiles */
    int total_tiles = my_rack->count + opp_rack->count;
    int target_depth = max_depth;
    if (target_depth <= 0) target_depth = 4;
    if (total_tiles <= 5) target_depth = 8;
    else if (total_tiles <= 8) target_depth = 6;
    else if (total_tiles <= 10) target_depth = 5;

    Move overall_best_move = root_moves.moves[0];
    int overall_best_spread = -100000;

    /* Iterative Deepening from depth 1 to target_depth */
    for (int d = 1; d <= target_depth; d++) {
        int alpha = -100000;
        int beta = 100000;
        int depth_best_spread = -100000;
        int depth_best_idx = 0;

        for (int i = 0; i < root_moves.count; i++) {
            const Move *m = &root_moves.moves[i];

            /* Direct outplay check */
            if (m->num_placed == my_rack->count) {
                int spread = m->score + (2 * opp_initial_val);
                if (spread > depth_best_spread) {
                    depth_best_spread = spread;
                    depth_best_idx = i;
                }
                if (depth_best_spread > alpha) alpha = depth_best_spread;
                continue;
            }

            Board spec_board;
            board_copy(&spec_board, board);
            board_apply_move(&spec_board, m);

            Rack next_rack;
            memcpy(&next_rack, my_rack, sizeof(Rack));
            for (int p = 0; p < m->num_placed; p++) {
                char needed = m->placed[p].is_blank ? '?' : m->placed[p].letter;
                rack_remove_tile(&next_rack, needed);
            }

            int score = m->score - minimax_endgame(&spec_board, dawg, opp_rack, &next_rack,
                                                   d - 1, -beta, -alpha, &res.nodes_searched);

            if (score > depth_best_spread) {
                depth_best_spread = score;
                depth_best_idx = i;
            }
            if (depth_best_spread > alpha) alpha = depth_best_spread;
        }

        overall_best_spread = depth_best_spread;
        overall_best_move = root_moves.moves[depth_best_idx];

        /* Put best move at index 0 for next depth iteration */
        if (depth_best_idx != 0) {
            Move tmp = root_moves.moves[0];
            root_moves.moves[0] = root_moves.moves[depth_best_idx];
            root_moves.moves[depth_best_idx] = tmp;
        }
    }

    res.final_spread = overall_best_spread;
    res.best_move = overall_best_move;
    res.best_move.sim_equity = (float)res.final_spread;
    res.best_move.leave_equity = (float)res.final_spread;
    return res;
}
