#include "bayes_rack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const int ENGLISH_BAG_DISTRIBUTION[NUM_DISTINCT_TILES] = {
    9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2
};

static inline int tile_to_index(char t) {
    if (t >= 'A' && t <= 'Z') return t - 'A';
    if (t == '?') return 26;
    return -1;
}

static inline char index_to_tile(int idx) {
    if (idx >= 0 && idx < 26) return (char)('A' + idx);
    if (idx == 26) return '?';
    return 0;
}

void bayes_rack_init(BayesRackTracker *tracker) {
    for (int i = 0; i < NUM_DISTINCT_TILES; i++) {
        tracker->tile_weights[i] = 1.0f;
        tracker->unseen_counts[i] = ENGLISH_BAG_DISTRIBUTION[i];
    }
    tracker->total_unseen = 100;
}

void bayes_rack_update_unseen(BayesRackTracker *tracker, const Board *board, const Rack *active_rack) {
    for (int i = 0; i < NUM_DISTINCT_TILES; i++) {
        tracker->unseen_counts[i] = ENGLISH_BAG_DISTRIBUTION[i];
    }
    tracker->total_unseen = 100;

    /* Subtract tiles on board */
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (board->cells[r][c].tile != 0) {
                int idx = board->cells[r][c].is_blank ? 26 : tile_to_index(board->cells[r][c].tile);
                if (idx >= 0 && tracker->unseen_counts[idx] > 0) {
                    tracker->unseen_counts[idx]--;
                    tracker->total_unseen--;
                }
            }
        }
    }

    /* Subtract tiles on active player's rack */
    for (int i = 0; i < active_rack->count; i++) {
        int idx = tile_to_index(active_rack->tiles[i]);
        if (idx >= 0 && tracker->unseen_counts[idx] > 0) {
            tracker->unseen_counts[idx]--;
            tracker->total_unseen--;
        }
    }
}

void bayes_rack_observe_opponent(BayesRackTracker *tracker, const Board *board_before, const Move *opp_move) {
    (void)board_before;
    if (!opp_move) return;

    if (opp_move->type == MOVE_PASS || opp_move->type == MOVE_EXCHANGE) {
        /* Opponent exchanged or passed: high likelihood of clunky/heavy consonants */
        int heavy_tiles[] = {'Q' - 'A', 'Z' - 'A', 'X' - 'A', 'J' - 'A', 'V' - 'A', 'W' - 'A'};
        for (size_t i = 0; i < sizeof(heavy_tiles)/sizeof(heavy_tiles[0]); i++) {
            tracker->tile_weights[heavy_tiles[i]] = fminf(3.5f, tracker->tile_weights[heavy_tiles[i]] * 1.35f);
        }
        /* Lower probability of holding a blank or S */
        tracker->tile_weights[26] = fmaxf(0.2f, tracker->tile_weights[26] * 0.50f);
        tracker->tile_weights['S' - 'A'] = fmaxf(0.3f, tracker->tile_weights['S' - 'A'] * 0.65f);
        return;
    }

    if (opp_move->type == MOVE_PLAY) {
        /* If opponent played a low scoring play (< 22 pts), discount power tiles */
        if (opp_move->score < 22 && opp_move->num_placed <= 3) {
            tracker->tile_weights['Q' - 'A'] = fmaxf(0.25f, tracker->tile_weights['Q' - 'A'] * 0.70f);
            tracker->tile_weights['Z' - 'A'] = fmaxf(0.30f, tracker->tile_weights['Z' - 'A'] * 0.75f);
            tracker->tile_weights['X' - 'A'] = fmaxf(0.30f, tracker->tile_weights['X' - 'A'] * 0.75f);
            tracker->tile_weights[26]        = fmaxf(0.35f, tracker->tile_weights[26] * 0.80f);
            /* Increase likelihood of holding stem consonants (saving for bingo) */
            tracker->tile_weights['R' - 'A'] = fminf(2.5f, tracker->tile_weights['R' - 'A'] * 1.15f);
            tracker->tile_weights['S' - 'A'] = fminf(2.5f, tracker->tile_weights['S' - 'A'] * 1.20f);
            tracker->tile_weights['T' - 'A'] = fminf(2.5f, tracker->tile_weights['T' - 'A'] * 1.15f);
        } else if (opp_move->score >= 45 || opp_move->num_placed >= 6) {
            /* High scoring or bingo: opponent expended key assets, reset to uniform */
            for (int i = 0; i < NUM_DISTINCT_TILES; i++) {
                tracker->tile_weights[i] = (tracker->tile_weights[i] * 0.5f) + 0.5f;
            }
        }
    }
}

void bayes_rack_sample_rack(const BayesRackTracker *tracker, Rack *out_rack, int draw_count) {
    rack_init(out_rack);
    if (draw_count <= 0 || tracker->total_unseen <= 0) return;

    /* Temporary pool copy for sampling without replacement */
    int pool[NUM_DISTINCT_TILES];
    memcpy(pool, tracker->unseen_counts, sizeof(pool));

    for (int d = 0; d < draw_count; d++) {
        float total_prob = 0.0f;
        float probs[NUM_DISTINCT_TILES] = {0};

        for (int i = 0; i < NUM_DISTINCT_TILES; i++) {
            if (pool[i] > 0) {
                probs[i] = (float)pool[i] * tracker->tile_weights[i];
                total_prob += probs[i];
            }
        }

        if (total_prob <= 0.0f) break;

        /* Spin roulette wheel */
        float r = ((float)rand() / (float)RAND_MAX) * total_prob;
        float accum = 0.0f;
        int chosen_idx = -1;

        for (int i = 0; i < NUM_DISTINCT_TILES; i++) {
            if (pool[i] > 0) {
                accum += probs[i];
                if (r <= accum) {
                    chosen_idx = i;
                    break;
                }
            }
        }

        if (chosen_idx == -1) {
            /* Fallback to last non-empty slot */
            for (int i = NUM_DISTINCT_TILES - 1; i >= 0; i--) {
                if (pool[i] > 0) { chosen_idx = i; break; }
            }
        }

        if (chosen_idx >= 0 && pool[chosen_idx] > 0) {
            char tile = index_to_tile(chosen_idx);
            rack_add_tile(out_rack, tile);
            pool[chosen_idx]--;
        }
    }
}
