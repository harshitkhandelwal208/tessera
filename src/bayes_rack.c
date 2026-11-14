#include "bayes_rack.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static _Thread_local uint64_t s_bayes_prng = 0x9e3779b97f4a7c15ULL;

static inline float fast_bayes_uniform(void) {
    s_bayes_prng ^= s_bayes_prng >> 12;
    s_bayes_prng ^= s_bayes_prng << 25;
    s_bayes_prng ^= s_bayes_prng >> 27;
    uint64_t x = s_bayes_prng * 0x2545F4914F6CDD1DULL;
    return (float)(x >> 40) / 16777216.0f;
}

static inline uint32_t fast_bayes_rand(void) {
    s_bayes_prng ^= s_bayes_prng >> 12;
    s_bayes_prng ^= s_bayes_prng << 25;
    s_bayes_prng ^= s_bayes_prng >> 27;
    return (uint32_t)(s_bayes_prng * 0x2545F4914F6CDD1DULL);
}

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

    /* Subtract tiles on board using occupied bitboard */
    for (int w = 0; w < 4; w++) {
        uint64_t mask = board->occupied.b[w];
        while (mask) {
            int bit = __builtin_ctzll(mask);
            mask &= mask - 1;
            int sq = (w << 6) + bit;
            if (sq >= TOTAL_SQUARES) break;
            int r = sq / BOARD_SIZE;
            int c = sq % BOARD_SIZE;
            int idx = board->cells[r][c].is_blank ? 26 : tile_to_index(board->cells[r][c].tile);
            if (idx >= 0 && tracker->unseen_counts[idx] > 0) {
                tracker->unseen_counts[idx]--;
                tracker->total_unseen--;
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
    if (!opp_move) return;

    if (opp_move->type == MOVE_PASS || opp_move->type == MOVE_EXCHANGE) {
        /* Opponent exchanged or passed: high likelihood of clunky/heavy consonants */
        int heavy_tiles[] = {'Q' - 'A', 'Z' - 'A', 'X' - 'A', 'J' - 'A', 'V' - 'A', 'W' - 'A'};
        for (size_t i = 0; i < sizeof(heavy_tiles)/sizeof(heavy_tiles[0]); i++) {
            tracker->tile_weights[heavy_tiles[i]] = fminf(4.0f, tracker->tile_weights[heavy_tiles[i]] * 1.50f);
        }
        /* Lower probability of holding a blank or S */
        tracker->tile_weights[26] = fmaxf(0.15f, tracker->tile_weights[26] * 0.40f);
        tracker->tile_weights['S' - 'A'] = fmaxf(0.25f, tracker->tile_weights['S' - 'A'] * 0.55f);
        return;
    }

    if (opp_move->type == MOVE_PLAY) {
        /* Check board openness before the opponent's play using bitboards */
        int open_tw = board_before ? bb_popcount_and(&board_before->anchors, &BB_TW) : 0;

        /* If opponent played a low scoring play (< 24 pts) with <= 3 tiles */
        if (opp_move->score < 24 && opp_move->num_placed <= 3) {
            /* If an open TW hotspot existed and opponent didn't use it, power tiles are highly improbable */
            float discount = (open_tw > 0) ? 0.50f : 0.70f;
            tracker->tile_weights['Q' - 'A'] = fmaxf(0.15f, tracker->tile_weights['Q' - 'A'] * discount);
            tracker->tile_weights['Z' - 'A'] = fmaxf(0.20f, tracker->tile_weights['Z' - 'A'] * discount);
            tracker->tile_weights['X' - 'A'] = fmaxf(0.20f, tracker->tile_weights['X' - 'A'] * discount);
            tracker->tile_weights['J' - 'A'] = fmaxf(0.25f, tracker->tile_weights['J' - 'A'] * discount);
            tracker->tile_weights[26]        = fmaxf(0.25f, tracker->tile_weights[26] * discount);

            /* Increase likelihood of holding stem consonants (preserving a 4-5 tile bingo stem) */
            tracker->tile_weights['E' - 'A'] = fminf(2.5f, tracker->tile_weights['E' - 'A'] * 1.15f);
            tracker->tile_weights['R' - 'A'] = fminf(2.5f, tracker->tile_weights['R' - 'A'] * 1.20f);
            tracker->tile_weights['S' - 'A'] = fminf(2.5f, tracker->tile_weights['S' - 'A'] * 1.25f);
            tracker->tile_weights['T' - 'A'] = fminf(2.5f, tracker->tile_weights['T' - 'A'] * 1.20f);
            tracker->tile_weights['I' - 'A'] = fminf(2.5f, tracker->tile_weights['I' - 'A'] * 1.15f);
            tracker->tile_weights['N' - 'A'] = fminf(2.5f, tracker->tile_weights['N' - 'A'] * 1.15f);
            tracker->tile_weights['A' - 'A'] = fminf(2.5f, tracker->tile_weights['A' - 'A'] * 1.15f);
        } else if (opp_move->score >= 45 || opp_move->num_placed >= 6) {
            /* High scoring or bingo: opponent expended key assets, reset to uniform */
            for (int i = 0; i < NUM_DISTINCT_TILES; i++) {
                tracker->tile_weights[i] = (tracker->tile_weights[i] * 0.4f) + 0.6f;
            }
        }
    }
}

void bayes_rack_sample_with_bag(const BayesRackTracker *tracker, Rack *out_opp_rack, int opp_draw_count,
                                char *out_remaining_bag, int *out_remaining_count) {
    rack_init(out_opp_rack);
    if (out_remaining_count) *out_remaining_count = 0;
    if (tracker->total_unseen <= 0) return;

    /* Temporary pool copy for sampling without replacement */
    int pool[NUM_DISTINCT_TILES];
    memcpy(pool, tracker->unseen_counts, sizeof(pool));

    /* Precompute unnormalized probabilities once; update incrementally in O(1) per draw */
    float probs[NUM_DISTINCT_TILES];
    float total_prob = 0.0f;
    for (int i = 0; i < NUM_DISTINCT_TILES; i++) {
        probs[i] = (float)pool[i] * tracker->tile_weights[i];
        total_prob += probs[i];
    }

    for (int d = 0; d < opp_draw_count; d++) {
        if (total_prob <= 1e-6f) break;

        /* Spin roulette wheel */
        float r = fast_bayes_uniform() * total_prob;
        float accum = 0.0f;
        int chosen_idx = -1;

        for (int i = 0; i < NUM_DISTINCT_TILES; i++) {
            if (probs[i] > 0.0f) {
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
            rack_add_tile(out_opp_rack, tile);
            pool[chosen_idx]--;
            float w = tracker->tile_weights[chosen_idx];
            probs[chosen_idx] = fmaxf(0.0f, probs[chosen_idx] - w);
            total_prob = fmaxf(0.0f, total_prob - w);
        }
    }

    /* Collect remaining unseen tiles into out_remaining_bag */
    if (out_remaining_bag && out_remaining_count) {
        int rem_cnt = 0;
        for (int i = 0; i < NUM_DISTINCT_TILES; i++) {
            char t = index_to_tile(i);
            for (int k = 0; k < pool[i]; k++) {
                out_remaining_bag[rem_cnt++] = t;
            }
        }
        /* Knuth / Fisher-Yates selection sampling: shuffle only the prefix needed for draw realization */
        int num_to_shuffle = rem_cnt < RACK_CAPACITY ? rem_cnt : RACK_CAPACITY;
        for (int i = 0; i < num_to_shuffle; i++) {
            int j = i + (int)(fast_bayes_rand() % (uint32_t)(rem_cnt - i));
            char tmp = out_remaining_bag[i];
            out_remaining_bag[i] = out_remaining_bag[j];
            out_remaining_bag[j] = tmp;
        }
        *out_remaining_count = rem_cnt;
    }
}

void bayes_rack_sample_rack(const BayesRackTracker *tracker, Rack *out_rack, int draw_count) {
    bayes_rack_sample_with_bag(tracker, out_rack, draw_count, NULL, NULL);
}
