#include "eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

/*
 * Hand-tuned single-tile leave equities (based on Quackle/Maven championship values).
 * Index 0..25 for 'A'..'Z', index 26 for Blank '?'.
 */
static const float SINGLE_TILE_EQUITY[27] = {
    /* A */  0.2f, /* B */ -1.0f, /* C */  1.5f, /* D */  0.6f, /* E */  0.5f,
    /* F */ -2.0f, /* G */ -1.5f, /* H */  0.8f, /* I */ -1.5f, /* J */ -3.0f,
    /* K */ -1.2f, /* L */ -0.2f, /* M */  0.8f, /* N */  0.5f, /* O */ -2.5f,
    /* P */ -0.5f, /* Q */ -7.5f, /* R */  1.2f, /* S */  8.0f, /* T */  0.4f,
    /* U */ -3.0f, /* V */ -5.5f, /* W */ -2.0f, /* X */  3.5f, /* Y */ -1.5f,
    /* Z */  4.0f,
    /* ? (Blank) */ 26.5f
};

static bool is_vowel(char c) {
    c = (char)toupper((unsigned char)c);
    return (c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U');
}

float eval_rack_leave(const char *leave) {
    if (!leave) return 0.0f;
    int len = (int)strlen(leave);
    if (len == 0) return 0.0f; /* 0 tiles left = Bingo just played! */

    float equity = 0.0f;
    int letter_counts[27] = {0};
    int vowels = 0;
    int consonants = 0;
    int blanks = 0;

    for (int i = 0; i < len; i++) {
        char c = (char)toupper((unsigned char)leave[i]);
        if (c == '?') {
            letter_counts[26]++;
            blanks++;
            equity += SINGLE_TILE_EQUITY[26];
        } else if (c >= 'A' && c <= 'Z') {
            int idx = c - 'A';
            letter_counts[idx]++;
            equity += SINGLE_TILE_EQUITY[idx];
            if (is_vowel(c)) vowels++;
            else consonants++;
        }
    }

    /* 1. Duplicate tile penalties */
    for (int i = 0; i < 26; i++) {
        int cnt = letter_counts[i];
        if (cnt >= 2) {
            char let = (char)('A' + i);
            if (let == 'I') equity -= (cnt == 2) ? 4.5f : 12.0f;
            else if (let == 'U') equity -= (cnt == 2) ? 6.0f : 14.0f;
            else if (let == 'O') equity -= (cnt == 2) ? 4.5f : 12.0f;
            else if (let == 'A') equity -= (cnt == 2) ? 3.0f : 10.0f;
            else if (let == 'E') equity -= (cnt == 2) ? 1.0f : 6.0f;
            else if (let == 'V' || let == 'W' || let == 'F' || let == 'J' || let == 'Q' || let == 'Z') {
                equity -= (cnt == 2) ? 7.0f : 16.0f;
            } else {
                equity -= (cnt == 2) ? 2.5f : 8.0f;
            }
        }
    }

    /* 2. Vowel / Consonant ratio balance */
    /* Blanks act as flexible wildcard balancing */
    int non_blanks = vowels + consonants;
    if (non_blanks >= 2) {
        if (vowels == 0) {
            /* Severe consonant-clog penalty */
            if (non_blanks == 2) equity -= 2.5f;
            else if (non_blanks == 3) equity -= 6.0f;
            else if (non_blanks == 4) equity -= 10.0f;
            else if (non_blanks >= 5) equity -= 16.0f;
        } else if (consonants == 0) {
            /* Severe vowel-dump penalty */
            if (non_blanks == 2) equity -= 3.5f;
            else if (non_blanks == 3) equity -= 7.0f;
            else if (non_blanks == 4) equity -= 12.0f;
            else if (non_blanks >= 5) equity -= 18.0f;
        } else {
            /* Balanced mixtures */
            if (non_blanks == 2 && vowels == 1 && consonants == 1) equity += 1.0f;
            else if (non_blanks == 3 && (vowels == 1 || vowels == 2)) equity += 1.2f;
            else if (non_blanks == 4) {
                if (vowels == 2 && consonants == 2) equity += 2.0f;
                else if (vowels == 3) equity -= 2.0f;
            } else if (non_blanks == 5) {
                if (vowels == 2 && consonants == 3) equity += 2.0f;
                else if (vowels == 3 && consonants == 2) equity += 1.5f;
                else if (vowels >= 4) equity -= 8.0f;
            } else if (non_blanks == 6) {
                if (vowels == 2 || vowels == 3) equity += 1.5f;
                else if (vowels >= 4) equity -= 9.0f;
            }
        }
    }

    /* 3. Synergies & Bingo Kernels */
    /* QU synergy */
    if (letter_counts['Q' - 'A'] > 0 && letter_counts['U' - 'A'] > 0) {
        equity += 9.0f; /* Offsets Q penalty dramatically */
    }

    /* Blank + S combo */
    if (blanks > 0 && letter_counts['S' - 'A'] > 0) {
        equity += 4.0f;
    }

    /* Common bingo components */
    if (letter_counts['I' - 'A'] > 0 && letter_counts['N' - 'A'] > 0 && letter_counts['G' - 'A'] > 0) {
        equity += 5.0f;
    }
    if (letter_counts['E' - 'A'] > 0 && letter_counts['R' - 'A'] > 0) {
        equity += 2.5f;
    }
    if (letter_counts['E' - 'A'] > 0 && letter_counts['D' - 'A'] > 0) {
        equity += 2.0f;
    }
    if (letter_counts['S' - 'A'] > 0 && letter_counts['T' - 'A'] > 0) {
        equity += 1.5f;
    }
    if (letter_counts['R' - 'A'] > 0 && letter_counts['E' - 'A'] > 0 && letter_counts['T' - 'A'] > 0) {
        equity += 2.0f;
    }

    return equity;
}

float eval_board_risk(const Board *board, const Move *move) {
    if (move->type != MOVE_PLAY) return 0.0f;

    float risk = 0.0f;

    /*
     * Check if newly placed tiles expose high-value premium squares (TW, TL)
     * to the opponent.
     */
    for (int p = 0; p < move->num_placed; p++) {
        int r = move->placed[p].row;
        int c = move->placed[p].col;
        char let = move->placed[p].letter;
        bool is_v = is_vowel(let);

        /* Check orthogonal neighbors for unblocked TW/TL */
        int dr[4] = {-1, 1, 0, 0};
        int dc[4] = {0, 0, -1, 1};

        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d];
            int nc = c + dc[d];
            if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                if (board->cells[nr][nc].tile == 0) {
                    MultiplierType m = board->cells[nr][nc].mult;
                    if (m == MULT_TW) {
                        /* Exposing TW directly: dangerous! */
                        risk -= is_v ? 9.0f : 5.0f;
                    } else if (m == MULT_TL) {
                        risk -= is_v ? 4.0f : 2.0f;
                    }
                }
            }
        }

        /* Check if our move actually covered a TW or TL */
        MultiplierType original_mult = board_get_multiplier(r, c);
        if (original_mult == MULT_TW) {
            risk += 5.0f; /* Denied opponent TW */
        } else if (original_mult == MULT_TL) {
            risk += 2.0f; /* Denied opponent TL */
        }
    }

    return risk;
}

float eval_move_static(const Board *board, const Move *move) {
    if (move->type != MOVE_PLAY) return 0.0f;

    float leave_eq = eval_rack_leave(move->leave);
    float risk = eval_board_risk(board, move);
    return (float)move->score + leave_eq + risk;
}

static int compare_moves_desc(const void *a, const void *b) {
    const Move *ma = (const Move*)a;
    const Move *mb = (const Move*)b;
    float ea = ma->sim_equity != 0.0f ? ma->sim_equity : ma->leave_equity;
    float eb = mb->sim_equity != 0.0f ? mb->sim_equity : mb->leave_equity;
    if (eb > ea) return 1;
    if (eb < ea) return -1;
    return 0;
}

void eval_sort_moves(MoveList *list) {
    if (list->count > 1) {
        qsort(list->moves, list->count, sizeof(Move), compare_moves_desc);
    }
}

bool eval_should_exchange(const Board *board, const Rack *rack, int bag_count,
                         float best_play_equity, Move *out_exchange_move) {
    (void)board;
    /* Can only exchange if bag has at least 7 tiles */
    if (bag_count < RACK_CAPACITY || rack->count < RACK_CAPACITY) {
        return false;
    }

    /* Evaluate the current rack's leave */
    char full_rack[RACK_CAPACITY + 1];
    rack_to_string(rack, full_rack);
    float current_rack_equity = eval_rack_leave(full_rack);

    /* If best play has very poor equity and current rack is severely clogged */
    if (best_play_equity < 12.0f && current_rack_equity < -10.0f) {
        /* Identify tiles to discard (discard vowels if too many vowels, or duplicate heavy tiles, or Q without U) */
        out_exchange_move->type = MOVE_EXCHANGE;
        out_exchange_move->num_placed = 0;
        out_exchange_move->score = 0;

        int vowels = 0;
        for (int i = 0; i < rack->count; i++) {
            if (is_vowel(rack->tiles[i])) vowels++;
        }

        int discard_count = 0;
        char keep[RACK_CAPACITY + 1];
        int keep_count = 0;

        for (int i = 0; i < rack->count; i++) {
            char c = rack->tiles[i];
            bool discard = false;
            if (c == 'Q') discard = true;
            else if (c == 'V' || c == 'W' || c == 'J') discard = true;
            else if (vowels >= 4 && is_vowel(c)) { discard = true; vowels--; }
            else if (c == '?') discard = false; /* Never discard blank */
            else if (c == 'S') discard = false; /* Never discard S */

            if (discard) {
                out_exchange_move->placed[discard_count].letter = c;
                discard_count++;
            } else {
                keep[keep_count++] = c;
            }
        }
        keep[keep_count] = '\0';

        if (discard_count > 0) {
            out_exchange_move->num_placed = discard_count;
            strcpy(out_exchange_move->leave, keep);
            out_exchange_move->leave_equity = eval_rack_leave(keep) - 2.0f; /* Small tempo penalty */
            out_exchange_move->sim_equity = out_exchange_move->leave_equity;
            return (out_exchange_move->leave_equity > best_play_equity);
        }
    }

    return false;
}
