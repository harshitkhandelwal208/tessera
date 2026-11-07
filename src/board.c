#include "board.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* Multiplier grid lookup */
MultiplierType board_get_multiplier(int row, int col) {
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) {
        return MULT_NONE;
    }

    /* TW: 8 squares */
    if ((row == 0 || row == 7 || row == 14) && (col == 0 || col == 7 || col == 14)) {
        if (row == 7 && col == 7) {
            return MULT_DW; /* Center star is Double Word */
        }
        return MULT_TW;
    }

    /* DW: 17 squares (diagonals + center star) */
    if (row == col || row == (BOARD_SIZE - 1 - col)) {
        if ((row >= 1 && row <= 4) || (row >= 10 && row <= 13) || row == 7) {
            return MULT_DW;
        }
    }

    /* TL: 12 squares */
    if ((row == 1 || row == 13) && (col == 5 || col == 9)) return MULT_TL;
    if ((row == 5 || row == 9) && (col == 1 || col == 5 || col == 9 || col == 13)) return MULT_TL;

    /* DL: 24 squares */
    if ((row == 0 || row == 14) && (col == 3 || col == 11)) return MULT_DL;
    if ((row == 2 || row == 12) && (col == 6 || col == 8)) return MULT_DL;
    if ((row == 3 || row == 11) && (col == 0 || col == 7 || col == 14)) return MULT_DL;
    if ((row == 6 || row == 8) && (col == 2 || col == 6 || col == 8 || col == 12)) return MULT_DL;
    if (row == 7 && (col == 3 || col == 11)) return MULT_DL;

    return MULT_NONE;
}

void board_init(Board *b) {
    memset(b, 0, sizeof(Board));
    b->is_empty = true;
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            b->cells[r][c].tile = 0;
            b->cells[r][c].is_blank = false;
            b->cells[r][c].mult = board_get_multiplier(r, c);
        }
    }
}

void board_copy(Board *dst, const Board *src) {
    memcpy(dst, src, sizeof(Board));
}

void board_print(const Board *b) {
    printf("    ");
    for (int c = 0; c < BOARD_SIZE; c++) {
        printf(" %c ", 'A' + c);
    }
    printf("\n   +");
    for (int c = 0; c < BOARD_SIZE; c++) printf("---");
    printf("+\n");

    for (int r = 0; r < BOARD_SIZE; r++) {
        printf("%2d |", r + 1);
        for (int c = 0; c < BOARD_SIZE; c++) {
            char t = b->cells[r][c].tile;
            if (t != 0) {
                if (b->cells[r][c].is_blank) {
                    printf(" %c ", (char)tolower((unsigned char)t));
                } else {
                    printf(" %c ", t);
                }
            } else {
                MultiplierType m = b->cells[r][c].mult;
                switch (m) {
                    case MULT_TW: printf("3W "); break;
                    case MULT_DW: printf(r == 7 && c == 7 ? " ★ " : "2W "); break;
                    case MULT_TL: printf("3L "); break;
                    case MULT_DL: printf("2L "); break;
                    default:      printf(" . "); break;
                }
            }
        }
        printf("|\n");
    }
    printf("   +");
    for (int c = 0; c < BOARD_SIZE; c++) printf("---");
    printf("+\n");
}

bool board_is_anchor(const Board *b, int row, int col) {
    if (b->cells[row][col].tile != 0) return false;

    if (b->is_empty) {
        return (row == 7 && col == 7);
    }

    /* Adjacent to an occupied cell */
    if (row > 0 && b->cells[row - 1][col].tile != 0) return true;
    if (row < BOARD_SIZE - 1 && b->cells[row + 1][col].tile != 0) return true;
    if (col > 0 && b->cells[row][col - 1].tile != 0) return true;
    if (col < BOARD_SIZE - 1 && b->cells[row][col + 1].tile != 0) return true;

    return false;
}

/*
 * Cross-checks for horizontal word placement:
 * Looks vertically (above and below row r at col c).
 */
void board_compute_horizontal_cross_checks(const Board *b, const Dawg *dawg,
                                          CrossCheck cross_checks[BOARD_SIZE][BOARD_SIZE]) {
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            cross_checks[r][c].valid_letters_mask = 0;
            for (int l = 0; l < NUM_LETTERS; l++) {
                cross_checks[r][c].cross_scores[l] = 0;
            }

            if (b->cells[r][c].tile != 0) {
                /* Occupied square cannot receive tiles */
                continue;
            }

            /* Check if there are vertical neighbors */
            bool has_up = (r > 0 && b->cells[r - 1][c].tile != 0);
            bool has_down = (r < BOARD_SIZE - 1 && b->cells[r + 1][c].tile != 0);

            if (!has_up && !has_down) {
                /* No vertical cross-word formed; all letters are allowed */
                cross_checks[r][c].valid_letters_mask = 0x03FFFFFF; /* Bits 0..25 set */
                continue;
            }

            /* Scan upward prefix */
            int up_start = r - 1;
            while (up_start >= 0 && b->cells[up_start][c].tile != 0) {
                up_start--;
            }
            up_start++; /* First letter of upward prefix */

            /* Scan downward suffix */
            int down_end = r + 1;
            while (down_end < BOARD_SIZE && b->cells[down_end][c].tile != 0) {
                down_end++;
            }
            down_end--; /* Last letter of downward suffix */

            /* Calculate prefix base score */
            int prefix_score = 0;
            for (int k = up_start; k < r; k++) {
                if (!b->cells[k][c].is_blank) {
                    prefix_score += TILE_VALUES[b->cells[k][c].tile - 'A'];
                }
            }

            /* Calculate suffix base score */
            int suffix_score = 0;
            for (int k = r + 1; k <= down_end; k++) {
                if (!b->cells[k][c].is_blank) {
                    suffix_score += TILE_VALUES[b->cells[k][c].tile - 'A'];
                }
            }

            /* Multiplier on cell (r, c) */
            MultiplierType m = b->cells[r][c].mult;
            int letter_mult = 1;
            int word_mult = 1;
            if (m == MULT_DL) letter_mult = 2;
            else if (m == MULT_TL) letter_mult = 3;
            else if (m == MULT_DW) word_mult = 2;
            else if (m == MULT_TW) word_mult = 3;

            /* Step upward prefix once if present */
            uint32_t prefix_node = dawg->root_index;
            bool prefix_valid = true;
            for (int k = up_start; k < r; k++) {
                bool term = false;
                prefix_node = dawg_step(dawg, prefix_node, b->cells[k][c].tile, &term);
                if (prefix_node == 0) {
                    prefix_valid = false;
                    break;
                }
            }

            if (!prefix_valid) {
                continue;
            }

            bool has_down_suffix = (r < down_end);

            /* Test each candidate letter A-Z */
            for (int l = 0; l < NUM_LETTERS; l++) {
                char let = (char)('A' + l);
                bool is_term = false;
                uint32_t mid_node = dawg_step(dawg, prefix_node, let, &is_term);

                bool word_valid = false;
                if (!has_down_suffix) {
                    word_valid = is_term;
                } else if (mid_node != 0) {
                    /* Step downward suffix */
                    uint32_t suf_node = mid_node;
                    bool suf_valid = true;
                    for (int k = r + 1; k <= down_end; k++) {
                        is_term = false;
                        suf_node = dawg_step(dawg, suf_node, b->cells[k][c].tile, &is_term);
                        if (suf_node == 0 && k < down_end) {
                            suf_valid = false;
                            break;
                        }
                    }
                    word_valid = (suf_valid && is_term);
                }

                if (word_valid) {
                    cross_checks[r][c].valid_letters_mask |= (1U << l);
                    int placed_letter_val = TILE_VALUES[l] * letter_mult;
                    cross_checks[r][c].cross_scores[l] = (prefix_score + suffix_score + placed_letter_val) * word_mult;
                }
            }
        }
    }
}

/*
 * Cross-checks for vertical word placement:
 * Looks horizontally (left and right of col c at row r).
 */
void board_compute_vertical_cross_checks(const Board *b, const Dawg *dawg,
                                        CrossCheck cross_checks[BOARD_SIZE][BOARD_SIZE]) {
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            cross_checks[r][c].valid_letters_mask = 0;
            for (int l = 0; l < NUM_LETTERS; l++) {
                cross_checks[r][c].cross_scores[l] = 0;
            }

            if (b->cells[r][c].tile != 0) {
                continue;
            }

            bool has_left = (c > 0 && b->cells[r][c - 1].tile != 0);
            bool has_right = (c < BOARD_SIZE - 1 && b->cells[r][c + 1].tile != 0);

            if (!has_left && !has_right) {
                cross_checks[r][c].valid_letters_mask = 0x03FFFFFF;
                continue;
            }

            /* Scan left prefix */
            int left_start = c - 1;
            while (left_start >= 0 && b->cells[r][left_start].tile != 0) {
                left_start--;
            }
            left_start++;

            /* Scan right suffix */
            int right_end = c + 1;
            while (right_end < BOARD_SIZE && b->cells[r][right_end].tile != 0) {
                right_end++;
            }
            right_end--;

            int prefix_score = 0;
            for (int k = left_start; k < c; k++) {
                if (!b->cells[r][k].is_blank) {
                    prefix_score += TILE_VALUES[b->cells[r][k].tile - 'A'];
                }
            }

            int suffix_score = 0;
            for (int k = c + 1; k <= right_end; k++) {
                if (!b->cells[r][k].is_blank) {
                    suffix_score += TILE_VALUES[b->cells[r][k].tile - 'A'];
                }
            }

            MultiplierType m = b->cells[r][c].mult;
            int letter_mult = 1;
            int word_mult = 1;
            if (m == MULT_DL) letter_mult = 2;
            else if (m == MULT_TL) letter_mult = 3;
            else if (m == MULT_DW) word_mult = 2;
            else if (m == MULT_TW) word_mult = 3;

            /* Step left prefix once if present */
            uint32_t prefix_node = dawg->root_index;
            bool prefix_valid = true;
            for (int k = left_start; k < c; k++) {
                bool term = false;
                prefix_node = dawg_step(dawg, prefix_node, b->cells[r][k].tile, &term);
                if (prefix_node == 0) {
                    prefix_valid = false;
                    break;
                }
            }

            if (!prefix_valid) {
                continue;
            }

            bool has_right_suffix = (c < right_end);

            /* Test each candidate letter A-Z */
            for (int l = 0; l < NUM_LETTERS; l++) {
                char let = (char)('A' + l);
                bool is_term = false;
                uint32_t mid_node = dawg_step(dawg, prefix_node, let, &is_term);

                bool word_valid = false;
                if (!has_right_suffix) {
                    word_valid = is_term;
                } else if (mid_node != 0) {
                    uint32_t suf_node = mid_node;
                    bool suf_valid = true;
                    for (int k = c + 1; k <= right_end; k++) {
                        is_term = false;
                        suf_node = dawg_step(dawg, suf_node, b->cells[r][k].tile, &is_term);
                        if (suf_node == 0 && k < right_end) {
                            suf_valid = false;
                            break;
                        }
                    }
                    word_valid = (suf_valid && is_term);
                }

                if (word_valid) {
                    cross_checks[r][c].valid_letters_mask |= (1U << l);
                    int placed_letter_val = TILE_VALUES[l] * letter_mult;
                    cross_checks[r][c].cross_scores[l] = (prefix_score + suffix_score + placed_letter_val) * word_mult;
                }
            }
        }
    }
}

int board_score_move(const Board *b, const Dawg *dawg, const Move *move) {
    (void)dawg;
    if (move->type != MOVE_PLAY) return 0;

    int total_score = 0;
    int main_word_sum = 0;
    int main_word_mult = 1;

    int r = move->row;
    int c = move->col;
    int len = (int)strlen(move->word);

    /* Track placed tile index */
    int placed_idx = 0;

    for (int i = 0; i < len; i++) {
        int cur_r = (move->dir == DIR_HORIZONTAL) ? r : r + i;
        int cur_c = (move->dir == DIR_HORIZONTAL) ? c + i : c;

        bool newly_placed = false;
        bool is_blank = false;
        char letter = move->word[i];

        if (b->cells[cur_r][cur_c].tile == 0) {
            newly_placed = true;
            if (placed_idx < move->num_placed) {
                is_blank = move->placed[placed_idx].is_blank;
                placed_idx++;
            }
        } else {
            is_blank = b->cells[cur_r][cur_c].is_blank;
        }

        int letter_val = is_blank ? 0 : TILE_VALUES[letter - 'A'];

        if (newly_placed) {
            MultiplierType m = b->cells[cur_r][cur_c].mult;
            int letter_mult = 1;
            int cross_word_mult = 1;

            if (m == MULT_DL) letter_mult = 2;
            else if (m == MULT_TL) letter_mult = 3;
            else if (m == MULT_DW) { main_word_mult *= 2; cross_word_mult = 2; }
            else if (m == MULT_TW) { main_word_mult *= 3; cross_word_mult = 3; }

            main_word_sum += letter_val * letter_mult;

            /* Check for cross-word */
            if (move->dir == DIR_HORIZONTAL) {
                bool has_up = (cur_r > 0 && b->cells[cur_r - 1][cur_c].tile != 0);
                bool has_down = (cur_r < BOARD_SIZE - 1 && b->cells[cur_r + 1][cur_c].tile != 0);
                if (has_up || has_down) {
                    int cross_sum = letter_val * letter_mult;
                    /* Upward */
                    int kr = cur_r - 1;
                    while (kr >= 0 && b->cells[kr][cur_c].tile != 0) {
                        if (!b->cells[kr][cur_c].is_blank) {
                            cross_sum += TILE_VALUES[b->cells[kr][cur_c].tile - 'A'];
                        }
                        kr--;
                    }
                    /* Downward */
                    kr = cur_r + 1;
                    while (kr < BOARD_SIZE && b->cells[kr][cur_c].tile != 0) {
                        if (!b->cells[kr][cur_c].is_blank) {
                            cross_sum += TILE_VALUES[b->cells[kr][cur_c].tile - 'A'];
                        }
                        kr++;
                    }
                    total_score += cross_sum * cross_word_mult;
                }
            } else {
                /* DIR_VERTICAL */
                bool has_left = (cur_c > 0 && b->cells[cur_r][cur_c - 1].tile != 0);
                bool has_right = (cur_c < BOARD_SIZE - 1 && b->cells[cur_r][cur_c + 1].tile != 0);
                if (has_left || has_right) {
                    int cross_sum = letter_val * letter_mult;
                    /* Left */
                    int kc = cur_c - 1;
                    while (kc >= 0 && b->cells[cur_r][kc].tile != 0) {
                        if (!b->cells[cur_r][kc].is_blank) {
                            cross_sum += TILE_VALUES[b->cells[cur_r][kc].tile - 'A'];
                        }
                        kc--;
                    }
                    /* Right */
                    kc = cur_c + 1;
                    while (kc < BOARD_SIZE && b->cells[cur_r][kc].tile != 0) {
                        if (!b->cells[cur_r][kc].is_blank) {
                            cross_sum += TILE_VALUES[b->cells[cur_r][kc].tile - 'A'];
                        }
                        kc++;
                    }
                    total_score += cross_sum * cross_word_mult;
                }
            }
        } else {
            /* Existing tile - face value only, no multipliers */
            main_word_sum += letter_val;
        }
    }

    /* Main word score */
    total_score += main_word_sum * main_word_mult;

    /* Bingo bonus */
    if (move->num_placed == 7) {
        total_score += 50;
    }

    return total_score;
}

bool board_apply_move(Board *b, const Move *move) {
    if (move->type != MOVE_PLAY) return true;

    for (int i = 0; i < move->num_placed; i++) {
        int r = move->placed[i].row;
        int c = move->placed[i].col;
        b->cells[r][c].tile = move->placed[i].letter;
        b->cells[r][c].is_blank = move->placed[i].is_blank;
        b->cells[r][c].mult = MULT_NONE; /* Multiplier is consumed */
    }

    b->is_empty = false;
    return true;
}

/* Racks & Bag */
void rack_init(Rack *rack) {
    memset(rack, 0, sizeof(Rack));
}

void rack_add_tile(Rack *rack, char tile) {
    if (rack->count < RACK_CAPACITY) {
        rack->tiles[rack->count++] = (char)toupper((unsigned char)tile);
        rack->tiles[rack->count] = '\0';
    }
}

bool rack_remove_tile(Rack *rack, char tile) {
    tile = (char)toupper((unsigned char)tile);
    for (int i = 0; i < rack->count; i++) {
        if (rack->tiles[i] == tile) {
            for (int j = i; j < rack->count - 1; j++) {
                rack->tiles[j] = rack->tiles[j + 1];
            }
            rack->count--;
            rack->tiles[rack->count] = '\0';
            return true;
        }
    }
    return false;
}

static int compare_chars(const void *a, const void *b) {
    return (*(const char*)a - *(const char*)b);
}

void rack_sort(Rack *rack) {
    if (rack->count > 1) {
        qsort(rack->tiles, rack->count, sizeof(char), compare_chars);
    }
}

void rack_to_string(const Rack *rack, char *out) {
    strncpy(out, rack->tiles, rack->count);
    out[rack->count] = '\0';
}

void bag_init(TileBag *bag) {
    bag->count = 0;
    for (int i = 0; i < 26; i++) {
        int cnt = TILE_BAG_DISTRIBUTION[i];
        for (int k = 0; k < cnt; k++) {
            bag->tiles[bag->count++] = (char)('A' + i);
        }
    }
    /* Blanks */
    for (int k = 0; k < TILE_BAG_DISTRIBUTION[26]; k++) {
        bag->tiles[bag->count++] = '?';
    }
}

void bag_shuffle(TileBag *bag) {
    for (int i = bag->count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        char tmp = bag->tiles[i];
        bag->tiles[i] = bag->tiles[j];
        bag->tiles[j] = tmp;
    }
}

char bag_draw_tile(TileBag *bag) {
    if (bag->count <= 0) return 0;
    char tile = bag->tiles[--bag->count];
    return tile;
}

int bag_draw_tiles(TileBag *bag, Rack *rack, int count) {
    int drawn = 0;
    while (rack->count < RACK_CAPACITY && bag->count > 0 && drawn < count) {
        rack_add_tile(rack, bag_draw_tile(bag));
        drawn++;
    }
    return drawn;
}
