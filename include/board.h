#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "dict.h"

/* Cross-check information for an empty square */
typedef struct {
    uint32_t valid_letters_mask; /* Bits 0..25: which 'A'..'Z' are valid */
    int cross_scores[NUM_LETTERS]; /* Score of cross-word formed by letter (0 if no cross-word) */
} CrossCheck;

/* Board state and fast lookups */
typedef struct {
    BoardCell cells[BOARD_SIZE][BOARD_SIZE];
    bool is_empty; /* True before the very first move of the game */
} Board;

/* Lifecycle */
void board_init(Board *b);
void board_copy(Board *dst, const Board *src);
void board_print(const Board *b);

/* Multipliers */
MultiplierType board_get_multiplier(int row, int col);

/* Cross-check calculation */
/*
 * Computes cross-checks for placing words horizontally.
 * (Cross-words formed vertically above/below).
 */
void board_compute_horizontal_cross_checks(const Board *b, const Dawg *dawg,
                                          CrossCheck cross_checks[BOARD_SIZE][BOARD_SIZE]);

/*
 * Computes cross-checks for placing words vertically.
 * (Cross-words formed horizontally left/right).
 */
void board_compute_vertical_cross_checks(const Board *b, const Dawg *dawg,
                                        CrossCheck cross_checks[BOARD_SIZE][BOARD_SIZE]);

/* Anchor identification */
/* An anchor is an empty square adjacent to an existing tile (or center H8 on first move) */
bool board_is_anchor(const Board *b, int row, int col);

/* Scoring */
int board_score_move(const Board *b, const Dawg *dawg, const Move *move);

/* Move application */
bool board_apply_move(Board *b, const Move *move);

/* Rack & Bag functions */
void rack_init(Rack *rack);
void rack_add_tile(Rack *rack, char tile);
bool rack_remove_tile(Rack *rack, char tile);
void rack_sort(Rack *rack);
void rack_to_string(const Rack *rack, char *out);

void bag_init(TileBag *bag);
void bag_shuffle(TileBag *bag);
char bag_draw_tile(TileBag *bag);
int  bag_draw_tiles(TileBag *bag, Rack *rack, int count);

#endif /* BOARD_H */
