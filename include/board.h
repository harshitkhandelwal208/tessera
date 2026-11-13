#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "dict.h"

/* 256-bit compact bitboard covering all 225 squares (15x15) */
typedef struct {
    uint64_t b[4];
} Bitboard256;

static inline void bb_set(Bitboard256 *bb, int row, int col) {
    int idx = row * BOARD_SIZE + col;
    bb->b[idx >> 6] |= (1ULL << (idx & 63));
}

static inline void bb_clear(Bitboard256 *bb, int row, int col) {
    int idx = row * BOARD_SIZE + col;
    bb->b[idx >> 6] &= ~(1ULL << (idx & 63));
}

static inline bool bb_test(const Bitboard256 *bb, int row, int col) {
    int idx = row * BOARD_SIZE + col;
    return (bb->b[idx >> 6] & (1ULL << (idx & 63))) != 0;
}

static inline int bb_popcount(const Bitboard256 *bb) {
    return __builtin_popcountll(bb->b[0]) +
           __builtin_popcountll(bb->b[1]) +
           __builtin_popcountll(bb->b[2]) +
           __builtin_popcountll(bb->b[3]);
}

static inline void bb_set_sq(Bitboard256 *bb, int sq) {
    bb->b[sq >> 6] |= (1ULL << (sq & 63));
}

static inline void bb_clear_sq(Bitboard256 *bb, int sq) {
    bb->b[sq >> 6] &= ~(1ULL << (sq & 63));
}

static inline bool bb_test_sq(const Bitboard256 *bb, int sq) {
    return (bb->b[sq >> 6] & (1ULL << (sq & 63))) != 0;
}

static inline int bb_popcount_and(const Bitboard256 *a, const Bitboard256 *b) {
    return __builtin_popcountll(a->b[0] & b->b[0]) +
           __builtin_popcountll(a->b[1] & b->b[1]) +
           __builtin_popcountll(a->b[2] & b->b[2]) +
           __builtin_popcountll(a->b[3] & b->b[3]);
}

static inline bool bb_is_empty(const Bitboard256 *bb) {
    return (bb->b[0] | bb->b[1] | bb->b[2] | bb->b[3]) == 0;
}

static inline int bb_popcount_andnot(const Bitboard256 *a, const Bitboard256 *b) {
    return __builtin_popcountll(a->b[0] & ~b->b[0]) +
           __builtin_popcountll(a->b[1] & ~b->b[1]) +
           __builtin_popcountll(a->b[2] & ~b->b[2]) +
           __builtin_popcountll(a->b[3] & ~b->b[3]);
}

static inline void bb_and(Bitboard256 *dst, const Bitboard256 *a, const Bitboard256 *b) {
    dst->b[0] = a->b[0] & b->b[0];
    dst->b[1] = a->b[1] & b->b[1];
    dst->b[2] = a->b[2] & b->b[2];
    dst->b[3] = a->b[3] & b->b[3];
}

static inline void bb_or(Bitboard256 *dst, const Bitboard256 *a, const Bitboard256 *b) {
    dst->b[0] = a->b[0] | b->b[0];
    dst->b[1] = a->b[1] | b->b[1];
    dst->b[2] = a->b[2] | b->b[2];
    dst->b[3] = a->b[3] | b->b[3];
}

static inline void bb_xor(Bitboard256 *dst, const Bitboard256 *a, const Bitboard256 *b) {
    dst->b[0] = a->b[0] ^ b->b[0];
    dst->b[1] = a->b[1] ^ b->b[1];
    dst->b[2] = a->b[2] ^ b->b[2];
    dst->b[3] = a->b[3] ^ b->b[3];
}

static inline void bb_not(Bitboard256 *dst, const Bitboard256 *a) {
    dst->b[0] = ~a->b[0];
    dst->b[1] = ~a->b[1];
    dst->b[2] = ~a->b[2];
    dst->b[3] = ~a->b[3];
}

static inline int bb_first_sq(const Bitboard256 *bb) {
    if (bb->b[0]) return __builtin_ctzll(bb->b[0]);
    if (bb->b[1]) return 64 + __builtin_ctzll(bb->b[1]);
    if (bb->b[2]) return 128 + __builtin_ctzll(bb->b[2]);
    if (bb->b[3]) {
        int sq = 192 + __builtin_ctzll(bb->b[3]);
        return (sq < TOTAL_SQUARES) ? sq : -1;
    }
    return -1;
}

/* Static multiplier bitboard masks */
extern Bitboard256 BB_TW;
extern Bitboard256 BB_DW;
extern Bitboard256 BB_TL;
extern Bitboard256 BB_DL;
extern Bitboard256 BB_CENTER;

/* Cross-check information for an empty square */
typedef struct {
    uint32_t valid_letters_mask; /* Bits 0..25: which 'A'..'Z' are valid */
    int cross_scores[NUM_LETTERS]; /* Score of cross-word formed by letter (0 if no cross-word) */
} CrossCheck;

/* Board state and fast lookups with compact 256-bit bitboards */
typedef struct {
    BoardCell cells[BOARD_SIZE][BOARD_SIZE];
    Bitboard256 occupied;
    Bitboard256 anchors;
    Bitboard256 blanks;
    bool is_empty; /* True before the very first move of the game */
} Board;

/* Lifecycle */
void board_init(Board *b);
void board_copy(Board *dst, const Board *src);
void board_print(const Board *b);
void board_recompute_bitboards(Board *b);

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
