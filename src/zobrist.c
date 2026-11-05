#include "zobrist.h"
#include <ctype.h>

static uint64_t zobrist_cells[TOTAL_SQUARES][ZOBRIST_CELL_STATES];
static uint64_t zobrist_turn[2];
static bool zobrist_initialized = false;

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void zobrist_init(void) {
    if (zobrist_initialized) return;

    uint64_t seed = 0x853c49e6748fea9bULL;
    for (int i = 0; i < TOTAL_SQUARES; i++) {
        for (int s = 0; s < ZOBRIST_CELL_STATES; s++) {
            zobrist_cells[i][s] = splitmix64(&seed);
        }
    }
    zobrist_turn[0] = splitmix64(&seed);
    zobrist_turn[1] = splitmix64(&seed);

    zobrist_initialized = true;
}

static inline int cell_state_index(char letter, bool is_blank) {
    if (letter == 0) return 0;
    char up = (char)toupper((unsigned char)letter);
    if (up < 'A' || up > 'Z') return 0;
    int let_idx = up - 'A';
    return is_blank ? (27 + let_idx) : (1 + let_idx);
}

uint64_t zobrist_compute_board(const Board *board, int turn) {
    if (!zobrist_initialized) zobrist_init();

    uint64_t h = zobrist_turn[turn ? 1 : 0];
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            int sq = r * BOARD_SIZE + c;
            int state = cell_state_index(board->cells[r][c].tile, board->cells[r][c].is_blank);
            h ^= zobrist_cells[sq][state];
        }
    }
    return h;
}

uint64_t zobrist_update_tile(uint64_t hash, int row, int col, char letter, bool is_blank) {
    if (!zobrist_initialized) zobrist_init();

    int sq = row * BOARD_SIZE + col;
    int state = cell_state_index(letter, is_blank);
    /* XOR out empty state 0 and XOR in new state */
    return hash ^ zobrist_cells[sq][0] ^ zobrist_cells[sq][state];
}
