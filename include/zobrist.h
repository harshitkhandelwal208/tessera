#ifndef ZOBRIST_H
#define ZOBRIST_H

#include <stdint.h>
#include "types.h"
#include "board.h"

/*
 * Zobrist hash table for Scrabble:
 * 225 board cells * 53 states (0 = empty, 1..26 = regular letters, 27..52 = blanks)
 * + Turn salt (2 values)
 */
#define ZOBRIST_CELL_STATES 53

void zobrist_init(void);

/* Computes 64-bit Zobrist hash of board */
uint64_t zobrist_compute_board(const Board *board, int turn);

/* Incremental update for a placed tile */
uint64_t zobrist_update_tile(uint64_t hash, int row, int col, char letter, bool is_blank);

#endif /* ZOBRIST_H */
