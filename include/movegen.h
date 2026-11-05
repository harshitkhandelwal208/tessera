#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "types.h"
#include "board.h"
#include "dict.h"

/* Initialize a move list */
void move_list_init(MoveList *list);

/*
 * Generate all legal moves for the given rack on the board.
 * Returns the total count of legal moves generated.
 * All moves are fully scored and validated against the DAWG.
 */
int movegen_generate_all(const Board *board, const Dawg *dawg, const Rack *rack, MoveList *list);

/* Transpose helper */
void board_transpose(Board *dst, const Board *src);

#endif /* MOVEGEN_H */
