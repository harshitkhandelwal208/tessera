#ifndef EVAL_H
#define EVAL_H

#include "types.h"
#include "board.h"
#include "dict.h"

/*
 * Evaluates the static equity of a rack leave.
 * Accounts for:
 * 1. Single-letter equity (blanks, S, high consonants, heavy tile penalties).
 * 2. Vowel/consonant balance (severe penalties for all-vowel or all-consonant racks).
 * 3. Duplicate tile penalties (II, UU, OO, VV, WW, etc.).
 * 4. High-value synergies and bingo suffixes (QU, ING, ER, ED, ST, RE).
 */
float eval_rack_leave(const char *leave);

/*
 * Evaluates board openness and defensive risk of a move.
 * Penalizes exposing Triple Word / Triple Letter squares to the opponent.
 */
float eval_board_risk(const Board *board, const Move *move);

/*
 * Computes the total static evaluation of a candidate move:
 * Total Equity = Move Score + Leave Equity + Board Risk
 */
float eval_move_static(const Board *board, const Move *move);

/*
 * Evaluates whether an exchange move is superior to any played move.
 * If so, generates the optimal exchange move.
 */
bool eval_should_exchange(const Board *board, const Rack *rack, int bag_count,
                         float best_play_equity, Move *out_exchange_move);

/* Sort moves by static equity descending */
void eval_sort_moves(MoveList *list);

#endif /* EVAL_H */
