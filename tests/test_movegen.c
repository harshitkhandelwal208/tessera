#include "dict.h"
#include "board.h"
#include "movegen.h"
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <string.h>

int main(void) {
    printf("[TEST] Loading DAWG from data/dawg.bin...\n");
    Dawg *dawg = dawg_load_binary("data/dawg.bin");
    assert(dawg != NULL);

    Board board;
    board_init(&board);

    Rack rack;
    rack_init(&rack);
    /* Classic championship rack: RETAINS */
    rack_add_tile(&rack, 'R');
    rack_add_tile(&rack, 'E');
    rack_add_tile(&rack, 'T');
    rack_add_tile(&rack, 'A');
    rack_add_tile(&rack, 'I');
    rack_add_tile(&rack, 'N');
    rack_add_tile(&rack, 'S');

    printf("[TEST] Generating opening moves with rack RETAINS...\n");
    MoveList moves;
    clock_t start = clock();
    int count = movegen_generate_all(&board, dawg, &rack, &moves);
    clock_t end = clock();
    double elapsed_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    printf("[TEST] Generated %d legal moves in %.2f ms!\n", count, elapsed_ms);
    assert(count > 0);

    /* Find the highest scoring move */
    int max_score = -1;
    int best_idx = -1;
    for (int i = 0; i < moves.count; i++) {
        if (moves.moves[i].score > max_score) {
            max_score = moves.moves[i].score;
            best_idx = i;
        }
    }

    assert(best_idx >= 0);
    Move *best = &moves.moves[best_idx];
    printf("[TEST] Best opening move: %s at (%d, %d) dir=%s score=%d leave='%s'\n",
           best->word, best->row, best->col,
           best->dir == DIR_HORIZONTAL ? "H" : "V",
           best->score, best->leave);

    /* 7-letter bingo on double word center should score at least 50 + 2*word_score! */
    assert(best->score >= 60);
    assert(best->num_placed == 7);

    /* Apply the best move to the board */
    board_apply_move(&board, best);
    printf("[TEST] Board state after best opening move:\n");
    board_print(&board);

    /* Opponent's turn with rack: BLANK + letters */
    Rack opp_rack;
    rack_init(&opp_rack);
    rack_add_tile(&opp_rack, 'Q');
    rack_add_tile(&opp_rack, 'U');
    rack_add_tile(&opp_rack, 'I');
    rack_add_tile(&opp_rack, 'Z');
    rack_add_tile(&opp_rack, 'Z');
    rack_add_tile(&opp_rack, 'E');
    rack_add_tile(&opp_rack, '?'); /* Blank tile */

    printf("[TEST] Generating moves for opponent rack QUIZZE? on populated board...\n");
    start = clock();
    count = movegen_generate_all(&board, dawg, &opp_rack, &moves);
    end = clock();
    elapsed_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("[TEST] Generated %d legal moves in %.2f ms!\n", count, elapsed_ms);
    assert(count > 0);

    max_score = -1;
    best_idx = -1;
    for (int i = 0; i < moves.count; i++) {
        if (moves.moves[i].score > max_score) {
            max_score = moves.moves[i].score;
            best_idx = i;
        }
    }
    best = &moves.moves[best_idx];
    printf("[TEST] Best response move: %s at (%d, %d) dir=%s score=%d leave='%s'\n",
           best->word, best->row, best->col,
           best->dir == DIR_HORIZONTAL ? "H" : "V",
           best->score, best->leave);

    board_apply_move(&board, best);
    printf("[TEST] Board after response:\n");
    board_print(&board);

    dawg_free(dawg);
    printf("[TEST] Move generation tests PASSED flawlessly!\n");
    return 0;
}
