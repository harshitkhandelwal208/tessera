#include "dict.h"
#include "board.h"
#include "movegen.h"
#include "endgame.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

int main(void) {
    printf("[TEST] Testing Exact Minimax Endgame Solver...\n");

    Dawg *dawg = dawg_load_binary("data/dawg.bin");
    assert(dawg != NULL);

    Board board;
    board_init(&board);

    /* Place a word in center to create anchors */
    Move m;
    memset(&m, 0, sizeof(Move));
    m.type = MOVE_PLAY;
    m.row = 7; m.col = 5; m.dir = DIR_HORIZONTAL;
    strcpy(m.word, "TRAIN");
    m.num_placed = 5;
    for (int i = 0; i < 5; i++) {
        m.placed[i].row = 7;
        m.placed[i].col = 5 + i;
        m.placed[i].letter = m.word[i];
        m.placed[i].is_blank = false;
    }
    board_apply_move(&board, &m);

    /* Setup endgames: AI rack has "ED" (2 tiles), Opponent has "QX" (2 tiles = 18 pts) */
    Rack ai_rack;
    rack_init(&ai_rack);
    rack_add_tile(&ai_rack, 'E');
    rack_add_tile(&ai_rack, 'D');

    Rack opp_rack;
    rack_init(&opp_rack);
    rack_add_tile(&opp_rack, 'Q');
    rack_add_tile(&opp_rack, 'X');

    EndgameResult res = endgame_solve(&board, dawg, &ai_rack, &opp_rack, 4);

    printf("  Endgame solved!\n");
    printf("  Best play: %s at (%d, %d) dir=%s score=%d\n",
           res.best_move.word, res.best_move.row, res.best_move.col,
           res.best_move.dir == DIR_HORIZONTAL ? "H" : "V", res.best_move.score);
    printf("  Proven Final Spread: %+d pts\n", res.final_spread);
    printf("  Nodes searched: %u\n", res.nodes_searched);

    /* AI can hook ED onto TRAIN to make TRAINED (emptying rack!) */
    assert(res.is_proven);
    assert(res.best_move.num_placed == 2); /* Played out all tiles */
    /* Outplay bonus: 2 * (10 + 8) = +36 pts from opponent's unplayed QX! */
    assert(res.final_spread > 36);

    dawg_free(dawg);
    printf("[TEST] Exact Endgame Solver test PASSED flawlessly!\n");
    return 0;
}
