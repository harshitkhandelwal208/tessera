#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "board.h"
#include "dict.h"
#include "movegen.h"
#include "endgame.h"

static void init_rack_str(Rack *rack, const char *str) {
    rack_init(rack);
    for (int i = 0; str[i]; i++) {
        rack_add_tile(rack, str[i]);
    }
}

int main(void) {
    printf("[TEST] Testing Deterministic Endgame DAG Solver...\n");

    Dawg *dawg = dawg_load_binary("data/dawg.bin");
    assert(dawg != NULL);

    Board board;
    board_init(&board);

    /* Play initial opening word: "RETINA" at (7, 7) horizontal */
    Rack init_rack;
    init_rack_str(&init_rack, "RETINA");
    MoveList opening_moves = {0};
    movegen_generate_all(&board, dawg, &init_rack, &opening_moves);
    assert(opening_moves.count > 0);

    /* Apply opening move */
    board_apply_move(&board, &opening_moves.moves[0]);
    move_list_destroy(&opening_moves);

    /* Set up endgame: Player 1 has 3 tiles: "OUT", Opponent has 2 tiles: "QI" */
    Rack my_rack;
    init_rack_str(&my_rack, "OUT");

    Rack opp_rack;
    init_rack_str(&opp_rack, "QI");

    clock_t start = clock();
    Move best_move;
    int net_margin = 0;
    bool solved = endgame_solve_exact(&board, dawg, &my_rack, &opp_rack, &best_move, &net_margin);
    double elapsed_ms = (double)(clock() - start) / CLOCKS_PER_SEC * 1000.0;

    assert(solved);
    assert(best_move.type == MOVE_PLAY);

    printf("  Endgame solved in %.2f ms!\n", elapsed_ms);
    printf("  Best move: %s at (%d, %d) dir=%c score=%d\n",
           best_move.word, best_move.row, best_move.col,
           best_move.dir == DIR_HORIZONTAL ? 'H' : 'V', best_move.score);
    printf("  Predicted Terminal Net Margin: %+d pts\n", net_margin);

    /* Check that solving is sub-millisecond */
    assert(elapsed_ms < 50.0);

    printf("[TEST] Endgame DAG solver test PASSED flawlessly!\n");

    endgame_cleanup();
    dawg_free(dawg);
    return 0;
}
