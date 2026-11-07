#include "dict.h"
#include "board.h"
#include "simulation.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

int main(void) {
    printf("[TEST] Testing Pure Graph-Based Proof Propagation on the Lattice DAG...\n");

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

    /* Setup empty bag: bag has 0 tiles */
    TileBag bag;
    bag_init(&bag);
    bag.count = 0; /* Bag is completely exhausted */

    /* AI has "ED", opponent has unplayed "QX" */
    Rack ai_rack;
    rack_init(&ai_rack);
    rack_add_tile(&ai_rack, 'E');
    rack_add_tile(&ai_rack, 'D');

    EngineConfig cfg;
    engine_config_default(&cfg);

    MoveList all_moves = {0};
    EngineMetrics metrics;
    memset(&metrics, 0, sizeof(EngineMetrics));
    Move best = lattice_search(&board, dawg, &ai_rack, &bag, &cfg, &all_moves, &metrics);

    printf("  Graph search completed in %.2f ms!\n", metrics.deliberation_ms);
    printf("  Best move found: %s at (%d, %d) dir=%s score=%d\n",
           best.word, best.row, best.col, best.dir == DIR_HORIZONTAL ? "H" : "V", best.score);
    printf("  Root Court Confidence: %.1f%%\n", metrics.court_confidence * 100.0f);
    printf("  Lattice Nodes: %u\n", metrics.nodes_in_lattice);

    /* Must play out all tiles (outplay proof) and achieve 100% confidence */
    assert(best.num_placed == 2);
    assert(metrics.court_confidence >= 0.98f);

    move_list_destroy(&all_moves);
    dawg_free(dawg);
    printf("[TEST] Pure Graph-Based Proof Propagation test PASSED flawlessly!\n");
    return 0;
}
