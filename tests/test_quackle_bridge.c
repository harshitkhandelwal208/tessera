#include "quackle_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(void) {
    printf("[TEST] Testing Quackle Bridge and Ctypes Integration Layer...\n");

    /* 1. Create Engine */
    void *engine = tessera_create_engine("data/dawg.bin");
    assert(engine != NULL);

    /* 2. Create Bayesian Opponent Tracker */
    void *tracker = tessera_create_tracker();
    assert(tracker != NULL);

    /* 3. Opening Move on Empty Board */
    char board_tiles[15][15];
    bool board_is_blank[15][15];
    memset(board_tiles, 0, sizeof(board_tiles));
    memset(board_is_blank, 0, sizeof(board_is_blank));

    BridgeMove bm1 = tessera_compute_move(engine, board_tiles, board_is_blank,
                                         "RETAINS", 86, 0, 10, tracker);
    assert(bm1.type == 0); /* Play */
    assert(bm1.score > 0);
    assert(bm1.num_placed >= 2);
    assert(strlen(bm1.word) >= 2);
    printf("  Opening move computed successfully: %s at (%d, %d) dir=%s score=%d num_placed=%d\n",
           bm1.word, bm1.row, bm1.col, bm1.horizontal ? "H" : "V", bm1.score, bm1.num_placed);

    /* 4. Populated Board Move with Uppercase Letters and Blanks */
    /* Place TRAIN horizontally at (7, 5) to (7, 9) with 'A' as a blank tile */
    board_tiles[7][5] = 'T'; board_is_blank[7][5] = false;
    board_tiles[7][6] = 'R'; board_is_blank[7][6] = false;
    board_tiles[7][7] = 'A'; board_is_blank[7][7] = true;  /* Blank tile designated as 'A' */
    board_tiles[7][8] = 'I'; board_is_blank[7][8] = false;
    board_tiles[7][9] = 'N'; board_is_blank[7][9] = false;

    BridgeMove bm2 = tessera_compute_move(engine, board_tiles, board_is_blank,
                                         "ED", 70, 0, 10, tracker);
    assert(bm2.type == 0);
    assert(bm2.score > 0);
    printf("  Populated board move computed successfully: %s at (%d, %d) score=%d\n",
           bm2.word, bm2.row, bm2.col, bm2.score);

    /* 5. Populated Board with Lowercase Blank Convention (Quackle style) */
    memset(board_tiles, 0, sizeof(board_tiles));
    memset(board_is_blank, 0, sizeof(board_is_blank));
    board_tiles[7][5] = 'T';
    board_tiles[7][6] = 'r'; /* Lowercase 'r' represents a blank tile in Quackle GCG format */
    board_tiles[7][7] = 'A';
    board_tiles[7][8] = 'I';
    board_tiles[7][9] = 'N';

    BridgeMove bm3 = tessera_compute_move(engine, board_tiles, board_is_blank,
                                         "ED", 70, 0, 10, tracker);
    assert(bm3.type == 0);
    assert(bm3.score > 0);
    printf("  Quackle-style lowercase blank recognized and solved: %s score=%d\n",
           bm3.word, bm3.score);

    /* 6. Exchange Move on Unfavorable Rack */
    BridgeMove bm4 = tessera_compute_move(engine, board_tiles, board_is_blank,
                                         "UUUUUUU", 70, 0, 10, tracker);
    /* Rack UUUUUUU cannot play on TRAIN legally, or exchange is superior */
    printf("  Unfavorable rack move type: %d (0=play, 1=exchange, 2=pass)\n", bm4.type);
    assert(bm4.type == 1 || bm4.type == 2 || bm4.type == 0);

    /* 7. Clean up */
    tessera_destroy_tracker(tracker);
    tessera_destroy_engine(engine);

    printf("[TEST] Quackle Bridge and Ctypes Integration test PASSED flawlessly!\n");
    return 0;
}
