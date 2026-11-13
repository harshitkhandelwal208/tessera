#include "quackle_bridge.h"
#include "types.h"
#include "dict.h"
#include "board.h"
#include "movegen.h"
#include "eval.h"
#include "simulation.h"
#include "bayes_rack.h"
#include <stdlib.h>
#include <string.h>

void* tessera_create_engine(const char *dawg_path) {
    Dawg *dawg = dawg_load_binary(dawg_path);
    return (void*)dawg;
}

void tessera_destroy_engine(void *engine) {
    if (engine) {
        dawg_free((Dawg*)engine);
    }
}

void* tessera_create_tracker(void) {
    BayesRackTracker *tracker = (BayesRackTracker*)malloc(sizeof(BayesRackTracker));
    if (tracker) {
        bayes_rack_init(tracker);
    }
    return (void*)tracker;
}

void tessera_destroy_tracker(void *tracker) {
    if (tracker) {
        free(tracker);
    }
}

BridgeMove tessera_compute_move(void *engine,
                                const char board_tiles[15][15],
                                const bool board_is_blank[15][15],
                                const char *rack_tiles,
                                int bag_count,
                                int score_diff,
                                int delib_steps,
                                void *tracker) {
    Dawg *dawg = (Dawg*)engine;
    BridgeMove bm;
    memset(&bm, 0, sizeof(bm));

    if (!dawg) {
        bm.type = 2; /* Pass */
        return bm;
    }

    /* 1. Build Tessera Board */
    Board tb;
    board_init(&tb);
    bool has_any = false;
    for (int r = 0; r < 15; r++) {
        for (int c = 0; c < 15; c++) {
            if (board_tiles[r][c] != 0) {
                has_any = true;
                tb.cells[r][c].tile = board_tiles[r][c];
                tb.cells[r][c].is_blank = board_is_blank[r][c];
            }
        }
    }
    tb.is_empty = !has_any;

    /* 2. Build Tessera Rack */
    Rack tr;
    rack_init(&tr);
    if (rack_tiles) {
        for (size_t i = 0; rack_tiles[i] != '\0'; i++) {
            rack_add_tile(&tr, rack_tiles[i]);
        }
    }
    rack_sort(&tr);

    /* 3. Configure Engine */
    EngineConfig cfg;
    engine_config_default(&cfg);
    cfg.max_deliberation_steps = delib_steps;
    cfg.persistent_tracker = tracker;
    cfg.score_diff = score_diff;

    TileBag bag_stub;
    bag_stub.count = bag_count;

    /* 4. Run Belief Lattice Deliberation */
    Move best = lattice_search(&tb, dawg, &tr, &bag_stub, &cfg, NULL, NULL);

    /* 5. Convert to BridgeMove */
    if (best.type == MOVE_PASS) {
        bm.type = 2;
    } else if (best.type == MOVE_EXCHANGE) {
        bm.type = 1;
        bm.num_placed = best.num_placed;
        for (int i = 0; i < best.num_placed; i++) {
            bm.placed[i].letter = best.placed[i].letter;
            bm.placed[i].is_blank = best.placed[i].is_blank;
        }
    } else {
        bm.type = 0; /* Play */
        bm.row = best.row;
        bm.col = best.col;
        bm.horizontal = (best.dir == DIR_HORIZONTAL);
        strncpy(bm.word, best.word, sizeof(bm.word) - 1);
        bm.num_placed = best.num_placed;
        bm.score = best.score;
        for (int i = 0; i < best.num_placed; i++) {
            bm.placed[i].row = best.placed[i].row;
            bm.placed[i].col = best.placed[i].col;
            bm.placed[i].letter = best.placed[i].letter;
            bm.placed[i].is_blank = best.placed[i].is_blank;
        }
    }

    return bm;
}
