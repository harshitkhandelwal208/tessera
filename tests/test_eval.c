#include "dict.h"
#include "board.h"
#include "eval.h"
#include "simulation.h"
#include <stdio.h>
#include <assert.h>
#include <time.h>

int main(void) {
    printf("[TEST] Testing rack leave evaluations...\n");
    float eq_blank = eval_rack_leave("?");
    float eq_s = eval_rack_leave("S");
    float eq_vowels = eval_rack_leave("AAAA");
    float eq_ing = eval_rack_leave("ING");
    float eq_vv = eval_rack_leave("VV");

    printf("  Leave '?':    %.2f pts\n", eq_blank);
    printf("  Leave 'S':    %.2f pts\n", eq_s);
    printf("  Leave 'ING':  %.2f pts\n", eq_ing);
    printf("  Leave 'AAAA': %.2f pts\n", eq_vowels);
    printf("  Leave 'VV':   %.2f pts\n", eq_vv);

    assert(eq_blank > 20.0f);
    assert(eq_s > 5.0f);
    assert(eq_vowels < -10.0f);
    assert(eq_vv < -10.0f);
    assert(eq_ing > 3.0f);

    printf("[TEST] Loading DAWG and initializing game...\n");
    Dawg *dawg = dawg_load_binary("data/dawg.bin");
    assert(dawg != NULL);

    Board board;
    board_init(&board);

    TileBag bag;
    bag_init(&bag);
    bag_shuffle(&bag);

    Rack ai_rack;
    rack_init(&ai_rack);
    bag_draw_tiles(&bag, &ai_rack, RACK_CAPACITY);

    char rack_str[RACK_CAPACITY + 1];
    rack_to_string(&ai_rack, rack_str);
    printf("[TEST] AI Rack drawn: %s (Bag remaining: %d)\n", rack_str, bag.count);

    EngineConfig config;
    engine_config_default(&config);
    config.max_deliberation_steps = 25;

    MoveList moves;
    EngineMetrics metrics;
    clock_t start = clock();
    Move best = lattice_search(&board, dawg, &ai_rack, &bag, &config, &moves, &metrics);
    clock_t end = clock();
    double elapsed_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    printf("[TEST] Found best move in %.2f ms!\n", elapsed_ms);
    printf("  Word: %s\n", best.word);
    printf("  Location: (%d, %d), Dir: %s\n", best.row, best.col,
           best.dir == DIR_HORIZONTAL ? "H" : "V");
    printf("  Score: %d pts\n", best.score);
    printf("  Leave: '%s' (Equity: %.2f, Net Sim Equity: %.2f)\n",
           best.leave, best.leave_equity, best.sim_equity);

    assert(best.score > 0 || best.type == MOVE_EXCHANGE);

    dawg_free(dawg);
    printf("[TEST] Evaluation & simulation tests PASSED flawlessly!\n");
    return 0;
}
