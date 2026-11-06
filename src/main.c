#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dict.h"
#include "board.h"
#include "movegen.h"
#include "eval.h"
#include "simulation.h"
#include "protocol.h"
#include "nnue.h"

#define DEFAULT_DAWG_BIN "data/dawg.bin"
#define DEFAULT_DICT_TXT "data/dictionary.txt"

static Dawg* get_or_load_dawg(void) {
    Dawg *dawg = dawg_load_binary(DEFAULT_DAWG_BIN);
    if (!dawg) {
        printf("[INFO] Precompiled DAWG not found. Compiling %s...\n", DEFAULT_DICT_TXT);
        dawg = dawg_create_from_text(DEFAULT_DICT_TXT);
        if (!dawg) {
            fprintf(stderr, "Error: Could not load or compile dictionary from %s\n", DEFAULT_DICT_TXT);
            exit(1);
        }
        dawg_save_binary(dawg, DEFAULT_DAWG_BIN);
        printf("[INFO] Compiled and saved DAWG to %s (%u edges)\n", DEFAULT_DAWG_BIN, dawg->num_edges);
    }
    return dawg;
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    if (argc >= 2 && strcmp(argv[1], "build-dict") == 0) {
        const char *txt = (argc >= 3) ? argv[2] : DEFAULT_DICT_TXT;
        const char *bin = (argc >= 4) ? argv[3] : DEFAULT_DAWG_BIN;
        printf("Building DAWG from %s -> %s...\n", txt, bin);
        Dawg *d = dawg_create_from_text(txt);
        if (!d) {
            fprintf(stderr, "Failed to build DAWG\n");
            return 1;
        }
        dawg_save_binary(d, bin);
        printf("Successfully built DAWG with %u words and %u edges.\n", d->word_count, d->num_edges);
        dawg_free(d);
        return 0;
    }

    Dawg *dawg = get_or_load_dawg();

    if (nnue_load(&g_nnue, "data/nnue.bin")) {
        fprintf(stderr, "[INFO] Deep NNUE Evaluator loaded successfully (448->128->32->1, 246 KB)\n");
    } else {
        fprintf(stderr, "[INFO] NNUE model data/nnue.bin not found. Using hand-tuned Grandmaster Evaluator.\n");
    }

    if (argc >= 2 && (strcmp(argv[1], "server") == 0 || strcmp(argv[1], "protocol") == 0)) {
        protocol_run_server_loop(dawg);
    } else if (argc >= 2 && strcmp(argv[1], "ai-vs-ai") == 0) {
        protocol_run_terminal_game(dawg, true);
    } else if (argc >= 2 && strcmp(argv[1], "bench") == 0) {
        protocol_run_benchmark(dawg);
    } else if (argc >= 2 && strcmp(argv[1], "play") == 0) {
        protocol_run_terminal_game(dawg, false);
    } else {
        printf("TESSERA — Grandmaster Scrabble Engine (Native C)\n\n");
        printf("Usage:\n");
        printf("  %s play            Play Human vs AI in terminal\n", argv[0]);
        printf("  %s ai-vs-ai        Watch AI vs AI game simulation\n", argv[0]);
        printf("  %s server          Run JSON protocol server over stdin/stdout for GUI\n", argv[0]);
        printf("  %s bench           Run engine speed benchmarks\n", argv[0]);
        printf("  %s build-dict      Rebuild binary DAWG from text dictionary\n", argv[0]);
        printf("\nStarting interactive game by default...\n\n");
        protocol_run_terminal_game(dawg, false);
    }

    dawg_free(dawg);
    return 0;
}
