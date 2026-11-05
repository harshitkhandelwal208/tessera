#include "dict.h"
#include <stdio.h>
#include <assert.h>
#include <time.h>

int main(void) {
    printf("[TEST] Building DAWG from data/dictionary.txt...\n");
    clock_t start = clock();
    Dawg *dawg = dawg_create_from_text("data/dictionary.txt");
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    assert(dawg != NULL);
    printf("[TEST] DAWG built successfully in %.3f seconds!\n", elapsed);
    printf("[TEST] Words: %u, Total Edges: %u (%.2f MB)\n",
           dawg->word_count, dawg->num_edges,
           (double)(dawg->num_edges * sizeof(DawgEdge)) / (1024.0 * 1024.0));

    /* Verify essential Scrabble words */
    assert(dawg_contains(dawg, "AA"));
    assert(dawg_contains(dawg, "QI"));
    assert(dawg_contains(dawg, "ZA"));
    assert(dawg_contains(dawg, "CAT"));
    assert(dawg_contains(dawg, "SCRABBLE"));
    assert(dawg_contains(dawg, "QUIZZES"));
    assert(dawg_contains(dawg, "OXYPHENBUTAZONE"));

    /* Verify non-words */
    assert(!dawg_contains(dawg, "XYZ"));
    assert(!dawg_contains(dawg, "ABC"));
    assert(!dawg_contains(dawg, "ZZZZ"));
    assert(!dawg_contains(dawg, "NOTAWALLWORD"));

    printf("[TEST] Saving binary DAWG to data/dawg.bin...\n");
    bool saved = dawg_save_binary(dawg, "data/dawg.bin");
    assert(saved);
    dawg_free(dawg);

    printf("[TEST] Loading binary DAWG from data/dawg.bin...\n");
    start = clock();
    Dawg *loaded = dawg_load_binary("data/dawg.bin");
    end = clock();
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    assert(loaded != NULL);
    printf("[TEST] Binary DAWG loaded in %.4f seconds!\n", elapsed);

    assert(dawg_contains(loaded, "AA"));
    assert(dawg_contains(loaded, "QI"));
    assert(dawg_contains(loaded, "ZA"));
    assert(dawg_contains(loaded, "SCRABBLE"));
    assert(!dawg_contains(loaded, "XYZZYPLUGH"));

    dawg_free(loaded);
    printf("[TEST] DAWG test PASSED flawlessly!\n");
    return 0;
}
