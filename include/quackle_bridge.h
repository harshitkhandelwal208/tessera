#ifndef QUACKLE_BRIDGE_H
#define QUACKLE_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int type;           /* 0 = Play, 1 = Exchange, 2 = Pass */
    int row;            /* 0-14 */
    int col;            /* 0-14 */
    bool horizontal;    /* true = across, false = down */
    char word[16];      /* Full word formed */
    int num_placed;
    struct {
        int row;
        int col;
        char letter;
        bool is_blank;
    } placed[7];
    int score;
} BridgeMove;

void* tessera_create_engine(const char *dawg_path);
void  tessera_destroy_engine(void *engine);

void* tessera_create_tracker(void);
void  tessera_destroy_tracker(void *tracker);

BridgeMove tessera_compute_move(void *engine,
                                const char board_tiles[15][15],
                                const bool board_is_blank[15][15],
                                const char *rack_tiles,
                                int bag_count,
                                int score_diff,
                                int delib_steps,
                                void *tracker);

#ifdef __cplusplus
}
#endif

#endif /* QUACKLE_BRIDGE_H */
