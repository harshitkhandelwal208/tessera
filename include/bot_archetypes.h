#ifndef BOT_ARCHETYPES_H
#define BOT_ARCHETYPES_H

#include "types.h"
#include "board.h"
#include "dict.h"

typedef enum {
    BOT_BASELINE_GREEDY   = 0, /* Greedy 1-ply: Immediate word score (~1550 Elo) */
    BOT_DEFENSIVE_TURTLER = 1, /* Defensive Turtler: Avoids opening TW/DW (~1650 Elo) */
    BOT_BINGO_HUNTER      = 2, /* Bingo Hunter: 7-letter stem optimizer (~1750 Elo) */
    BOT_HEURISTIC_STATIC  = 3, /* Grandmaster 8-Phase Leave Heuristic (~1850 Elo) */
    BOT_NEURAL_STATIC     = 4, /* Raw Deep NNUE 1-Ply Evaluator (~1950 Elo) */
    BOT_CHAMPIONSHIP_SIM  = 5, /* Quackle-Style Lookahead Simulation (~2100 Elo) */
    BOT_MAGPIE_2PLY       = 6, /* MAGPIE/Woogles 2-Ply Realization Simulation (~2150 Elo) */
    BOT_TYPE_COUNT        = 7
} BotType;

const char *bot_archetype_name(BotType type);
const char *bot_archetype_desc(BotType type);
int bot_archetype_elo(BotType type);
Move bot_archetype_select_move(BotType type, const Board *board, const Dawg *dawg, const Rack *rack, int bag_count);

#endif /* BOT_ARCHETYPES_H */
