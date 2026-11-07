#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define BOARD_SIZE 15
#define TOTAL_SQUARES (BOARD_SIZE * BOARD_SIZE)
#define RACK_CAPACITY 7
#define MAX_WORD_LEN BOARD_SIZE
#define MAX_MOVES 8192
#define INITIAL_MOVE_CAPACITY 8192
#define NUM_LETTERS 26

/* Multiplier types */
typedef enum {
    MULT_NONE = 0,
    MULT_DL   = 1, /* Double Letter */
    MULT_TL   = 2, /* Triple Letter */
    MULT_DW   = 3, /* Double Word */
    MULT_TW   = 4  /* Triple Word */
} MultiplierType;

/* Direction */
typedef enum {
    DIR_HORIZONTAL = 0,
    DIR_VERTICAL   = 1
} Direction;

/* Move type */
typedef enum {
    MOVE_PLAY     = 0,
    MOVE_EXCHANGE = 1,
    MOVE_PASS     = 2
} MoveType;

/* Cell on board */
typedef struct {
    char tile;          /* 'A'-'Z' or 0 if empty */
    bool is_blank;      /* True if played as blank (0 pts) */
    MultiplierType mult;/* Premium square type */
} BoardCell;

/* Rack of a player */
typedef struct {
    char tiles[RACK_CAPACITY + 1];
    int count;
} Rack;

/* Single placed tile in a move */
typedef struct {
    int row;
    int col;
    char letter;        /* The letter displayed ('A'-'Z') */
    bool is_blank;      /* True if played using a blank tile */
} PlacedTile;

/* Move description */
typedef struct {
    MoveType type;
    Direction dir;
    int row;            /* Start row (0-14) */
    int col;            /* Start col (0-14) */
    char word[MAX_WORD_LEN + 1]; /* Full word formed in primary direction */
    PlacedTile placed[RACK_CAPACITY];
    int num_placed;
    int score;          /* Points earned on this turn */
    char leave[RACK_CAPACITY + 1]; /* Remaining tiles in rack */
    float leave_equity; /* Heuristic leave value */
    float sim_equity;   /* Monte Carlo simulated net equity */
} Move;

/* Dynamic move list (moves are never capped; dynamically resized if needed) */
typedef struct {
    Move *moves;
    int count;
    int capacity;
} MoveList;

/* Bag of tiles */
typedef struct {
    char tiles[100];
    int count;
} TileBag;

/* Full Game State */
typedef struct {
    BoardCell board[BOARD_SIZE][BOARD_SIZE];
    Rack player_rack;
    Rack ai_rack;
    TileBag bag;
    int player_score;
    int ai_score;
    int turn;           /* 0 for player, 1 for AI */
    int pass_count;     /* Consecutive passes */
    bool game_over;
} GameState;

/* Standard Scrabble Letter Point Values (A-Z, Blank=0) */
static const int TILE_VALUES[27] = {
    /* A */ 1, /* B */ 3, /* C */ 3, /* D */ 2, /* E */ 1,
    /* F */ 4, /* G */ 2, /* H */ 4, /* I */ 1, /* J */ 8,
    /* K */ 5, /* L */ 1, /* M */ 3, /* N */ 1, /* O */ 1,
    /* P */ 3, /* Q */ 10, /* R */ 1, /* S */ 1, /* T */ 1,
    /* U */ 1, /* V */ 4, /* W */ 4, /* X */ 8, /* Y */ 4,
    /* Z */ 10,
    /* Blank */ 0
};

/* Standard English Tile Counts */
static const int TILE_BAG_DISTRIBUTION[27] = {
    /* A */ 9,  /* B */ 2,  /* C */ 2,  /* D */ 4,  /* E */ 12,
    /* F */ 2,  /* G */ 3,  /* H */ 2,  /* I */ 9,  /* J */ 1,
    /* K */ 1,  /* L */ 4,  /* M */ 2,  /* N */ 6,  /* O */ 8,
    /* P */ 2,  /* Q */ 1,  /* R */ 6,  /* S */ 4,  /* T */ 6,
    /* U */ 4,  /* V */ 2,  /* W */ 2,  /* X */ 1,  /* Y */ 2,
    /* Z */ 1,
    /* Blank */ 2
};

#endif /* TYPES_H */
