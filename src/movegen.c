#include "movegen.h"
#include "eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdalign.h>

#define MOVE_ARENA_MAX_DEPTH 8
static _Thread_local alignas(64) Move s_move_arena[MOVE_ARENA_MAX_DEPTH][INITIAL_MOVE_CAPACITY];
static _Thread_local int s_move_arena_in_use[MOVE_ARENA_MAX_DEPTH] = {0};

void move_list_init(MoveList *list) {
    if (!list) return;
    list->count = 0;
    if (list->moves != NULL && list->capacity > 0) {
        return; /* Already initialized with valid arena or heap buffer */
    }
    list->arena_slot = 0;
    /* Attempt to acquire a 64-byte cacheline-aligned thread-local arena slot */
    for (int i = 0; i < MOVE_ARENA_MAX_DEPTH; i++) {
        if (!s_move_arena_in_use[i]) {
            s_move_arena_in_use[i] = 1;
            list->moves = s_move_arena[i];
            list->capacity = INITIAL_MOVE_CAPACITY;
            list->arena_slot = i + 1;
            return;
        }
    }
    /* Fallback if all arena slots in use (rare deep recursion) */
    list->capacity = INITIAL_MOVE_CAPACITY;
    list->moves = (Move*)malloc(sizeof(Move) * list->capacity);
    list->arena_slot = 0;
}

void move_list_destroy(MoveList *list) {
    if (!list) return;
    if (list->arena_slot > 0 && list->arena_slot <= MOVE_ARENA_MAX_DEPTH) {
        s_move_arena_in_use[list->arena_slot - 1] = 0;
        list->moves = NULL;
    } else if (list->moves) {
        free(list->moves);
        list->moves = NULL;
    }
    list->count = 0;
    list->capacity = 0;
    list->arena_slot = 0;
}

void board_transpose(Board *dst, const Board *src) {
    dst->is_empty = src->is_empty;
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            dst->cells[r][c] = src->cells[c][r];
        }
    }
    board_recompute_bitboards(dst);
}

typedef struct {
    const Board *board;
    const Board *orig_board;
    const Dawg *dawg;
    const Rack *orig_rack;
    Rack cur_rack;
    MoveList *list;
    bool is_transposed;
    CrossCheck cross_checks[BOARD_SIZE][BOARD_SIZE];
    int best_score;
    float best_equity;
    int bag_count;
    bool track_equity;
} GenContext;

static void compute_leave(const Rack *orig_rack, const PlacedTile *placed, int num_placed, char *leave) {
    char rack_copy[RACK_CAPACITY + 1];
    int count = orig_rack->count;
    memcpy(rack_copy, orig_rack->tiles, count);

    for (int p = 0; p < num_placed; p++) {
        char needed = placed[p].is_blank ? '?' : placed[p].letter;
        for (int i = 0; i < count; i++) {
            if (rack_copy[i] == needed) {
                for (int j = i; j < count - 1; j++) {
                    rack_copy[j] = rack_copy[j + 1];
                }
                count--;
                break;
            }
        }
    }

    /* Sort leave */
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (rack_copy[i] > rack_copy[j]) {
                char tmp = rack_copy[i];
                rack_copy[i] = rack_copy[j];
                rack_copy[j] = tmp;
            }
        }
    }
    memcpy(leave, rack_copy, count);
    leave[count] = '\0';
}

static inline void record_move(GenContext *ctx, int row, int start_col, const char *word,
                               const PlacedTile *placed, int num_placed) {
    if (ctx->list == NULL) {
        /* Fast mode: directly compute score in registers without allocating Move or computing leaves */
        Move m;
        m.type = MOVE_PLAY;
        m.num_placed = num_placed;
        strcpy(m.word, word);
        if (!ctx->is_transposed) {
            m.dir = DIR_HORIZONTAL;
            m.row = row;
            m.col = start_col;
            for (int i = 0; i < num_placed; i++) m.placed[i] = placed[i];
        } else {
            m.dir = DIR_VERTICAL;
            m.row = start_col;
            m.col = row;
            for (int i = 0; i < num_placed; i++) {
                m.placed[i].row = placed[i].col;
                m.placed[i].col = placed[i].row;
                m.placed[i].letter = placed[i].letter;
                m.placed[i].is_blank = placed[i].is_blank;
            }
        }
        int score = board_score_move(ctx->orig_board, ctx->dawg, &m);
        if (score > ctx->best_score) ctx->best_score = score;
        if (ctx->track_equity) {
            char leave[RACK_CAPACITY + 1];
            int lcnt = ctx->cur_rack.count;
            if (lcnt > RACK_CAPACITY) lcnt = RACK_CAPACITY;
            memcpy(leave, ctx->cur_rack.tiles, lcnt);
            leave[lcnt] = '\0';
            PhaseBucket phase = eval_classify_phase(ctx->bag_count);
            float leave_eq = eval_rack_leave_bucketed(leave, phase);
            float total_eq = (float)score + (0.85f * leave_eq);
            if (total_eq > ctx->best_equity) ctx->best_equity = total_eq;
        }
        return;
    }

    /* Dynamically grow move list when needed so moves are never capped */
    if (ctx->list->count >= ctx->list->capacity) {
        int new_cap = ctx->list->capacity > 0 ? ctx->list->capacity * 2 : INITIAL_MOVE_CAPACITY;
        Move *new_moves = (Move*)malloc(sizeof(Move) * new_cap);
        if (!new_moves) return;
        memcpy(new_moves, ctx->list->moves, sizeof(Move) * ctx->list->count);
        if (ctx->list->arena_slot > 0 && ctx->list->arena_slot <= MOVE_ARENA_MAX_DEPTH) {
            s_move_arena_in_use[ctx->list->arena_slot - 1] = 0;
            ctx->list->arena_slot = 0;
        } else if (ctx->list->moves) {
            free(ctx->list->moves);
        }
        ctx->list->moves = new_moves;
        ctx->list->capacity = new_cap;
    }

    Move *m = &ctx->list->moves[ctx->list->count];
    m->type = MOVE_PLAY;
    m->num_placed = num_placed;
    strcpy(m->word, word);

    if (!ctx->is_transposed) {
        m->dir = DIR_HORIZONTAL;
        m->row = row;
        m->col = start_col;
        for (int i = 0; i < num_placed; i++) {
            m->placed[i] = placed[i];
        }
        m->score = board_score_move(ctx->orig_board, ctx->dawg, m);
    } else {
        m->dir = DIR_VERTICAL;
        m->row = start_col;
        m->col = row;
        for (int i = 0; i < num_placed; i++) {
            m->placed[i].row = placed[i].col;
            m->placed[i].col = placed[i].row;
            m->placed[i].letter = placed[i].letter;
            m->placed[i].is_blank = placed[i].is_blank;
        }
        m->score = board_score_move(ctx->orig_board, ctx->dawg, m);
    }

    compute_leave(ctx->orig_rack, m->placed, num_placed, m->leave);
    m->leave_equity = 0.0f;
    m->sim_equity = 0.0f;

    ctx->list->count++;
}

static void extend_right(GenContext *ctx, int row, int col, uint32_t node, DawgEdge leading_edge,
                         int start_col, int anchor_col,
                         char *word, int word_len,
                         PlacedTile *placed, int num_placed) {
    /* If beyond board boundary */
    if (col >= BOARD_SIZE) {
        if (col > anchor_col && dawg_is_terminal(leading_edge) && num_placed > 0) {
            word[word_len] = '\0';
            if (!ctx->board->is_empty || word_len >= 2) {
                record_move(ctx, row, start_col, word, placed, num_placed);
            }
        }
        return;
    }

    /* Check if current square is empty */
    if (ctx->board->cells[row][col].tile == 0) {
        /* Can we terminate a valid word here? */
        if (col > anchor_col && dawg_is_terminal(leading_edge) && num_placed > 0) {
            word[word_len] = '\0';
            if (!ctx->board->is_empty || word_len >= 2) {
                record_move(ctx, row, start_col, word, placed, num_placed);
            }
        }

        if (ctx->cur_rack.count == 0) return;
        if (node == 0) return;

        uint32_t mask = ctx->cross_checks[row][col].valid_letters_mask;
        if (mask == 0) return;

        /* Iterate through available DAWG transitions */
        uint32_t edge_idx = node;
        while (1) {
            DawgEdge edge = ctx->dawg->edges[edge_idx];
            char let = dawg_letter(edge);
            int let_idx = let - 'A';

            if ((mask & (1U << let_idx)) != 0) {
                /* Check regular letter in rack */
                if (rack_remove_tile(&ctx->cur_rack, let)) {
                    placed[num_placed].row = row;
                    placed[num_placed].col = col;
                    placed[num_placed].letter = let;
                    placed[num_placed].is_blank = false;

                    word[word_len] = let;
                    extend_right(ctx, row, col + 1, dawg_child_index(edge), edge,
                                 start_col, anchor_col, word, word_len + 1,
                                 placed, num_placed + 1);

                    rack_add_tile(&ctx->cur_rack, let);
                }

                /* Check blank '?' in rack */
                if (rack_remove_tile(&ctx->cur_rack, '?')) {
                    placed[num_placed].row = row;
                    placed[num_placed].col = col;
                    placed[num_placed].letter = let;
                    placed[num_placed].is_blank = true;

                    word[word_len] = let;
                    extend_right(ctx, row, col + 1, dawg_child_index(edge), edge,
                                 start_col, anchor_col, word, word_len + 1,
                                 placed, num_placed + 1);

                    rack_add_tile(&ctx->cur_rack, '?');
                }
            }

            if (dawg_is_last_child(edge)) break;
            edge_idx++;
        }
    } else {
        /* Square is occupied by existing tile on board */
        if (node == 0) return;

        char existing_tile = ctx->board->cells[row][col].tile;
        bool is_term = false;
        uint32_t next_node = dawg_step(ctx->dawg, node, existing_tile, &is_term);

        if (next_node != 0 || is_term) {
            /* Construct synthetic leading edge for next step */
            DawgEdge syn_edge = ((uint32_t)(existing_tile - 'A') << DAWG_LETTER_SHIFT) |
                                (next_node << DAWG_CHILD_SHIFT) |
                                (is_term ? DAWG_TERMINAL_MASK : 0);

            word[word_len] = existing_tile;
            extend_right(ctx, row, col + 1, next_node, syn_edge,
                         start_col, anchor_col, word, word_len + 1,
                         placed, num_placed);
        }
    }
}

static void left_part(GenContext *ctx, int row, int anchor_col, uint32_t node, int limit,
                      int target_prefix_len,
                      char *word, int word_len,
                      PlacedTile *placed, int num_placed) {
    if (word_len == target_prefix_len) {
        /* Reached desired prefix length, now extend right from anchor */
        word[word_len] = '\0';
        extend_right(ctx, row, anchor_col, node, 0,
                     anchor_col - target_prefix_len, anchor_col,
                     word, word_len, placed, num_placed);
        return;
    }

    if (node == 0 || ctx->cur_rack.count == 0) return;

    int cur_col = anchor_col - target_prefix_len + word_len;
    uint32_t mask = ctx->cross_checks[row][cur_col].valid_letters_mask;
    if (mask == 0) return;

    /* Iterate through rack tiles uniquely to prevent duplicate branching */
    bool tried[27] = {false};
    int rcount = ctx->cur_rack.count;
    char tiles_copy[RACK_CAPACITY];
    memcpy(tiles_copy, ctx->cur_rack.tiles, rcount);

    for (int i = 0; i < rcount; i++) {
        char t = tiles_copy[i];
        if (t == '?') {
            if (tried[26]) continue;
            tried[26] = true;

            /* Try blank as all letters that have edges */
            uint32_t edge_idx = node;
            while (1) {
                DawgEdge edge = ctx->dawg->edges[edge_idx];
                char let = dawg_letter(edge);
                int let_idx = let - 'A';

                if ((mask & (1U << let_idx)) != 0) {
                    rack_remove_tile(&ctx->cur_rack, '?');
                    placed[num_placed].row = row;
                    placed[num_placed].col = cur_col;
                    placed[num_placed].letter = let;
                    placed[num_placed].is_blank = true;

                    word[word_len] = let;
                    left_part(ctx, row, anchor_col, dawg_child_index(edge), limit,
                              target_prefix_len, word, word_len + 1,
                              placed, num_placed + 1);

                    rack_add_tile(&ctx->cur_rack, '?');
                }
                if (dawg_is_last_child(edge)) break;
                edge_idx++;
            }
        } else {
            int t_idx = t - 'A';
            if (tried[t_idx]) continue;
            tried[t_idx] = true;

            if ((mask & (1U << t_idx)) == 0) continue;

            bool is_term = false;
            uint32_t next_node = dawg_step(ctx->dawg, node, t, &is_term);
            if (next_node != 0 || is_term) {
                rack_remove_tile(&ctx->cur_rack, t);
                placed[num_placed].row = row;
                placed[num_placed].col = cur_col;
                placed[num_placed].letter = t;
                placed[num_placed].is_blank = false;

                word[word_len] = t;
                left_part(ctx, row, anchor_col, next_node, limit,
                          target_prefix_len, word, word_len + 1,
                          placed, num_placed + 1);

                rack_add_tile(&ctx->cur_rack, t);
            }
        }
    }
}

static void generate_row_moves(GenContext *ctx) {
    board_compute_horizontal_cross_checks(ctx->board, ctx->dawg, ctx->cross_checks);

    for (int w = 0; w < 4; w++) {
        uint64_t mask = ctx->board->anchors.b[w];
        while (mask) {
            int bit = __builtin_ctzll(mask);
            mask &= mask - 1;
            int sq = (w << 6) + bit;
            if (sq >= TOTAL_SQUARES) break;
            int r = sq / BOARD_SIZE;
            int c = sq % BOARD_SIZE;

            /* Check if square to the left is occupied */
            if (c > 0 && ctx->board->cells[r][c - 1].tile != 0) {
                /* Prefix is fixed by existing board tiles */
                int start_c = c - 1;
                while (start_c > 0 && ctx->board->cells[r][start_c - 1].tile != 0) {
                    start_c--;
                }

                /* Step DAWG through this prefix */
                uint32_t node = ctx->dawg->root_index;
                char word[MAX_WORD_LEN + 1];
                int word_len = 0;
                bool valid_prefix = true;

                for (int k = start_c; k < c; k++) {
                    char t = ctx->board->cells[r][k].tile;
                    word[word_len++] = t;
                    bool is_term = false;
                    node = dawg_step(ctx->dawg, node, t, &is_term);
                    if (node == 0 && !is_term) {
                        valid_prefix = false;
                        break;
                    }
                }

                if (valid_prefix && node != 0) {
                    word[word_len] = '\0';
                    PlacedTile placed[RACK_CAPACITY];
                    extend_right(ctx, r, c, node, 0, start_c, c,
                                 word, word_len, placed, 0);
                }
            } else {
                /* Square to left is empty; generate left prefixes of various lengths */
                int limit = 0;
                int k = c - 1;
                while (k >= 0 && ctx->board->cells[r][k].tile == 0 &&
                       !board_is_anchor(ctx->board, r, k) && limit < RACK_CAPACITY) {
                    limit++;
                    k--;
                }

                for (int prefix_len = 0; prefix_len <= limit; prefix_len++) {
                    char word[MAX_WORD_LEN + 1];
                    PlacedTile placed[RACK_CAPACITY];
                    left_part(ctx, r, c, ctx->dawg->root_index, limit, prefix_len,
                              word, 0, placed, 0);
                }
            }
        }
    }
}

int movegen_generate_all(const Board *board, const Dawg *dawg, const Rack *rack, MoveList *list) {
    if (!list) return 0;
    if (!list->moves || list->capacity == 0) {
        move_list_init(list);
    } else {
        list->count = 0;
    }

    GenContext ctx;
    ctx.orig_board = board;
    ctx.dawg = dawg;
    ctx.orig_rack = rack;
    ctx.list = list;

    /* 1. Generate Horizontal Moves */
    ctx.board = board;
    ctx.is_transposed = false;
    memcpy(&ctx.cur_rack, rack, sizeof(Rack));
    generate_row_moves(&ctx);

    /* 2. Generate Vertical Moves via Transposed Board */
    Board transposed;
    board_transpose(&transposed, board);
    ctx.board = &transposed;
    ctx.is_transposed = true;
    memcpy(&ctx.cur_rack, rack, sizeof(Rack));
    generate_row_moves(&ctx);

    return list->count;
}

int movegen_find_best_response_score(const Board *board, const Dawg *dawg, const Rack *rack) {
    GenContext ctx;
    ctx.orig_board = board;
    ctx.dawg = dawg;
    ctx.orig_rack = rack;
    ctx.list = NULL;
    ctx.best_score = 0;
    ctx.best_equity = 0.0f;
    ctx.bag_count = 50;
    ctx.track_equity = false;

    /* 1. Generate Horizontal Moves */
    ctx.board = board;
    ctx.is_transposed = false;
    memcpy(&ctx.cur_rack, rack, sizeof(Rack));
    generate_row_moves(&ctx);

    /* 2. Generate Vertical Moves via Transposed Board */
    Board transposed;
    board_transpose(&transposed, board);
    ctx.board = &transposed;
    ctx.is_transposed = true;
    memcpy(&ctx.cur_rack, rack, sizeof(Rack));
    generate_row_moves(&ctx);

    return ctx.best_score;
}

float movegen_find_best_response_equity(const Board *board, const Dawg *dawg, const Rack *rack, int bag_count) {
    GenContext ctx;
    ctx.orig_board = board;
    ctx.dawg = dawg;
    ctx.orig_rack = rack;
    ctx.list = NULL;
    ctx.best_score = 0;
    ctx.best_equity = -1000.0f;
    ctx.bag_count = bag_count;
    ctx.track_equity = true;

    /* 1. Generate Horizontal Moves */
    ctx.board = board;
    ctx.is_transposed = false;
    memcpy(&ctx.cur_rack, rack, sizeof(Rack));
    generate_row_moves(&ctx);

    /* 2. Generate Vertical Moves via Transposed Board */
    Board transposed;
    board_transpose(&transposed, board);
    ctx.board = &transposed;
    ctx.is_transposed = true;
    memcpy(&ctx.cur_rack, rack, sizeof(Rack));
    generate_row_moves(&ctx);

    if (ctx.best_equity < -500.0f) return 0.0f;
    return ctx.best_equity;
}
