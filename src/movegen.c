#include "movegen.h"
#include "eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdalign.h>
#include <assert.h>

#define MOVE_ARENA_MAX_DEPTH 8
static _Thread_local alignas(64) Move s_move_arena[MOVE_ARENA_MAX_DEPTH][INITIAL_MOVE_CAPACITY];
static _Thread_local int s_move_arena_in_use[MOVE_ARENA_MAX_DEPTH] = {0};

static const uint8_t s_letter_mults[5] = { 1, 2, 3, 1, 1 };
static const uint8_t s_word_mults[5]   = { 1, 1, 1, 2, 3 };

/*
 * Ultra-fast in-register move scorer.
 * Directly evaluates the score of a horizontal word on ctx->board using precomputed cross-checks.
 * Eliminates all coordinate conversions, direction branches, and board cell walks.
 */
static inline int score_horizontal_move(const Board *b, const CrossCheck (*cross_checks)[BOARD_SIZE],
                                        int row, int start_col, const char *word, int word_len,
                                        const PlacedTile *placed, int num_placed) {
    int main_word_sum = 0;
    int main_word_mult = 1;
    int total_score = 0;
    int placed_k = 0;

    for (int i = 0; i < word_len; i++) {
        int c = start_col + i;
        if (b->cells[row][c].tile != 0) {
            bool is_blank = b->cells[row][c].is_blank;
            main_word_sum += is_blank ? 0 : TILE_VALUES[b->cells[row][c].tile - 'A'];
        } else {
            bool is_blank = placed[placed_k].is_blank;
            int let_val = is_blank ? 0 : TILE_VALUES[word[i] - 'A'];
            MultiplierType m = b->cells[row][c].mult;
            int letter_mult = s_letter_mults[m];
            int word_mult = s_word_mults[m];
            main_word_mult *= word_mult;

            main_word_sum += let_val * letter_mult;

            if (cross_checks) {
                int let_idx = word[i] - 'A';
                int cscore = cross_checks[row][c].cross_scores[let_idx];
                if (cscore > 0) {
                    if (is_blank) {
                        cscore -= TILE_VALUES[let_idx] * letter_mult * word_mult;
                    }
                    total_score += cscore;
                }
            }
            placed_k++;
        }
    }

    total_score += main_word_sum * main_word_mult;
    if (num_placed == 7) {
        total_score += 50;
    }
    return total_score;
}

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
    if (src->is_empty) {
        board_init(dst);
        return;
    }
    dst->is_empty = false;
    memset(&dst->occupied, 0, sizeof(Bitboard256));
    memset(&dst->anchors, 0, sizeof(Bitboard256));
    memset(&dst->blanks, 0, sizeof(Bitboard256));

    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            dst->cells[r][c] = src->cells[c][r];
            if (dst->cells[r][c].tile != 0) {
                bb_set(&dst->occupied, r, c);
                if (dst->cells[r][c].is_blank) {
                    bb_set(&dst->blanks, r, c);
                }
            }
        }
    }
    /* Directly transpose anchor bitboard: anchor at (r, c) becomes (c, r) */
    for (int w = 0; w < 4; w++) {
        uint64_t mask = src->anchors.b[w];
        while (mask) {
            int bit = __builtin_ctzll(mask);
            mask &= mask - 1;
            int sq = (w << 6) + bit;
            if (sq >= TOTAL_SQUARES) break;
            int r = sq / BOARD_SIZE;
            int c = sq % BOARD_SIZE;
            bb_set(&dst->anchors, c, r);
        }
    }
}

typedef struct {
    const Board *board;
    const Board *orig_board;
    const Dawg *dawg;
    const Rack *orig_rack;
    MoveList *list;
    bool is_transposed;
    const CrossCheck (*cross_checks)[BOARD_SIZE];
    int rack_counts[27];
    uint32_t rack_mask;
    int tiles_left;
    int best_score;
    float best_equity;
    int bag_count;
    bool track_equity;
} GenContext;

static inline void init_context_rack(GenContext *ctx, const Rack *rack) {
    memset(ctx->rack_counts, 0, sizeof(ctx->rack_counts));
    ctx->rack_mask = 0;
    ctx->tiles_left = rack->count;
    for (int i = 0; i < rack->count; i++) {
        char t = rack->tiles[i];
        int idx = (t == '?') ? 26 : (t - 'A');
        if (idx >= 0 && idx < 26) {
            ctx->rack_counts[idx]++;
            ctx->rack_mask |= (1U << idx);
        } else if (idx == 26) {
            ctx->rack_counts[26]++;
        }
    }
}

static inline void compute_leave(const Rack *orig_rack, const PlacedTile *placed, int num_placed, char *leave) {
    if (num_placed >= orig_rack->count) {
        leave[0] = '\0';
        return;
    }
    int remove_counts[27] = {0};
    for (int p = 0; p < num_placed; p++) {
        char needed = placed[p].is_blank ? '?' : placed[p].letter;
        int idx = (needed == '?') ? 26 : (needed - 'A');
        remove_counts[idx]++;
    }
    int out_len = 0;
    for (int i = 0; i < orig_rack->count; i++) {
        char t = orig_rack->tiles[i];
        int idx = (t == '?') ? 26 : (t - 'A');
        if (remove_counts[idx] > 0) {
            remove_counts[idx]--;
        } else {
            leave[out_len++] = t;
        }
    }
    leave[out_len] = '\0';
}

static inline void record_move(GenContext *ctx, int row, int start_col, const char *word, int word_len,
                               const PlacedTile *placed, int num_placed) {
    if (ctx->list == NULL) {
        /* Fast mode: directly compute score in registers without allocating Move or computing leaves */
        int score = score_horizontal_move(ctx->board, ctx->cross_checks, row, start_col, word, word_len, placed, num_placed);
        if (score > ctx->best_score) ctx->best_score = score;
        if (ctx->track_equity) {
            char leave[RACK_CAPACITY + 1];
            int lcnt = 0;
            for (int l = 0; l < 26; l++) {
                for (int c = 0; c < ctx->rack_counts[l]; c++) {
                    leave[lcnt++] = (char)('A' + l);
                }
            }
            for (int c = 0; c < ctx->rack_counts[26]; c++) {
                leave[lcnt++] = '?';
            }
            leave[lcnt] = '\0';
            PhaseBucket phase = eval_classify_phase(ctx->bag_count);
            float leave_eq = eval_rack_leave_bucketed(leave, phase);
            float total_eq = (float)score + leave_eq;
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
    memcpy(m->word, word, word_len + 1);

    if (!ctx->is_transposed) {
        m->dir = DIR_HORIZONTAL;
        m->row = row;
        m->col = start_col;
        for (int i = 0; i < num_placed; i++) {
            m->placed[i] = placed[i];
        }
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
    }
    m->score = score_horizontal_move(ctx->board, ctx->cross_checks, row, start_col, word, word_len, placed, num_placed);

    int lcnt = 0;
    for (int l = 0; l < 26; l++) {
        for (int c = 0; c < ctx->rack_counts[l]; c++) {
            m->leave[lcnt++] = (char)('A' + l);
        }
    }
    for (int c = 0; c < ctx->rack_counts[26]; c++) {
        m->leave[lcnt++] = '?';
    }
    m->leave[lcnt] = '\0';
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
                record_move(ctx, row, start_col, word, word_len, placed, num_placed);
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
                record_move(ctx, row, start_col, word, word_len, placed, num_placed);
            }
        }

        if (ctx->tiles_left == 0) return;
        if (node == 0) return;

        uint32_t mask = ctx->cross_checks[row][col].valid_letters_mask;
        if (mask == 0) return;

        uint32_t avail_mask = (ctx->rack_counts[26] > 0) ? mask : (mask & ctx->rack_mask);
        if (avail_mask == 0) return;

        /* Iterate through available DAWG transitions */
        uint32_t edge_idx = node;
        while (1) {
            DawgEdge edge = ctx->dawg->edges[edge_idx];
            uint32_t let_idx = dawg_letter_idx(edge);

            if ((avail_mask & (1U << let_idx)) != 0) {
                char let = (char)('A' + let_idx);

                /* Check regular letter in rack */
                if (ctx->rack_counts[let_idx] > 0) {
                    ctx->rack_counts[let_idx]--;
                    if (ctx->rack_counts[let_idx] == 0) ctx->rack_mask &= ~(1U << let_idx);
                    ctx->tiles_left--;
                    placed[num_placed].row = row;
                    placed[num_placed].col = col;
                    placed[num_placed].letter = let;
                    placed[num_placed].is_blank = false;

                    word[word_len] = let;
                    extend_right(ctx, row, col + 1, dawg_child_index(edge), edge,
                                 start_col, anchor_col, word, word_len + 1,
                                 placed, num_placed + 1);

                    ctx->rack_counts[let_idx]++;
                    ctx->rack_mask |= (1U << let_idx);
                    ctx->tiles_left++;
                }

                /* Check blank '?' in rack */
                if (ctx->rack_counts[26] > 0) {
                    ctx->rack_counts[26]--;
                    ctx->tiles_left--;
                    placed[num_placed].row = row;
                    placed[num_placed].col = col;
                    placed[num_placed].letter = let;
                    placed[num_placed].is_blank = true;

                    word[word_len] = let;
                    extend_right(ctx, row, col + 1, dawg_child_index(edge), edge,
                                 start_col, anchor_col, word, word_len + 1,
                                 placed, num_placed + 1);

                    ctx->rack_counts[26]++;
                    ctx->tiles_left++;
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

    if (node == 0 || ctx->tiles_left == 0) return;

    int cur_col = anchor_col - target_prefix_len + word_len;
    uint32_t mask = ctx->cross_checks[row][cur_col].valid_letters_mask;
    if (mask == 0) return;

    uint32_t avail_mask = (ctx->rack_counts[26] > 0) ? mask : (mask & ctx->rack_mask);
    if (avail_mask == 0) return;

    /* Single-pass DAWG edge walk: iterate over sibling edges directly */
    uint32_t edge_idx = node;
    while (1) {
        DawgEdge edge = ctx->dawg->edges[edge_idx];
        uint32_t let_idx = dawg_letter_idx(edge);

        if ((avail_mask & (1U << let_idx)) != 0) {
            uint32_t next_node = dawg_child_index(edge);
            bool is_term = dawg_is_terminal(edge);

            if (next_node != 0 || is_term) {
                char let = (char)('A' + let_idx);

                /* 1. Try regular letter in rack */
                if (ctx->rack_counts[let_idx] > 0) {
                    ctx->rack_counts[let_idx]--;
                    if (ctx->rack_counts[let_idx] == 0) ctx->rack_mask &= ~(1U << let_idx);
                    ctx->tiles_left--;
                    placed[num_placed].row = row;
                    placed[num_placed].col = cur_col;
                    placed[num_placed].letter = let;
                    placed[num_placed].is_blank = false;

                    word[word_len] = let;
                    left_part(ctx, row, anchor_col, next_node, limit,
                              target_prefix_len, word, word_len + 1,
                              placed, num_placed + 1);

                    ctx->rack_counts[let_idx]++;
                    ctx->rack_mask |= (1U << let_idx);
                    ctx->tiles_left++;
                }

                /* 2. Try blank '?' in rack */
                if (ctx->rack_counts[26] > 0) {
                    ctx->rack_counts[26]--;
                    ctx->tiles_left--;
                    placed[num_placed].row = row;
                    placed[num_placed].col = cur_col;
                    placed[num_placed].letter = let;
                    placed[num_placed].is_blank = true;

                    word[word_len] = let;
                    left_part(ctx, row, anchor_col, next_node, limit,
                              target_prefix_len, word, word_len + 1,
                              placed, num_placed + 1);

                    ctx->rack_counts[26]++;
                    ctx->tiles_left++;
                }
            }
        }

        if (dawg_is_last_child(edge)) break;
        edge_idx++;
    }
}

static void generate_row_moves(GenContext *ctx) {
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

                char word[MAX_WORD_LEN + 1];
                PlacedTile placed[RACK_CAPACITY];
                word[0] = '\0';
                extend_right(ctx, r, c, ctx->dawg->root_index, 0, c, c,
                             word, 0, placed, 0);

                for (int prefix_len = 1; prefix_len <= limit; prefix_len++) {
                    left_part(ctx, r, c, ctx->dawg->root_index, limit, prefix_len,
                              word, 0, placed, 0);
                }
            }
        }
    }
}

void board_cross_cache_init(BoardCrossCache *cache, const Board *board, const Dawg *dawg) {
    if (!cache || !board || !dawg) return;
    board_compute_horizontal_cross_checks(board, dawg, cache->h_checks);
    board_transpose(&cache->transposed, board);
    board_compute_horizontal_cross_checks(&cache->transposed, dawg, cache->v_checks);
}

int movegen_generate_all_cached(const Board *board, const BoardCrossCache *cache,
                                const Dawg *dawg, const Rack *rack, MoveList *list) {
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
    ctx.best_score = 0;
    ctx.best_equity = 0.0f;
    ctx.bag_count = 50;
    ctx.track_equity = false;

    /* 1. Generate Horizontal Moves with precomputed h_checks */
    ctx.board = board;
    ctx.is_transposed = false;
    ctx.cross_checks = (const CrossCheck (*)[BOARD_SIZE])cache->h_checks;
    init_context_rack(&ctx, rack);
    generate_row_moves(&ctx);

    /* 2. Generate Vertical Moves with precomputed transposed board and v_checks */
    ctx.board = &cache->transposed;
    ctx.is_transposed = true;
    ctx.cross_checks = (const CrossCheck (*)[BOARD_SIZE])cache->v_checks;
    init_context_rack(&ctx, rack);
    generate_row_moves(&ctx);

    return list->count;
}

int movegen_generate_all(const Board *board, const Dawg *dawg, const Rack *rack, MoveList *list) {
    BoardCrossCache cache;
    board_cross_cache_init(&cache, board, dawg);
    return movegen_generate_all_cached(board, &cache, dawg, rack, list);
}

int movegen_find_best_response_score_cached(const Board *board, const BoardCrossCache *cache,
                                            const Dawg *dawg, const Rack *rack) {
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
    ctx.cross_checks = (const CrossCheck (*)[BOARD_SIZE])cache->h_checks;
    init_context_rack(&ctx, rack);
    generate_row_moves(&ctx);

    /* 2. Generate Vertical Moves */
    ctx.board = &cache->transposed;
    ctx.is_transposed = true;
    ctx.cross_checks = (const CrossCheck (*)[BOARD_SIZE])cache->v_checks;
    init_context_rack(&ctx, rack);
    generate_row_moves(&ctx);

    return ctx.best_score;
}

int movegen_find_best_response_score(const Board *board, const Dawg *dawg, const Rack *rack) {
    BoardCrossCache cache;
    board_cross_cache_init(&cache, board, dawg);
    return movegen_find_best_response_score_cached(board, &cache, dawg, rack);
}

float movegen_find_best_response_equity_cached(const Board *board, const BoardCrossCache *cache,
                                               const Dawg *dawg, const Rack *rack, int bag_count) {
    GenContext ctx;
    ctx.orig_board = board;
    ctx.dawg = dawg;
    ctx.orig_rack = rack;
    ctx.list = NULL;
    ctx.best_score = 0;
    ctx.best_equity = -1000.0f;
    ctx.bag_count = bag_count;
    ctx.track_equity = true;

    /* 1. Generate Horizontal Moves with precomputed h_checks */
    ctx.board = board;
    ctx.is_transposed = false;
    ctx.cross_checks = (const CrossCheck (*)[BOARD_SIZE])cache->h_checks;
    init_context_rack(&ctx, rack);
    generate_row_moves(&ctx);

    /* 2. Generate Vertical Moves with precomputed transposed board and v_checks */
    ctx.board = &cache->transposed;
    ctx.is_transposed = true;
    ctx.cross_checks = (const CrossCheck (*)[BOARD_SIZE])cache->v_checks;
    init_context_rack(&ctx, rack);
    generate_row_moves(&ctx);

    if (ctx.best_equity < -500.0f) return 0.0f;
    return ctx.best_equity;
}

float movegen_find_best_response_equity(const Board *board, const Dawg *dawg, const Rack *rack, int bag_count) {
    BoardCrossCache cache;
    board_cross_cache_init(&cache, board, dawg);
    return movegen_find_best_response_equity_cached(board, &cache, dawg, rack, bag_count);
}
