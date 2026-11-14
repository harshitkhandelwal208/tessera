#ifndef DICT_H
#define DICT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/*
 * DAWG Edge representation (32-bit compact integer):
 * Bit 0:      is_terminal (1 if word ends here, 0 otherwise)
 * Bit 1:      is_last_child (1 if this is the last sibling for this parent node)
 * Bits 2-6:   letter (0..25 representing 'A'..'Z')
 * Bits 7-31:  child_index (index in edges array where children start; 0 if leaf)
 */
typedef uint32_t DawgEdge;

#define DAWG_TERMINAL_MASK    (1U << 0)
#define DAWG_LAST_CHILD_MASK  (1U << 1)
#define DAWG_LETTER_SHIFT     2
#define DAWG_LETTER_MASK      (0x1FU << DAWG_LETTER_SHIFT)
#define DAWG_CHILD_SHIFT      7
#define DAWG_CHILD_MASK       (0x1FFFFFFU << DAWG_CHILD_SHIFT)

static inline bool dawg_is_terminal(DawgEdge edge) {
    return (edge & DAWG_TERMINAL_MASK) != 0;
}

static inline bool dawg_is_last_child(DawgEdge edge) {
    return (edge & DAWG_LAST_CHILD_MASK) != 0;
}

static inline char dawg_letter(DawgEdge edge) {
    return (char)('A' + ((edge & DAWG_LETTER_MASK) >> DAWG_LETTER_SHIFT));
}

static inline uint32_t dawg_letter_idx(DawgEdge edge) {
    return (edge & DAWG_LETTER_MASK) >> DAWG_LETTER_SHIFT;
}

static inline uint32_t dawg_child_index(DawgEdge edge) {
    return (edge & DAWG_CHILD_MASK) >> DAWG_CHILD_SHIFT;
}

typedef struct {
    DawgEdge *edges;
    uint32_t num_edges;
    uint32_t root_index;
    uint32_t word_count;
} Dawg;

/* Lifecycle */
Dawg* dawg_create_from_text(const char *text_filepath);
Dawg* dawg_load_binary(const char *bin_filepath);
bool  dawg_save_binary(const Dawg *dawg, const char *bin_filepath);
void  dawg_free(Dawg *dawg);

/* Query */
bool dawg_contains(const Dawg *dawg, const char *word);

static inline uint32_t dawg_step(const Dawg *dawg, uint32_t node_index, char letter, bool *is_terminal) {
    if (node_index == 0 || node_index >= dawg->num_edges) return 0;

    uint32_t target_code = (letter >= 'a' && letter <= 'z') ? (uint32_t)(letter - 'a') : (uint32_t)(letter - 'A');
    uint32_t idx = node_index;

    while (1) {
        DawgEdge edge = dawg->edges[idx];
        uint32_t edge_code = (edge & DAWG_LETTER_MASK) >> DAWG_LETTER_SHIFT;
        if (edge_code == target_code) {
            if (is_terminal) {
                *is_terminal = (edge & DAWG_TERMINAL_MASK) != 0;
            }
            return (edge & DAWG_CHILD_MASK) >> DAWG_CHILD_SHIFT;
        }
        if (edge & DAWG_LAST_CHILD_MASK) break;
        idx++;
    }
    return 0;
}

static inline uint32_t dawg_step_child(const Dawg *dawg, uint32_t node_index, char letter) {
    if (node_index == 0 || node_index >= dawg->num_edges) return 0;

    uint32_t target_code = (letter >= 'a' && letter <= 'z') ? (uint32_t)(letter - 'a') : (uint32_t)(letter - 'A');
    uint32_t idx = node_index;

    while (1) {
        DawgEdge edge = dawg->edges[idx];
        if (dawg_letter_idx(edge) == target_code) {
            return dawg_child_index(edge);
        }
        if (dawg_is_last_child(edge)) break;
        idx++;
    }
    return 0;
}

static inline bool dawg_step_edge(const Dawg *dawg, uint32_t node_index, char letter, DawgEdge *out_edge) {
    if (node_index == 0 || node_index >= dawg->num_edges) return false;

    uint32_t target_code = (letter >= 'a' && letter <= 'z') ? (uint32_t)(letter - 'a') : (uint32_t)(letter - 'A');
    uint32_t idx = node_index;

    while (1) {
        DawgEdge edge = dawg->edges[idx];
        if (dawg_letter_idx(edge) == target_code) {
            *out_edge = edge;
            return true;
        }
        if (dawg_is_last_child(edge)) break;
        idx++;
    }
    return false;
}

#endif /* DICT_H */
