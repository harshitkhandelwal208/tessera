#include "dict.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_WORD_BUFFER 128
#define HASH_TABLE_SIZE 1048576 /* 2^20 buckets */

typedef struct BNode {
    uint32_t id;
    bool is_terminal;
    uint8_t num_children;
    char letters[26];
    struct BNode *children[26];
    struct BNode *next_hash; /* Hash table collision chain */
} BNode;

static BNode* bnode_create(void) {
    BNode *n = (BNode*)calloc(1, sizeof(BNode));
    return n;
}

static uint32_t bnode_hash(const BNode *n) {
    uint32_t h = n->is_terminal ? 0x9e3779b9 : 0;
    h ^= (uint32_t)n->num_children * 31;
    for (int i = 0; i < n->num_children; i++) {
        uint32_t child_id = n->children[i] ? n->children[i]->id : 0;
        h = (h << 5) - h + (uint8_t)n->letters[i];
        h ^= child_id + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

static bool bnode_equals(const BNode *a, const BNode *b) {
    if (a->is_terminal != b->is_terminal) return false;
    if (a->num_children != b->num_children) return false;
    for (int i = 0; i < a->num_children; i++) {
        if (a->letters[i] != b->letters[i]) return false;
        if (a->children[i] != b->children[i]) return false;
    }
    return true;
}

typedef struct {
    BNode **buckets;
    uint32_t node_counter;
} DawgBuilder;

static void builder_init(DawgBuilder *b) {
    b->buckets = (BNode**)calloc(HASH_TABLE_SIZE, sizeof(BNode*));
    b->node_counter = 1; /* 0 is reserved for null/leaf */
}

static void builder_free(DawgBuilder *b) {
    if (b->buckets) free(b->buckets);
}

static BNode* find_or_register(DawgBuilder *b, BNode *node) {
    uint32_t h = bnode_hash(node) % HASH_TABLE_SIZE;
    BNode *cur = b->buckets[h];
    while (cur) {
        if (bnode_equals(cur, node)) {
            /* Found identical existing state */
            free(node);
            return cur;
        }
        cur = cur->next_hash;
    }

    /* Assign new unique id and insert */
    node->id = b->node_counter++;
    node->next_hash = b->buckets[h];
    b->buckets[h] = node;
    return node;
}

/* Flatten the minimized DAG into flat DawgEdge array */
typedef struct {
    DawgEdge *edges;
    uint32_t capacity;
    uint32_t count;
    uint32_t *node_edge_offset; /* Map node id -> edge index */
    uint32_t max_id;
} Serializer;

static void collect_nodes(BNode *node, bool *visited, uint32_t *total_edges) {
    if (!node || visited[node->id]) return;
    visited[node->id] = true;
    *total_edges += node->num_children;
    for (int i = 0; i < node->num_children; i++) {
        collect_nodes(node->children[i], visited, total_edges);
    }
}

static void serialize_graph(BNode *node, Serializer *s, bool *placed) {
    if (!node || placed[node->id]) return;
    placed[node->id] = true;

    if (node->num_children == 0) return;

    /* Record where this node's edges start */
    s->node_edge_offset[node->id] = s->count;

    uint32_t start_edge = s->count;
    s->count += node->num_children;

    for (int i = 0; i < node->num_children; i++) {
        DawgEdge edge = 0;
        char l = node->letters[i];
        uint32_t letter_val = (uint32_t)(l - 'A');
        edge |= (letter_val << DAWG_LETTER_SHIFT);

        if (i == node->num_children - 1) {
            edge |= DAWG_LAST_CHILD_MASK;
        }
        if (node->children[i]->is_terminal) {
            edge |= DAWG_TERMINAL_MASK;
        }

        s->edges[start_edge + i] = edge;
    }

    for (int i = 0; i < node->num_children; i++) {
        serialize_graph(node->children[i], s, placed);
    }
}

static void patch_child_pointers(BNode *node, Serializer *s, bool *visited) {
    if (!node || visited[node->id]) return;
    visited[node->id] = true;

    if (node->num_children > 0) {
        uint32_t start_edge = s->node_edge_offset[node->id];
        for (int i = 0; i < node->num_children; i++) {
            BNode *child = node->children[i];
            uint32_t child_offset = 0;
            if (child->num_children > 0) {
                child_offset = s->node_edge_offset[child->id];
            }
            s->edges[start_edge + i] |= (child_offset << DAWG_CHILD_SHIFT);
        }
    }

    for (int i = 0; i < node->num_children; i++) {
        patch_child_pointers(node->children[i], s, visited);
    }
}

Dawg* dawg_create_from_text(const char *text_filepath) {
    FILE *fp = fopen(text_filepath, "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open dictionary file: %s\n", text_filepath);
        return NULL;
    }

    DawgBuilder builder;
    builder_init(&builder);

    BNode *root = bnode_create();
    root->id = builder.node_counter++;

    /* Active path tracking */
    BNode *active_nodes[MAX_WORD_BUFFER];
    char active_letters[MAX_WORD_BUFFER];
    active_nodes[0] = root;

    char prev_word[MAX_WORD_BUFFER] = "";
    char line[MAX_WORD_BUFFER];
    uint32_t word_count = 0;

    while (fgets(line, sizeof(line), fp)) {
        /* Strip whitespace */
        char word[MAX_WORD_BUFFER];
        int wlen = 0;
        for (int i = 0; line[i]; i++) {
            if (isalpha((unsigned char)line[i])) {
                word[wlen++] = (char)toupper((unsigned char)line[i]);
            }
        }
        word[wlen] = '\0';
        if (wlen < 2 || wlen > 15) continue; /* Standard Scrabble valid lengths */

        /* Find common prefix with previous word */
        int common_prefix = 0;
        while (common_prefix < wlen &&
               common_prefix < (int)strlen(prev_word) &&
               word[common_prefix] == prev_word[common_prefix]) {
            common_prefix++;
        }

        /* Minimize divergent suffix of previous word from bottom up */
        for (int i = (int)strlen(prev_word); i > common_prefix; i--) {
            BNode *parent = active_nodes[i - 1];
            char let = active_letters[i - 1];
            BNode *child = active_nodes[i];

            BNode *minimized_child = find_or_register(&builder, child);
            /* Update parent's child pointer */
            for (int c = 0; c < parent->num_children; c++) {
                if (parent->letters[c] == let) {
                    parent->children[c] = minimized_child;
                    break;
                }
            }
        }

        /* Add current word suffix */
        for (int i = common_prefix; i < wlen; i++) {
            BNode *new_node = bnode_create();
            BNode *parent = active_nodes[i];
            char let = word[i];

            parent->letters[parent->num_children] = let;
            parent->children[parent->num_children] = new_node;
            parent->num_children++;

            active_nodes[i + 1] = new_node;
            active_letters[i] = let;
        }

        active_nodes[wlen]->is_terminal = true;
        strcpy(prev_word, word);
        word_count++;
    }
    fclose(fp);

    /* Minimize remainder of active path down to root */
    for (int i = (int)strlen(prev_word); i > 0; i--) {
        BNode *parent = active_nodes[i - 1];
        char let = active_letters[i - 1];
        BNode *child = active_nodes[i];

        BNode *minimized_child = find_or_register(&builder, child);
        for (int c = 0; c < parent->num_children; c++) {
            if (parent->letters[c] == let) {
                parent->children[c] = minimized_child;
                break;
            }
        }
    }

    /* Serialize to flat array */
    bool *visited = (bool*)calloc(builder.node_counter + 1, sizeof(bool));
    uint32_t total_edges = 1; /* 1 dummy edge at index 0 for safety */
    collect_nodes(root, visited, &total_edges);
    free(visited);

    Serializer s;
    s.edges = (DawgEdge*)calloc(total_edges + 16, sizeof(DawgEdge));
    s.capacity = total_edges + 16;
    s.count = 1; /* Index 0 is root or null */
    s.max_id = builder.node_counter;
    s.node_edge_offset = (uint32_t*)calloc(builder.node_counter + 1, sizeof(uint32_t));

    bool *placed = (bool*)calloc(builder.node_counter + 1, sizeof(bool));
    serialize_graph(root, &s, placed);
    free(placed);

    bool *patched = (bool*)calloc(builder.node_counter + 1, sizeof(bool));
    patch_child_pointers(root, &s, patched);
    free(patched);

    Dawg *dawg = (Dawg*)malloc(sizeof(Dawg));
    dawg->edges = s.edges;
    dawg->num_edges = s.count;
    dawg->root_index = s.node_edge_offset[root->id];
    dawg->word_count = word_count;

    free(s.node_edge_offset);
    /* Clean up builder buckets (free remaining un-freed nodes) */
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        BNode *cur = builder.buckets[i];
        while (cur) {
            BNode *next = cur->next_hash;
            free(cur);
            cur = next;
        }
    }
    builder_free(&builder);
    free(root);

    return dawg;
}

#define DAWG_MAGIC 0x47574144 /* "DAWG" in little endian */
#define DAWG_VERSION 1

bool dawg_save_binary(const Dawg *dawg, const char *bin_filepath) {
    if (!dawg || !dawg->edges) return false;
    FILE *fp = fopen(bin_filepath, "wb");
    if (!fp) return false;

    uint32_t header[4] = {
        DAWG_MAGIC,
        DAWG_VERSION,
        dawg->num_edges,
        dawg->root_index
    };
    uint32_t wc = dawg->word_count;

    if (fwrite(header, sizeof(uint32_t), 4, fp) != 4) { fclose(fp); return false; }
    if (fwrite(&wc, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return false; }
    if (fwrite(dawg->edges, sizeof(DawgEdge), dawg->num_edges, fp) != dawg->num_edges) {
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

Dawg* dawg_load_binary(const char *bin_filepath) {
    FILE *fp = fopen(bin_filepath, "rb");
    if (!fp) return NULL;

    uint32_t header[4];
    if (fread(header, sizeof(uint32_t), 4, fp) != 4) { fclose(fp); return NULL; }
    if (header[0] != DAWG_MAGIC || header[1] != DAWG_VERSION) {
        fclose(fp);
        return NULL;
    }

    uint32_t wc = 0;
    if (fread(&wc, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return NULL; }

    uint32_t num_edges = header[2];
    uint32_t root_index = header[3];

    DawgEdge *edges = (DawgEdge*)malloc(sizeof(DawgEdge) * num_edges);
    if (!edges) { fclose(fp); return NULL; }

    if (fread(edges, sizeof(DawgEdge), num_edges, fp) != num_edges) {
        free(edges);
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    Dawg *dawg = (Dawg*)malloc(sizeof(Dawg));
    dawg->edges = edges;
    dawg->num_edges = num_edges;
    dawg->root_index = root_index;
    dawg->word_count = wc;
    return dawg;
}

void dawg_free(Dawg *dawg) {
    if (!dawg) return;
    if (dawg->edges) free(dawg->edges);
    free(dawg);
}

/*
 * Step through the DAWG by one letter.
 * node_index: offset of the first sibling edge of this node.
 * Returns child edge offset (>0) if found, or 0 if no matching letter edge.
 * If is_terminal is non-null, sets it to true if this edge completes a word.
 */
uint32_t dawg_step(const Dawg *dawg, uint32_t node_index, char letter, bool *is_terminal) {
    if (node_index == 0 || node_index >= dawg->num_edges) return 0;

    letter = (char)toupper((unsigned char)letter);
    uint32_t idx = node_index;

    while (1) {
        DawgEdge edge = dawg->edges[idx];
        if (dawg_letter(edge) == letter) {
            if (is_terminal) {
                *is_terminal = dawg_is_terminal(edge);
            }
            return dawg_child_index(edge);
        }
        if (dawg_is_last_child(edge)) break;
        idx++;
    }
    return 0;
}

bool dawg_contains(const Dawg *dawg, const char *word) {
    if (!dawg || !word) return false;
    int len = (int)strlen(word);
    if (len == 0) return false;

    uint32_t node = dawg->root_index;
    bool is_term = false;

    for (int i = 0; i < len; i++) {
        char c = (char)toupper((unsigned char)word[i]);
        is_term = false;
        node = dawg_step(dawg, node, c, &is_term);
        if (node == 0 && i < len - 1) {
            /* Prefix doesn't exist */
            return false;
        }
    }
    return is_term;
}
