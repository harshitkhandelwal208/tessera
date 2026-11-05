#include "broker.h"
#include <math.h>

void broker_init(Broker *b) {
    b->c_puct = BROKER_C_PUCT;
    b->c_var = BROKER_C_VAR;
    b->leaves_purchased = 0;
    b->proven_bypasses = 0;
}

uint32_t broker_select_frontier_leaf(Broker *b, const Lattice *lat, uint32_t root_idx) {
    if (root_idx == 0 || root_idx >= lat->num_nodes) return 0;

    uint32_t cur = root_idx;
    int depth = 0;

    while (depth < 6) { /* Bound descent depth to 6 plies */
        const LatticeNode *n = &lat->nodes[cur];

        /* If unexpanded, this is our frontier leaf to buy! */
        if (!n->is_expanded) {
            b->leaves_purchased++;
            return cur;
        }

        if (n->num_edges == 0) return 0;

        /* Total parent visits */
        uint32_t total_visits = 1;
        for (int i = 0; i < n->num_edges; i++) {
            total_visits += n->edges[i].visits;
        }
        float sqrt_n = sqrtf((float)total_visits);

        /* Find edge with highest acquisition price */
        int best_edge_idx = 0;
        float best_score = -1000000.0f;

        for (int i = 0; i < n->num_edges; i++) {
            const LatticeEdge *edge = &n->edges[i];

            /* Proven subtrees price at zero uncertainty bonus */
            float var_bonus = 0.0f;
            if (edge->target_node_idx > 0 && edge->target_node_idx < lat->num_nodes) {
                if (lat->nodes[edge->target_node_idx].belief.is_proven) {
                    b->proven_bypasses++;
                } else {
                    var_bonus = b->c_var * sqrtf(fmaxf(0.0f, edge->variance));
                }
            } else {
                var_bonus = b->c_var * sqrtf(fmaxf(0.0f, edge->variance));
            }

            /* PUCT exploration term */
            float u = b->c_puct * edge->prior_prob * (sqrt_n / (1.0f + (float)edge->visits));

            /* Total Broker Price */
            float price = edge->q_value + u + var_bonus;

            if (price > best_score) {
                best_score = price;
                best_edge_idx = i;
            }
        }

        uint32_t next = n->edges[best_edge_idx].target_node_idx;
        if (next == 0 || next >= lat->num_nodes) {
            return 0;
        }

        cur = next;
        depth++;
    }

    return 0;
}

uint32_t broker_select_top_child(Broker *b, Lattice *lat, uint32_t root_idx) {
    if (root_idx == 0 || root_idx >= lat->num_nodes) return 0;
    LatticeNode *root = &lat->nodes[root_idx];
    if (!root->is_expanded || root->num_edges == 0) return 0;

    uint32_t total_visits = 1;
    for (int i = 0; i < root->num_edges; i++) {
        total_visits += root->edges[i].visits;
    }
    float sqrt_n = sqrtf((float)total_visits);

    int best_edge_idx = 0;
    float best_price = -1000000.0f;

    for (int i = 0; i < root->num_edges; i++) {
        const LatticeEdge *edge = &root->edges[i];
        float var_bonus = b->c_var * sqrtf(fmaxf(0.0f, edge->variance));
        float u = b->c_puct * edge->prior_prob * (sqrt_n / (1.0f + (float)edge->visits));
        float price = edge->q_value + u + var_bonus;

        if (price > best_price) {
            best_price = price;
            best_edge_idx = i;
        }
    }

    root->edges[best_edge_idx].visits++;
    b->leaves_purchased++;
    return root->edges[best_edge_idx].target_node_idx;
}
