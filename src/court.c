#include "court.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Box-Muller transform for standard normal random variable */
static float sample_standard_normal(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.1415926535f * u2);
}

CourtVerdict court_deliberate(const Lattice *lat, uint32_t root_idx) {
    CourtVerdict v;
    v.confidence = 0.0f;
    v.best_edge_idx = 0;
    v.should_adjourn = false;
    v.deliberated_steps = 0;

    if (root_idx == 0 || root_idx >= lat->num_nodes) {
        v.should_adjourn = true;
        return v;
    }

    const LatticeNode *root = &lat->nodes[root_idx];
    if (!root->is_expanded || root->num_edges == 0) {
        v.should_adjourn = true;
        return v;
    }

    if (root->num_edges == 1) {
        /* Only 1 legal move: trivial instant snap decision! */
        v.confidence = 1.0f;
        v.best_edge_idx = 0;
        v.should_adjourn = true;
        return v;
    }

    /* If no deliberation visits have occurred yet, deliberation must proceed */
    if (root->edges[0].visits == 0) {
        v.confidence = 0.5f;
        v.best_edge_idx = 0;
        v.should_adjourn = false;
        return v;
    }

    /* 1. Identify current empirical argmax: Edge 0 is default unless proven or deliberated better */
    int argmax_idx = 0;
    float max_q = root->edges[0].q_value;
    for (int i = 1; i < root->num_edges; i++) {
        uint32_t t = root->edges[i].target_node_idx;
        bool is_proven_win = (t > 0 && t < lat->num_nodes && lat->nodes[t].belief.is_proven && lat->nodes[t].belief.proof_sign < 0);

        if (is_proven_win) {
            max_q = root->edges[i].q_value;
            argmax_idx = i;
            break;
        } else if (root->edges[i].visits > 0 && root->edges[i].q_value > root->edges[0].q_value + 1.5f) {
            if (root->edges[i].q_value > max_q) {
                max_q = root->edges[i].q_value;
                argmax_idx = i;
            }
        }
    }
    v.best_edge_idx = argmax_idx;

    /* 2. Check for proven win */
    uint32_t target = root->edges[argmax_idx].target_node_idx;
    if (target > 0 && target < lat->num_nodes) {
        if (lat->nodes[target].belief.is_proven) {
            v.confidence = 1.0f;
            v.should_adjourn = true;
            return v;
        }
    }

    /* 3. Joint posterior sampling to compute P(argmax is truly best) */
    int wins[MAX_LATTICE_EDGES] = {0};
    int n_edges = root->num_edges < MAX_LATTICE_EDGES ? root->num_edges : MAX_LATTICE_EDGES;

    float std_devs[MAX_LATTICE_EDGES];
    for (int i = 0; i < n_edges; i++) {
        float var = root->edges[i].variance / (1.0f + (float)root->edges[i].visits);
        std_devs[i] = sqrtf(fmaxf(0.01f, var));
    }

    for (int s = 0; s < COURT_SAMPLES; s++) {
        int best_sample_idx = 0;
        float best_sample_val = -100000.0f;

        for (int i = 0; i < n_edges; i++) {
            if (i > 0 && root->edges[i].visits == 0) continue;
            float sample = root->edges[i].q_value + (std_devs[i] * sample_standard_normal());
            if (sample > best_sample_val) {
                best_sample_val = sample;
                best_sample_idx = i;
            }
        }
        wins[best_sample_idx]++;
    }

    v.confidence = (float)wins[argmax_idx] / (float)COURT_SAMPLES;

    /* If confidence clears threshold, adjourn deliberation! */
    if (v.confidence >= COURT_CONFIDENCE_THRESHOLD) {
        v.should_adjourn = true;
    }

    return v;
}
