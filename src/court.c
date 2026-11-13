#include "court.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

static _Thread_local uint64_t s_court_prng = 0x853c49e6748fea9bULL;
static _Thread_local float s_cached_normal = 0.0f;
static _Thread_local bool s_has_cached_normal = false;

static inline float fast_court_uniform(void) {
    s_court_prng ^= s_court_prng >> 12;
    s_court_prng ^= s_court_prng << 25;
    s_court_prng ^= s_court_prng >> 27;
    uint64_t x = s_court_prng * 0x2545F4914F6CDD1DULL;
    return ((float)(x >> 40) + 1.0f) / 16777218.0f;
}

/* Fast Box-Muller transform with paired normal caching */
static inline float sample_standard_normal(void) {
    if (s_has_cached_normal) {
        s_has_cached_normal = false;
        return s_cached_normal;
    }
    float u1 = fast_court_uniform();
    float u2 = fast_court_uniform();
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 6.28318530718f * u2;
    s_cached_normal = r * sinf(theta);
    s_has_cached_normal = true;
    return r * cosf(theta);
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

    /* 1. Identify current empirical argmax */
    int argmax_idx = -1;
    float max_q = -1e9f;
    for (int i = 0; i < root->num_edges; i++) {
        uint32_t t = root->edges[i].target_node_idx;
        bool is_proven_win = (t > 0 && t < lat->num_nodes && lat->nodes[t].belief.is_proven && lat->nodes[t].belief.proof_sign < 0);

        if (is_proven_win) {
            max_q = root->edges[i].q_value;
            argmax_idx = i;
            break;
        } else if (root->edges[i].visits > 0 && root->edges[i].q_value > max_q) {
            max_q = root->edges[i].q_value;
            argmax_idx = i;
        }
    }
    if (argmax_idx < 0) {
        argmax_idx = 0; /* Fallback to prior best if no edges visited yet */
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

    /* Statistical Dominance Check: if argmax's lower bound dominates all rival upper bounds */
    float argmax_lower = root->edges[argmax_idx].q_value - (3.5f * std_devs[argmax_idx]);
    int contenders[MAX_LATTICE_EDGES];
    int num_contenders = 0;

    for (int i = 0; i < n_edges; i++) {
        if (i == argmax_idx) continue;
        if (root->edges[i].visits == 0) continue;
        float upper = root->edges[i].q_value + (3.5f * std_devs[i]);
        if (upper >= argmax_lower) {
            contenders[num_contenders++] = i;
        }
    }

    if (num_contenders == 0) {
        /* No other candidate has non-negligible probability of exceeding argmax: instant 100% confidence */
        v.confidence = 1.0f;
        v.should_adjourn = true;
        return v;
    }

    for (int s = 0; s < COURT_SAMPLES; s++) {
        int best_sample_idx = argmax_idx;
        float best_sample_val = root->edges[argmax_idx].q_value + (std_devs[argmax_idx] * sample_standard_normal());

        for (int c = 0; c < num_contenders; c++) {
            int i = contenders[c];
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
