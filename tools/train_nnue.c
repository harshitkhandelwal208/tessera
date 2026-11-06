#include "nnue.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define HUBER_DELTA 0.10f
#define DEFAULT_EPOCHS 25
#define DEFAULT_BATCH_SIZE 256
#define WEIGHT_DECAY 1e-5f

typedef struct {
    float features[NNUE_INPUT_DIM];
    float target;
} TrainingSample;

/* Adam optimizer state */
typedef struct {
    float m_w1[NNUE_L1_DIM * NNUE_INPUT_DIM];
    float v_w1[NNUE_L1_DIM * NNUE_INPUT_DIM];
    float m_b1[NNUE_L1_DIM];
    float v_b1[NNUE_L1_DIM];

    float m_w2[NNUE_L2_DIM * NNUE_L1_DIM];
    float v_w2[NNUE_L2_DIM * NNUE_L1_DIM];
    float m_b2[NNUE_L2_DIM];
    float v_b2[NNUE_L2_DIM];

    float m_w3[NNUE_L2_DIM];
    float v_w3[NNUE_L2_DIM];
    float m_b3;
    float v_b3;

    int t;
} AdamState;

static void adam_update_array(float *param, const float *grad, float *m, float *v,
                              int size, float lr, float beta1, float beta2,
                              float eps, float weight_decay, int t) {
    float beta1_t = 1.0f - powf(beta1, (float)t);
    float beta2_t = 1.0f - powf(beta2, (float)t);

    for (int i = 0; i < size; i++) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

        float m_hat = m[i] / beta1_t;
        float v_hat = v[i] / beta2_t;

        param[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param[i]);
    }
}

int main(int argc, char *argv[]) {
    const char *data_path = "data/train_data.bin";
    const char *model_path = "data/nnue.bin";
    int num_epochs = DEFAULT_EPOCHS;
    int batch_size = DEFAULT_BATCH_SIZE;

    if (argc >= 2) data_path = argv[1];
    if (argc >= 3) model_path = argv[2];
    if (argc >= 4) num_epochs = atoi(argv[3]);

    printf("======================================================================\n");
    printf("   TESSERA DEEP NNUE TRAINER (NATIVE C + OPENMP + AVX2)               \n");
    printf("======================================================================\n");
    printf("Dataset: %s | Target Model: %s | Epochs: %d | Batch Size: %d\n",
           data_path, model_path, num_epochs, batch_size);
    printf("Training on %d CPU threads with vectorized Adam optimizer...\n\n", omp_get_max_threads());

    FILE *fin = fopen(data_path, "rb");
    if (!fin) {
        fprintf(stderr, "Error: Could not open dataset %s\n", data_path);
        return 1;
    }

    fseek(fin, 0, SEEK_END);
    long file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    size_t total_samples = file_size / sizeof(TrainingSample);
    if (total_samples == 0) {
        fprintf(stderr, "Error: Dataset is empty!\n");
        fclose(fin);
        return 1;
    }

    printf("Loading %zu training samples (%.1f MB) into RAM...\n",
           total_samples, (double)file_size / (1024.0 * 1024.0));

    TrainingSample *dataset = (TrainingSample*)malloc(total_samples * sizeof(TrainingSample));
    if (!dataset) {
        fprintf(stderr, "Error: Out of memory allocating dataset!\n");
        fclose(fin);
        return 1;
    }

    if (fread(dataset, sizeof(TrainingSample), total_samples, fin) != total_samples) {
        fprintf(stderr, "Error reading dataset samples\n");
        free(dataset);
        fclose(fin);
        return 1;
    }
    fclose(fin);

    /* Train / Validation split (85% / 15%) */
    size_t train_size = (size_t)((double)total_samples * 0.85);
    size_t val_size = total_samples - train_size;
    printf("Dataset Split: %zu Training Samples | %zu Validation Samples\n\n", train_size, val_size);

    /* Initialize NNUE model (Warm-start from existing weights if present) */
    NNUEModel model;
    bool is_warm_start = false;
    if (nnue_load(&model, model_path)) {
        printf("[INFO] Warm-starting from existing model %s for fine-tuning...\n", model_path);
        is_warm_start = true;
    } else {
        printf("[INFO] No existing model found. Initializing random weights...\n");
        nnue_init_random(&model);
    }

    AdamState adam;
    memset(&adam, 0, sizeof(AdamState));

    float best_val_mae = 1e9f;
    if (is_warm_start) {
        double init_val_mae_sum = 0.0;
        #pragma omp parallel for reduction(+:init_val_mae_sum) schedule(static)
        for (size_t i = 0; i < val_size; i++) {
            const TrainingSample *sample = &dataset[train_size + i];
            float pred = nnue_forward(&model, sample->features) / 50.0f;
            init_val_mae_sum += fabsf(pred - sample->target);
        }
        best_val_mae = (float)(init_val_mae_sum / val_size);
        printf("[INFO] Baseline Validation MAE of loaded weights: %.4f (%.2f pts margin)\n\n",
               best_val_mae, best_val_mae * 50.0f);
    }

    float initial_lr = is_warm_start ? 3.5e-4f : 1.5e-3f;
    float min_lr = is_warm_start ? 1.0e-5f : 2.0e-5f;

    clock_t start_time = clock();

    /* Training Loop */
    for (int epoch = 1; epoch <= num_epochs; epoch++) {
        clock_t ep_start = clock();

        /* Cosine annealing learning rate */
        float progress = (float)(epoch - 1) / (float)num_epochs;
        float lr = min_lr + 0.5f * (initial_lr - min_lr) * (1.0f + cosf(progress * 3.14159265f));

        /* Shuffle training indices */
        for (size_t i = train_size - 1; i > 0; i--) {
            size_t j = (size_t)rand() % (i + 1);
            TrainingSample tmp = dataset[i];
            dataset[i] = dataset[j];
            dataset[j] = tmp;
        }

        double train_loss_sum = 0.0;
        size_t num_batches = train_size / batch_size;

        for (size_t b = 0; b < num_batches; b++) {
            size_t batch_offset = b * batch_size;

            /* Accumulate batch gradients across threads */
            float grad_w1[NNUE_L1_DIM * NNUE_INPUT_DIM] = {0};
            float grad_b1[NNUE_L1_DIM] = {0};
            float grad_w2[NNUE_L2_DIM * NNUE_L1_DIM] = {0};
            float grad_b2[NNUE_L2_DIM] = {0};
            float grad_w3[NNUE_L2_DIM] = {0};
            float grad_b3 = 0.0f;
            double batch_loss = 0.0;

            #pragma omp parallel reduction(+:batch_loss)
            {
                float local_gw1[NNUE_L1_DIM * NNUE_INPUT_DIM] = {0};
                float local_gb1[NNUE_L1_DIM] = {0};
                float local_gw2[NNUE_L2_DIM * NNUE_L1_DIM] = {0};
                float local_gb2[NNUE_L2_DIM] = {0};
                float local_gw3[NNUE_L2_DIM] = {0};
                float local_gb3 = 0.0f;

                #pragma omp for schedule(static)
                for (int s = 0; s < batch_size; s++) {
                    const TrainingSample *sample = &dataset[batch_offset + s];

                    /* Forward Pass */
                    float z1[NNUE_L1_DIM];
                    float a1[NNUE_L1_DIM];
                    for (int o = 0; o < NNUE_L1_DIM; o++) {
                        float sum = model.b1[o];
                        const float *w_row = &model.w1[o * NNUE_INPUT_DIM];
                        for (int i = 0; i < NNUE_INPUT_DIM; i++) {
                            sum += w_row[i] * sample->features[i];
                        }
                        z1[o] = sum;
                        a1[o] = fminf(fmaxf(sum, 0.0f), 1.0f); /* ClippedReLU */
                    }

                    float z2[NNUE_L2_DIM];
                    float a2[NNUE_L2_DIM];
                    for (int o = 0; o < NNUE_L2_DIM; o++) {
                        float sum = model.b2[o];
                        const float *w_row = &model.w2[o * NNUE_L1_DIM];
                        for (int i = 0; i < NNUE_L1_DIM; i++) {
                            sum += w_row[i] * a1[i];
                        }
                        z2[o] = sum;
                        a2[o] = fminf(fmaxf(sum, 0.0f), 1.0f); /* ClippedReLU */
                    }

                    float y_hat = model.b3;
                    for (int i = 0; i < NNUE_L2_DIM; i++) {
                        y_hat += model.w3[i] * a2[i];
                    }

                    /* Huber Loss & Gradient */
                    float diff = y_hat - sample->target;
                    float abs_diff = fabsf(diff);
                    float loss = 0.0f;
                    float d_out = 0.0f;

                    if (abs_diff <= HUBER_DELTA) {
                        loss = 0.5f * diff * diff;
                        d_out = diff;
                    } else {
                        loss = HUBER_DELTA * (abs_diff - 0.5f * HUBER_DELTA);
                        d_out = (diff > 0.0f) ? HUBER_DELTA : -HUBER_DELTA;
                    }
                    batch_loss += loss;

                    /* Backward Pass */
                    local_gb3 += d_out;
                    float d_a2[NNUE_L2_DIM];
                    for (int i = 0; i < NNUE_L2_DIM; i++) {
                        local_gw3[i] += d_out * a2[i];
                        d_a2[i] = d_out * model.w3[i];
                    }

                    float d_z2[NNUE_L2_DIM];
                    for (int i = 0; i < NNUE_L2_DIM; i++) {
                        d_z2[i] = (z2[i] > 0.0f && z2[i] < 1.0f) ? d_a2[i] : 0.0f;
                        local_gb2[i] += d_z2[i];
                    }

                    float d_a1[NNUE_L1_DIM] = {0};
                    for (int o = 0; o < NNUE_L2_DIM; o++) {
                        const float *w_row = &model.w2[o * NNUE_L1_DIM];
                        float *gw_row = &local_gw2[o * NNUE_L1_DIM];
                        float dz = d_z2[o];
                        for (int i = 0; i < NNUE_L1_DIM; i++) {
                            gw_row[i] += dz * a1[i];
                            d_a1[i] += dz * w_row[i];
                        }
                    }

                    float d_z1[NNUE_L1_DIM];
                    for (int i = 0; i < NNUE_L1_DIM; i++) {
                        d_z1[i] = (z1[i] > 0.0f && z1[i] < 1.0f) ? d_a1[i] : 0.0f;
                        local_gb1[i] += d_z1[i];
                    }

                    for (int o = 0; o < NNUE_L1_DIM; o++) {
                        float dz = d_z1[o];
                        float *gw_row = &local_gw1[o * NNUE_INPUT_DIM];
                        for (int i = 0; i < NNUE_INPUT_DIM; i++) {
                            gw_row[i] += dz * sample->features[i];
                        }
                    }
                }

                #pragma omp critical
                {
                    for (int i = 0; i < NNUE_L1_DIM * NNUE_INPUT_DIM; i++) grad_w1[i] += local_gw1[i];
                    for (int i = 0; i < NNUE_L1_DIM; i++) grad_b1[i] += local_gb1[i];
                    for (int i = 0; i < NNUE_L2_DIM * NNUE_L1_DIM; i++) grad_w2[i] += local_gw2[i];
                    for (int i = 0; i < NNUE_L2_DIM; i++) grad_b2[i] += local_gb2[i];
                    for (int i = 0; i < NNUE_L2_DIM; i++) grad_w3[i] += local_gw3[i];
                    grad_b3 += local_gb3;
                }
            }

            train_loss_sum += batch_loss;

            /* Normalize gradients by batch size */
            float inv_batch = 1.0f / (float)batch_size;
            for (int i = 0; i < NNUE_L1_DIM * NNUE_INPUT_DIM; i++) grad_w1[i] *= inv_batch;
            for (int i = 0; i < NNUE_L1_DIM; i++) grad_b1[i] *= inv_batch;
            for (int i = 0; i < NNUE_L2_DIM * NNUE_L1_DIM; i++) grad_w2[i] *= inv_batch;
            for (int i = 0; i < NNUE_L2_DIM; i++) grad_b2[i] *= inv_batch;
            for (int i = 0; i < NNUE_L2_DIM; i++) grad_w3[i] *= inv_batch;
            grad_b3 *= inv_batch;

            /* Adam Parameter Update */
            adam.t++;
            adam_update_array(model.w1, grad_w1, adam.m_w1, adam.v_w1,
                              NNUE_L1_DIM * NNUE_INPUT_DIM, lr, 0.9f, 0.999f, 1e-8f, WEIGHT_DECAY, adam.t);
            adam_update_array(model.b1, grad_b1, adam.m_b1, adam.v_b1,
                              NNUE_L1_DIM, lr, 0.9f, 0.999f, 1e-8f, 0.0f, adam.t);
            adam_update_array(model.w2, grad_w2, adam.m_w2, adam.v_w2,
                              NNUE_L2_DIM * NNUE_L1_DIM, lr, 0.9f, 0.999f, 1e-8f, WEIGHT_DECAY, adam.t);
            adam_update_array(model.b2, grad_b2, adam.m_b2, adam.v_b2,
                              NNUE_L2_DIM, lr, 0.9f, 0.999f, 1e-8f, 0.0f, adam.t);
            adam_update_array(model.w3, grad_w3, adam.m_w3, adam.v_w3,
                              NNUE_L2_DIM, lr, 0.9f, 0.999f, 1e-8f, WEIGHT_DECAY, adam.t);
            adam_update_array(&model.b3, &grad_b3, &adam.m_b3, &adam.v_b3,
                              1, lr, 0.9f, 0.999f, 1e-8f, 0.0f, adam.t);
        }

        /* Validation Evaluation */
        double val_mae_sum = 0.0;
        double val_mse_sum = 0.0;

        #pragma omp parallel for reduction(+:val_mae_sum, val_mse_sum) schedule(static)
        for (size_t i = 0; i < val_size; i++) {
            const TrainingSample *sample = &dataset[train_size + i];
            float pred = nnue_forward(&model, sample->features) / 50.0f;
            float err = pred - sample->target;
            val_mae_sum += fabsf(err);
            val_mse_sum += err * err;
        }

        float val_mae = (float)(val_mae_sum / val_size);
        float val_mse = (float)(val_mse_sum / val_size);
        float train_loss = (float)(train_loss_sum / (num_batches * batch_size));
        double ep_time = (double)(clock() - ep_start) / CLOCKS_PER_SEC;

        bool is_best = false;
        if (val_mae < best_val_mae) {
            best_val_mae = val_mae;
            is_best = true;
            /* Calibrate win head proportionally to score head */
            for (int i = 0; i < NNUE_L2_DIM; i++) {
                model.w_win[i] = model.w3[i] * 1.5f;
            }
            model.b_win = model.b3 * 1.5f;
            nnue_save(&model, model_path);
        }

        printf("  [Epoch %2d/%2d] Train Loss: %.5f | Val MAE: %.4f (%.2f pts) | Val MSE: %.5f | LR: %.6f | Time: %.2fs %s\n",
               epoch, num_epochs, train_loss, val_mae, val_mae * 50.0f, val_mse, lr, ep_time,
               is_best ? "<-- [BEST CHECKPOINT SAVED]" : "");
    }

    free(dataset);
    double total_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("\n>>> Training finished in %.2f seconds!\n", total_time);
    printf("  Best Validation MAE: %.4f (%.2f points margin)\n", best_val_mae, best_val_mae * 50.0f);
    printf("  Model successfully saved to: %s\n", model_path);
    printf("======================================================================\n");

    return 0;
}
