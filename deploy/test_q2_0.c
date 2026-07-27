// Численная сверка пути mul_mat для Q2_0 против точного float-эталона.
// Проверяет разом: type_traits, квантование активаций в Q8_0, vec_dot, repack.
#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K 768
#define ROWS 8
#define COLS 4
#define QK 64

int main(void) {
    srand(7);

    // тернарная матрица, как у нас: {-s, 0, +s}, свой s на блок 64
    float *dense = malloc(sizeof(float) * K * ROWS);
    for (int r = 0; r < ROWS; ++r) {
        for (int blk = 0; blk < K / QK; ++blk) {
            // масштаб делаем точным в fp16, чтобы эталон был честным
            float s = ggml_fp16_to_fp32(ggml_fp32_to_fp16(0.01f + 0.05f * (rand() % 100) / 100.0f));
            for (int j = 0; j < QK; ++j) {
                int code = rand() % 3 - 1;                  // {-1, 0, +1}
                dense[r * K + blk * QK + j] = s * code;
            }
        }
    }

    struct ggml_init_params ip = { 1024 * 1024 * 64, NULL, false };
    struct ggml_context *ctx = ggml_init(ip);

    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_0, K, ROWS);
    ggml_quantize_chunk(GGML_TYPE_Q2_0, dense, a->data, 0, ROWS, K, NULL);

    // проверка 1: их квантование нашей тернарной строки обратимо?
    {
        const int blocks = K / QK;
        float worst = 0.0f;
        const uint8_t *raw = a->data;
        for (int r = 0; r < ROWS; ++r) {
            for (int blk = 0; blk < blocks; ++blk) {
                const uint8_t *cell = raw + (r * blocks + blk) * 18;
                ggml_fp16_t h;
                memcpy(&h, cell, 2);
                float d = ggml_fp16_to_fp32(h);
                for (int j = 0; j < QK; ++j) {
                    int q = (cell[2 + j / 4] >> ((j % 4) * 2)) & 3;
                    float v = (q - 1) * d;
                    float ref = dense[r * K + blk * QK + j];
                    float e = fabsf(v - ref);
                    if (e > worst) worst = e;
                }
            }
        }
        printf("[1] квантование->распаковка, макс ошибка: %.3e %s\n",
               worst, worst < 1e-7 ? "ТОЧНО" : "РАСХОЖДЕНИЕ");
    }

    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, COLS);
    float *bx = (float *) b->data;
    for (int i = 0; i < K * COLS; ++i) bx[i] = (rand() % 2000 - 1000) / 500.0f;

    struct ggml_tensor *out = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // эталон: точный float-матмул по исходной плотной матрице
    float worst_rel = 0.0f;
    int bad = 0;
    for (int c = 0; c < COLS; ++c) {
        for (int r = 0; r < ROWS; ++r) {
            double ref = 0.0;
            for (int j = 0; j < K; ++j) ref += (double) dense[r * K + j] * bx[c * K + j];
            float got = ((float *) out->data)[c * ROWS + r];
            float rel = fabs(got - ref) / (fabs(ref) + 1e-6);
            if (rel > worst_rel) worst_rel = rel;
            if (rel > 0.05f) {
                if (bad < 5)
                    printf("    [%d,%d] эталон %+9.4f  ggml %+9.4f  отн %.3f\n",
                           r, c, (float) ref, got, rel);
                bad++;
            }
        }
    }
    printf("[2] mul_mat против эталона: макс отн ошибка %.4f, клеток >5%%: %d/%d\n",
           worst_rel, bad, ROWS * COLS);
    printf(worst_rel < 0.02f
           ? "[ВЕРДИКТ] путь Q2_0 ЧИСЛЕННО ВЕРЕН — ядро не виновато\n"
           : "[ВЕРДИКТ] путь Q2_0 СЛОМАН — вот воспроизведение для баг-репорта\n");
    ggml_free(ctx);
    free(dense);
    return worst_rel < 0.02f ? 0 : 1;
}
