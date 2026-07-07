
#include "../include/ops.h"
#include <cmath>
#include <iostream>
#include <memory>

void mat_mul(float *a, float *b, float *c, size_t M, size_t K, size_t N) {
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < K; i++) {
            for (int n = 0; n < N; n++) {
                c[m * N + n] += a[m * K + i] * b[i * N + n];
            }
        }
    }
}

// along the inner dimension
void softmax(float *in, float *out, size_t M, size_t N) {
    // first pass: get the max value and the sum
    for (int i = 0; i < M; i++) {
        float max = -1e20;
        float sum = 0;

        for (int j = 0; j < N; j++) {
            float new_max = std::max(in[i * N + j], max);
            float correction_factor = std::exp(max - new_max);
            max = new_max;

            sum *= correction_factor;
            sum += std::exp(in[i * N + j] - max);
        }

        for (int j = 0; j < N; j++) {
            out[i * N + j] = std::exp(in[i * N + j] - max) / sum;
        }
    }
}

// along the inner dimension
// element-wise-affine should be of the same size as the input
void rms_norm(float *in, float *element_wise_affine, float *out, size_t M, size_t N, float eps) {
    for (int i = 0; i < M; i++) {
        float sum = 0;
        for (int j = 0; j < N; j++) {
            sum += in[i * N + j] * in[i * N + j];
        }
        sum = std::sqrt(eps + sum / N);

        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            out[idx] = in[idx] / sum;
        }
    }
}
