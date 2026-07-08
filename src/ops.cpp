
#include "../include/ops.h"
#include "../include/config.h"
#include <cassert>
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

// apply rope on the inner dimension
void rope(float *in, float *out, size_t M, size_t N) {
    assert(N % 2 == 0 && "Inner RoPE dimension should be multiple of 2");

    for (int i = 0; i < M; i++) {        // this refers to the seq pos
        for (int j = 0; j < N; j += 2) { // this refers to the pointer within the hidden dimension
            float theta = 1.0f / std::pow(10000, static_cast<float>(j) / N);
            // TODO: verify that this absolute_position thing is correct
            // this should probably be for the latest entry in the forward pass
            // but I am not sure how it plays into this
            // Note also that I dont really know what the in and out should be
            float frequency = i * theta;

            float sin = std::sin(frequency);
            float cos = std::cos(frequency);

            int idx = i * N + j; // this refers to the even index, while +1 refers to the odd index
            out[idx] = in[idx] * cos - in[idx + 1] * sin;
            out[idx + 1] = in[idx] * sin + in[idx + 1] * cos;
        }
    }
}