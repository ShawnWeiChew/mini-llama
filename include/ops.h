#ifndef OPS_H_
#define OPS_H_

#include "llama.h"
#include <unistd.h>

// base ops
void mat_mul(float *a, float *b, float *c, size_t M, size_t K, size_t N);

void softmax(float *in, float *out, size_t M, size_t N);

void rms_norm(
    float *in, float *element_wise_affine, float *out, size_t M, size_t N, float eps = 1e-5
);

void rope(float *in, float *out, size_t M, size_t N);
void rope_with_pos(float *in, float *out, int pos, size_t N, size_t head_size);

void silu(float *in, float *out, size_t M, size_t N);

void transpose(float *in, float *out, size_t M, size_t N);

// nn ops
void multi_headed_attention(
    float *in, size_t layer, size_t pos, TransformerState &state, LlamaConfig &config
);

void feed_forward(float *in, size_t layer, TransformerState &state, LlamaConfig &config);

#endif