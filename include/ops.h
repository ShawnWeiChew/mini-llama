#ifndef OPS_H_
#define OPS_H_

#include <unistd.h>

void mat_mul(float *a, float *b, float *c, size_t M, size_t K, size_t N);

void softmax(float *in, float *out, size_t M, size_t N);

void rms_norm(
    float *in, float *element_wise_affine, float *out, size_t M, size_t N, float eps = 1e-5
);

void rope(float *in, float *out, size_t M, size_t N);

void silu(float *in, float *out, size_t M, size_t N);

#endif