
#include "../include/ops.h"
#include "../include/llama.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>

// NOTE: had to change this operation up a little
void mat_mul(float *a, float *b, float *c, size_t M, size_t K, size_t N) {
    // b (N, K) @ a(K, 1) -> c (N, 1)
    for (int n = 0; n < N; n++) {
        c[n] = 0.0f;
        for (int i = 0; i < K; i++) {
            // printf("%d, %f %f\n", n * K + i, a[i], b[n * K + i]);
            c[n] += a[i] * b[n * K + i];
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
            out[idx] = element_wise_affine[j] * in[idx] / sum;
        }
    }
}

// apply rope on the inner dimension
void rope(float *in, float *out, size_t M, size_t N) {
    assert(N % 2 == 0 && "Inner RoPE dimension should be multiple of 2");

    for (int i = 0; i < M; i++) {        // this refers to the seq pos
        for (int j = 0; j < N; j += 2) { // this refers to the pointer within the hidden dimension
            float theta = 1.0f / std::pow(10000.0f, static_cast<float>(j) / N);
            float frequency = i * theta;

            float sin = std::sin(frequency);
            float cos = std::cos(frequency);

            int idx = i * N + j; // this refers to the even index, while +1 refers to the odd index
            float v0 = in[idx];
            float v1 = in[idx + 1];
            out[idx] = v0 * cos - v1 * sin;
            out[idx + 1] = v0 * sin + v1 * cos;
        }
    }
}

// apply rope on the inner dimension with position
void rope_with_pos(float *in, float *out, int pos, size_t N, size_t head_size) {
    assert(N % 2 == 0 && "Inner RoPE dimension should be multiple of 2");
    for (int j = 0; j < N; j += 2) { // this refers to the pointer within the hidden dimension
        int head_dim = j % head_size;
        float theta = 1.0f / std::pow(10000.0f, static_cast<float>(head_dim) / head_size);
        float frequency = pos * theta;

        float sin = std::sin(frequency);
        float cos = std::cos(frequency);

        float v0 = in[j];
        float v1 = in[j + 1];
        out[j] = v0 * cos - v1 * sin;
        out[j + 1] = v0 * sin + v1 * cos;
    }
}

void silu(float *in, float *out, size_t M, size_t N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            out[idx] = in[idx] * 1.0f / (1.0f + std::exp(-in[idx]));
        }
    }
}

void transpose(float *in, float *out, size_t M, size_t N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            out[i * M + j] = in[j * N + i];
        }
    }
}

static void print_sequence(float *tokens, size_t size) {
    for (int i = 0; i < size; i++) {
        printf(" %f", tokens[i]);
    }
    printf("\n");
}

// given an input sequence, generate compressed attention representation
void multi_headed_attention(
    float *in, size_t layer, size_t pos, TransformerState &state, LlamaConfig &config
) {
    // since this is decode, the input is actually something of the shape (1, hidden)
    // made a small mistake estimating the size of the weight offset, it should have been hidden *
    // hidden
    int qkv_weight_offset = layer * config.hidden_dim * config.hidden_dim;
    int kv_cache_pos_offset =
        layer * config.max_sequence_length * config.hidden_dim + pos * config.hidden_dim;

    // generate qkv weights
    mat_mul(
        in, state.wq + qkv_weight_offset, state.q_current, 1, config.hidden_dim, config.hidden_dim
    ); // (1, hidden) / (M, N)
    mat_mul(
        in,
        state.wk + qkv_weight_offset,
        state.k_cache + kv_cache_pos_offset,
        1,
        config.hidden_dim,
        config.hidden_dim
    ); // (1, hidden)
    mat_mul(
        in,
        state.wv + qkv_weight_offset,
        state.v_cache + kv_cache_pos_offset,
        1,
        config.hidden_dim,
        config.hidden_dim
    ); // (1, hidden)

    // apply rope to the new value only
    rope_with_pos(
        state.k_cache + kv_cache_pos_offset,
        state.k_cache + kv_cache_pos_offset,
        pos,
        config.hidden_dim,
        config.head_dim
    );
    rope_with_pos(state.q_current, state.q_current, pos, config.hidden_dim, config.head_dim);

    // apply attention
    for (int h = 0; h < config.n_heads; h++) {
        int full_seq_kv_cache_offset =
            layer * config.max_sequence_length * config.hidden_dim + h * config.head_dim;

        // first calculate the attention scores
        float *current_q_entry = state.q_current + h * config.head_dim;
        float *current_k_entry = state.k_cache + full_seq_kv_cache_offset;

        for (int t = 0; t <= pos; t++) {
            float score = 0.0f;
            for (int d = 0; d < config.head_dim; d++) {
                score += current_q_entry[d] * current_k_entry[t * config.hidden_dim + d];
            }
            score /= std::sqrt(config.head_dim);
            state.attention_scores[t] = score;
        }

        // NOTE: forgot to inlcude the + 1 here !!!
        softmax(state.attention_scores, state.softmax_attention_scores, 1, pos + 1);

        // then calculate the layer output
        float *current_v_entry = state.v_cache + full_seq_kv_cache_offset;
        float *current_attention_activations =
            state.post_attention_activations + h * config.head_dim;
        memset(current_attention_activations, 0, config.head_dim * sizeof(float));

        for (int t = 0; t <= pos; t++) {
            for (int d = 0; d < config.head_dim; d++) {
                current_attention_activations[d] +=
                    state.softmax_attention_scores[t] * current_v_entry[t * config.hidden_dim + d];
            }
        }
    }
}

// applies the swiglu
// llama has no biases
void feed_forward(float *in, size_t layer, TransformerState &state, LlamaConfig &config) {
    // apply rms norm
    rms_norm(
        in,
        state.pre_ffn_rms_norm + layer * config.hidden_dim,
        state.post_ffn_rms_norm_activations,
        1,
        config.hidden_dim
    );

    // same offset for all weights in this series
    int ffn_weight_offset = layer * config.hidden_dim * config.ffn_proj_up;

    mat_mul(
        state.post_ffn_rms_norm_activations,
        state.ffn_w1 + ffn_weight_offset,
        state.ffn_w1_activation,
        1,
        config.hidden_dim,
        config.ffn_proj_up
    );

    silu(state.ffn_w1_activation, state.ffn_w1_activation, 1, config.ffn_proj_up);

    mat_mul(
        state.post_ffn_rms_norm_activations,
        state.ffn_w3 + ffn_weight_offset,
        state.ffn_w2_activation,
        1,
        config.hidden_dim,
        config.ffn_proj_up
    );

    // write everything back to ffn_w1
    for (int i = 0; i < config.ffn_proj_up; i++) {
        state.ffn_w1_activation[i] = state.ffn_w1_activation[i] * state.ffn_w2_activation[i];
    }

    mat_mul(
        state.ffn_w1_activation,
        state.ffn_w2 + ffn_weight_offset,
        state.post_ffn_activation,
        1,
        config.ffn_proj_up,
        config.hidden_dim
    );

    // residual connection
    for (int i = 0; i < config.hidden_dim; i++) {
        state.post_ffn_activation[i] += in[i];
    }
}