
#include "../include/ops.h"
#include "../include/llama.h"
#include <cassert>
#include <cmath>
#include <cstring>
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

void rope_with_pos(float *in, float *out, int pos, size_t N) {
    assert(N % 2 == 0 && "Inner RoPE dimension should be multiple of 2");
    for (int j = 0; j < N; j += 2) { // this refers to the pointer within the hidden dimension
        float theta = 1.0f / std::pow(10000, static_cast<float>(j) / N);
        // TODO: verify that this absolute_position thing is correct
        // this should probably be for the latest entry in the forward pass
        // but I am not sure how it plays into this
        // Note also that I dont really know what the in and out should be
        float frequency = pos * theta;

        float sin = std::sin(frequency);
        float cos = std::cos(frequency);

        int idx = j; // this refers to the even index, while +1 refers to the odd index
        out[idx] = in[idx] * cos - in[idx + 1] * sin;
        out[idx + 1] = in[idx] * sin + in[idx + 1] * cos;
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

// given an input sequence, generate compressed attention representation
void multi_headed_attention(
    float *in, size_t layer, size_t pos, TransformerState &state, LlamaConfig &config
) {
    // since this is decode, the input is actually something of the shape (1, hidden)
    // made a small mistake estimating the size of the weight offset, it should have been hidden *
    // hidden
    int qkv_weight_offset = layer * config.hidden_dim * config.hidden_dim;
    int kv_cache_pos_offset = layer * config.max_sequence_length + pos * config.hidden_dim;

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
    rope_with_pos(state.k_cache + kv_cache_pos_offset, state.k_cache, pos, config.hidden_dim);
    rope_with_pos(state.q_current, state.q_current, pos, config.hidden_dim);

    // apply attention
    for (int h = 0; h < config.n_heads; h++) {
        int full_seq_kv_cache_offset =
            layer * config.max_sequence_length * config.hidden_dim + h * config.head_dim;

        memset(state.attention_scores, 0, pos);
        // first calculate the attention scores
        float *current_k_entry = state.k_cache + full_seq_kv_cache_offset;
        float *current_q_entry = state.q_current + h * config.head_dim;
        for (int t = 0; t <= pos; t++) {
            for (int d = 0; d < config.head_dim; d++) {
                // find the position of the value in the v_cache
                state.attention_scores[t] +=
                    current_q_entry[d] * current_k_entry[t * config.head_dim + d];
            }
        }

        softmax(state.attention_scores, state.softmax_attention_scores, 1, pos);

        // then calculate the layer output
        float *current_v_entry = state.v_cache + full_seq_kv_cache_offset;
        float *current_attention_activations =
            state.post_attention_activations + h * config.head_dim;
        memset(current_attention_activations, 0, config.head_dim);

        for (int t = 0; t <= pos; t++) {
            for (int d = 0; d < config.head_dim; d++) {
                current_attention_activations[d] +=
                    state.softmax_attention_scores[t] * current_v_entry[t * config.head_dim + d];
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
        state.pre_ffn_rms_norm,
        state.pre_ffn_rms_norm_activations + layer * config.hidden_dim,
        1,
        config.hidden_dim
    );

    // same offset for all weights in this series
    int ffn_weight_offset = layer * config.hidden_dim * config.ffn_proj_up;

    mat_mul(
        in,
        state.ffn_w1 + ffn_weight_offset,
        state.ffn_w1_activation,
        1,
        config.hidden_dim,
        config.ffn_proj_up
    );
    silu(state.ffn_w1_activation, state.ffn_w1_activation, 1, config.ffn_proj_up);

    mat_mul(
        in,
        state.ffn_w2 + ffn_weight_offset,
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
        state.ffn_w3,
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