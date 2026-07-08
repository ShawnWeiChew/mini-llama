#ifndef CONFIG_H_
#define CONFIG_H_

struct LlamaConfig {
    int hidden_dim = 288;
    int ffn_proj_up = 768; // (((2 / 3 * (4 * 288)) + 255) // 256) * 256
    int n_layers = 6;
    int n_heads = 6;
    int head_dim = hidden_dim / n_heads;
    int max_sequence_length = 256;

} inline llama_config;

struct TransformerState {
    // weights
    // pre-attention rmsnorm element affine
    float *pre_attention_rms_norm; // (layer, dim)

    // attention states
    float *wq; // (layer, hidden_dim, hidden_dim)
    float *wk;
    float *wv;
    float *wo; // (layer, hidden_dim, hidden_dim)

    // pre ffn rmsnorm element affine
    float *pre_ffn_rms_norm; // (layer, hidden_dim)
    // ffn states
    float *ffn_w1; // up project
    float *ffn_w2; // another up project within the swiglu
    float *ffn_w3; // down project

    float *final_rms_norm;

    // intermediate activations
    float *q_current;

    float *k_cache; // (layer, seq, dim)
    float *v_cache; // (layer, seq, dim)

    float *attention_scores;         // (1, seq)
    float *softmax_attention_scores; // (1, seq)

    float *post_attention_activations; // (seq, hidden)

    float *pre_ffn_rms_norm_activations;

    float *ffn_w1_activation;
    float *ffn_w2_activation;

    float *post_ffn_activation; // (seq, hidden)

} inline transformer_state;

#endif