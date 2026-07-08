#ifndef CONFIG_H_
#define CONFIG_H_

struct LlamaConfig {
    int hidden_dim = 4096;
    int ffn_proj_up = 11008; // (((2 / 3 * (4 * 4096)) + 255) // 256) * 256
    int n_layers = 32;
    int n_heads = 32;
    int head_dim = hidden_dim / n_heads;

} inline llama_config;

struct TransformerState {
    // attention states
    float *wq;
    float *wk;
    float *wv;

    // ffn states
    float *ffn_activation;
} inline transformer_state;

#endif