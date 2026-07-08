#ifndef CONFIG_H_
#define CONFIG_H_

struct LlamaConfig {
    int hidden_dim = 4096;
    int n_layers = 32;
    int n_heads = 32;
    int head_dim = hidden_dim / n_heads;

} inline llama_config;

#endif