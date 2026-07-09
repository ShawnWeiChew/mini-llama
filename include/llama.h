#ifndef CONFIG_H_
#define CONFIG_H_

#include <memory>
#include <string>
#include <vector>

struct LlamaConfig {
    int hidden_dim = 288;
    int ffn_proj_up = 768; // (((2 / 3 * (4 * 288)) + 255) // 256) * 256
    int n_layers = 6;
    int n_heads = 6;
    int head_dim = hidden_dim / n_heads;
    int max_sequence_length = 256;
    int vocab_size = 32000;

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

struct TokenIndex {
    std::string str;
    int id;
};

struct Tokenizer {
    std::vector<std::unique_ptr<char[]>> vocab;
    // this gives the priority for the tokens to be merged later
    // ideally, you would try to merge stuff from the vocab before merging anything from
    // the single byte characters later
    std::vector<float> vocab_scores;
    std::unique_ptr<TokenIndex[]> sorted_vocab;

    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];

    void build_tokenizer(const std::string &tokenizer_path, int vocab_size);
    void encode(
        const std::string &test, const int8_t &bos, const int8_t &eos, int *tokens, int &n_tokens
    );
    std::string decode(int prev_token, int token);

  private:
    int str_lookup(std::string token_to_find);
};

#endif