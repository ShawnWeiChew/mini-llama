#include "llama.h"
#include "ops.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

static void print_sequence(float *tokens, size_t size) {
    for (int i = 0; i < size; i++) {
        printf(" %f", tokens[i]);
    }
    printf("\n");
}

int main() {
    Tokenizer tokenizer;
    tokenizer.build_tokenizer(std::string("tokenizer.bin"), llama_config.vocab_size);

    TransformerState state(llama_config);

    // generate the embeddings, and off we go!
    char prompt[] = "There once ";

    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(
        sizeof(int) * (strlen(prompt) + 3)
    ); // this is to make space for 0, BOS, EOS (in the worst case)

    tokenizer.encode(prompt, 1, 0, prompt_tokens, num_prompt_tokens);

    // used to track the next token in the sequence
    int pos = 0;
    int next;
    int token = prompt_tokens[0];
    float token_embedding[288] = {};

    // start timer here and end at the end of the while loop, then divide by 256 to get tokens/s
    auto start = std::chrono::steady_clock::now();
    while (pos < llama_config.max_sequence_length) {
        // first create the token embedding
        float *embedding = state.token_embedding_table + token * llama_config.hidden_dim;
        memcpy(token_embedding, embedding, llama_config.hidden_dim * sizeof(float));

        transformer_forward(state, llama_config, token_embedding, pos);

        // do a final rms norm, then a projection through the vocab layer
        rms_norm(
            state.post_ffn_activation,
            state.final_rms_norm,
            state.post_final_rms_norm,
            1,
            llama_config.hidden_dim
        );

        // project through the linear layer
        mat_mul(
            state.post_final_rms_norm,
            state.final_linear,
            state.logits,
            1,
            llama_config.hidden_dim,
            llama_config.vocab_size
        );

        // still processing the input
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample_argmax(state.logits, llama_config.vocab_size);
        }
        pos++;

        char *piece = tokenizer.decode(token, next);
        printf("%s", piece);
        fflush(stdout);
        token = next;
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf(
        "\n\nElapsed time: %.4f s, speed: %.2f tokens/s\n", elapsed.count(), pos / elapsed.count()
    );
}