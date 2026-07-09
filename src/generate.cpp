#include "llama.h"
#include <string>

int main() {
    Tokenizer t;
    t.build_tokenizer(std::string("tokenizer.bin"), llama_config.vocab_size);

    TransformerState state(llama_config);
}