#include "../include/llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

void Tokenizer::build_tokenizer(const std::string &tokenizer_path, int vocab_size) {
    vocab.resize(vocab_size);
    vocab_scores.resize(vocab_size);
    this->vocab_size = vocab_size;

    // fill up individual bytes
    for (int i = 0; i < 256; i++) {
        byte_pieces[i * 2] = static_cast<unsigned char>(i);
        byte_pieces[i * 2 + 1] = '\0';
    }

    // now actually load the weights
    std::ifstream file(tokenizer_path, std::ios::binary);
    if (!file) {
        std::cerr << "Could not open file at " << tokenizer_path << std::endl;
        std::exit(1);
    }

    file.read(reinterpret_cast<char *>(&max_token_length), sizeof(int));
    int len = 0;

    // file format is length first
    // then followed by the tokens that make it up
    for (int i = 0; i < vocab_size; i++) {
        file.read(reinterpret_cast<char *>(&vocab_scores[i]), sizeof(float));
        file.read(reinterpret_cast<char *>(&len), sizeof(int));
        vocab[i] = std::make_unique<char[]>(len + 1);
        file.read(vocab[i].get(), len);
        vocab[i][len] = '\0';
    }

    file.close();
}

static bool compare_tokens(const TokenIndex &a, const TokenIndex &b) {
    return std::lexicographical_compare(
        a.str.begin(),
        a.str.end(),
        b.str.begin(),
        b.str.end(),
        [](char lhs, char rhs) {
            return static_cast<unsigned char>(lhs) < static_cast<unsigned char>(rhs);
        }
    );
}

void Tokenizer::encode(
    const std::string &text, const int8_t &bos, const int8_t &eos, int *tokens, int &n_tokens
) {
    if (text.empty()) {
        std::cerr << "Cannot encode null text" << std::endl;
        std::exit(1);
    }

    if (!sorted_vocab) {
        sorted_vocab = std::make_unique<TokenIndex[]>(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            sorted_vocab[i].str = std::string(vocab[i].get());
            sorted_vocab[i].id = i;
        }

        // puts them in order for a binary search later
        std::sort(sorted_vocab.get(), sorted_vocab.get() + vocab_size, compare_tokens);
    }

    // add BOS sequence and a dummy prefix, " " <- this better matches what sentence piece does
    // during training

    // BOS is encoded as token=1
    if (bos)
        tokens[n_tokens++] = 1;

    tokens[n_tokens++] = str_lookup(" ");

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // the goal of this step is to add the individual encoding for each character
    // this is to then prepare for a merge that would come later
    std::string str_buffer;

    // we do that by inspecting the individual bytes for continuation characters
    // if there are, then we concat the input. Otherwise, we just write the individual bytes
    // I first thought that writing individual bytes forgoes accuracy, but that is not true
    // the only thing that we end up losing is token efficiency
    // if a token cannot be found in the vocab, the consequence is that you just use more individual
    // byte tokens to represent it
    for (const char *c = text.c_str(); *c != '\0'; c++) {
        // continuation character starts with 10xxxxxx
        if ((*c & 0b11000000) != 0b10000000) {
            str_buffer.clear();
        }

        str_buffer.push_back(*c);

        // continue this loop if the next character is part of the sequence
        // and the character size has yet to exceed 4 bytes
        if ((*(c + 1) & 0b11000000) == 0b10000000 && str_buffer.size() < 4) {
            continue;
        }

        // try to present the encoding for this token into the token buffer
        int token_idx = str_lookup(str_buffer);
        if (token_idx != -1) {
            tokens[n_tokens++] = token_idx;
        } else {
            for (char ch : str_buffer) {
                // cast to unsigned char is crucial to handle negative values correctly
                // ascii characters start at position 3
                tokens[n_tokens++] = static_cast<unsigned char>(ch) + 3;
            }
        }
        str_buffer.clear(); // protect against a sequence of stray UTF8 continuation bytes
    }

    // now we try to merge the characters

    char merge_buffer[2 * max_token_length + 1 + 2];
    while (1) {
        float best_score = -1e10;
        int merge_result_idx = -1;
        int target_idx = -1;

        for (int i = 0; i < (n_tokens - 1); i++) {
            // create the merge candidate
            snprintf(
                merge_buffer,
                2 * max_token_length + 1 + 2,
                "%s%s",
                vocab[tokens[i]].get(),
                vocab[tokens[i + 1]].get()
            );

            int merged_idx = str_lookup(merge_buffer);

            // merge only the best token each time
            if (merged_idx != -1 && vocab_scores[merged_idx] > best_score) {
                best_score = vocab_scores[merged_idx];
                merge_result_idx = merged_idx;
                target_idx = i;
            }
        }

        // if target idx was never changed, it means that there are no more tokens to merge
        if (target_idx == -1) {
            break;
        }

        // move all the other tokens over
        tokens[target_idx] = merge_result_idx;
        for (int i = target_idx + 1; i < (n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }

        n_tokens--;
    }

    if (eos)
        tokens[n_tokens++] = 2;
}

char *Tokenizer::decode(int prev_token, int token) {
    char *piece = vocab[token].get();

    // remove any whitespace after BOS token
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }

    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)byte_pieces + byte_val * 2;
    }

    return piece;
}

int Tokenizer::str_lookup(std::string str) {
    TokenIndex tok = {.str = str}; // acts as the key to search for

    auto it =
        std::lower_bound(sorted_vocab.get(), sorted_vocab.get() + vocab_size, tok, compare_tokens);

    // If we didn't reach the end and the string matches
    if (it != (sorted_vocab.get() + vocab_size) && it->str == str) {
        return it->id;
    }

    return -1; // Not found
}