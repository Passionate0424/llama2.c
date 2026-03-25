#define build_transformer file_build_transformer
#define free_transformer file_free_transformer
#define build_tokenizer file_build_tokenizer
#define error_usage file_error_usage
#define main file_main
#include "runq.c"
#undef build_transformer
#undef free_transformer
#undef build_tokenizer
#undef error_usage
#undef main

#include "embedded/stories260K/stories260K_q80.h"
#include "embedded/stories260K/tok512.h"

static void read_embedded_bytes(
    const unsigned char** cursor,
    const unsigned char* end,
    void* out,
    size_t len,
    const char* what
) {
    if ((size_t)(end - *cursor) < len) {
        fprintf(stderr, "embedded asset truncated while reading %s\n", what);
        exit(EXIT_FAILURE);
    }
    memcpy(out, *cursor, len);
    *cursor += len;
}

static void build_transformer_from_memory(
    Transformer* t,
    const unsigned char* checkpoint,
    size_t checkpoint_size
) {
    const int header_size = 256;
    unsigned char* checkpoint_copy = NULL;
    const unsigned char* cursor = NULL;
    const unsigned char* end = NULL;
    uint32_t magic_number = 0;
    int version = 0;
    uint8_t shared_classifier = 0;
    int group_size = 0;

    if (checkpoint_size < (size_t)header_size) {
        fprintf(stderr, "embedded checkpoint is too small: %zu bytes\n", checkpoint_size);
        exit(EXIT_FAILURE);
    }

    checkpoint_copy = (unsigned char*)malloc(checkpoint_size);
    if (checkpoint_copy == NULL) {
        fprintf(stderr, "malloc failed for embedded checkpoint copy\n");
        exit(EXIT_FAILURE);
    }
    memcpy(checkpoint_copy, checkpoint, checkpoint_size);
    cursor = checkpoint_copy;
    end = checkpoint_copy + checkpoint_size;

    read_embedded_bytes(&cursor, end, &magic_number, sizeof(magic_number), "magic number");
    if (magic_number != 0x616b3432) {
        fprintf(stderr, "bad embedded checkpoint magic number: 0x%x\n", magic_number);
        exit(EXIT_FAILURE);
    }

    read_embedded_bytes(&cursor, end, &version, sizeof(version), "version");
    if (version != 2) {
        fprintf(stderr, "bad embedded checkpoint version %d, need version 2\n", version);
        exit(EXIT_FAILURE);
    }

    read_embedded_bytes(&cursor, end, &t->config, sizeof(Config), "config");
    read_embedded_bytes(
        &cursor,
        end,
        &shared_classifier,
        sizeof(shared_classifier),
        "shared classifier flag"
    );
    read_embedded_bytes(&cursor, end, &group_size, sizeof(group_size), "group size");
    if (group_size <= 0) {
        fprintf(stderr, "bad embedded checkpoint group size %d\n", group_size);
        exit(EXIT_FAILURE);
    }

    GS = group_size;
    t->fd = -1;
    t->data = (float*)checkpoint_copy;
    t->file_size = (ssize_t)checkpoint_size;

    memory_map_weights(
        &t->weights,
        &t->config,
        (void*)(checkpoint_copy + header_size),
        shared_classifier
    );
    malloc_run_state(&t->state, &t->config);
}

static void build_tokenizer_from_memory(
    Tokenizer* t,
    const unsigned char* tokenizer_data,
    size_t tokenizer_size,
    int vocab_size
) {
    const unsigned char* cursor = tokenizer_data;
    const unsigned char* end = tokenizer_data + tokenizer_size;
    int len = 0;

    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    read_embedded_bytes(
        &cursor,
        end,
        &t->max_token_length,
        sizeof(t->max_token_length),
        "max token length"
    );

    for (int i = 0; i < vocab_size; i++) {
        read_embedded_bytes(
            &cursor,
            end,
            t->vocab_scores + i,
            sizeof(float),
            "token score"
        );
        read_embedded_bytes(&cursor, end, &len, sizeof(len), "token length");
        if (len < 0 || (size_t)(end - cursor) < (size_t)len) {
            fprintf(stderr, "embedded tokenizer token %d has invalid length %d\n", i, len);
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char*)malloc((size_t)len + 1);
        memcpy(t->vocab[i], cursor, (size_t)len);
        t->vocab[i][len] = '\0';
        cursor += len;
    }
}

void build_transformer(Transformer* t, char* checkpoint_path) {
    (void)checkpoint_path;
    build_transformer_from_memory(
        t,
        stories260K_q80_bin,
        (size_t)stories260K_q80_bin_len
    );
}

void free_transformer(Transformer* t) {
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if (t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    if (t->data != MAP_FAILED) { free(t->data); }
    free_run_state(&t->state);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    (void)tokenizer_path;
    build_tokenizer_from_memory(t, tok512_bin, (size_t)tok512_bin_len, vocab_size);
}

void error_usage() {
    fprintf(stderr, "Usage:   runq_embedded [options]\n");
    fprintf(stderr, "Example: runq_embedded -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "generate";
    char *system_prompt = NULL;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0f) temperature = 0.0f;
    if (topp < 0.0f || 1.0f < topp) topp = 0.9f;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_transformer(&transformer, NULL);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, NULL, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
