# llama.cpp/examples/predicted

Demonstration of predicted output generation with recovery.

```bash
build/bin/llama-predicted \
    -m  fast_models/qwen3-32b-q8_0.gguf \
    -md fast_models/qwen3-1.7b-q8_0.gguf \
    -c 0 -ngl 99 --color \
    --sampling-seq k --top-k 1 -fa --temp 0.0 \
    -ngld 99 --draft-max 16 --draft-min 5 \
    -p "What is the capital of France?" \
    --draft-text "The capital of France is Paris." \
    --log-verbose
```

```bash
build/bin/llama-predicted \
    -m  fast_models/qwen3-32b-q4_0.gguf \
    -c 0 -ngl 99 -fa --temp 0.0 \
    -p "remove the debug statement from the following code. just print out the modified code, no description needed:\n\n$(cat examples/predicted/test.txt)" \
    --draft-text "$(cat examples/predicted/test.txt)"
```

# Algorithm

`common_sampler_sample_and_accept_n`: Runs sampler on draft logits and accepts matches. If the full draft is accepted, it runs another sample to generate the last token. If the draft is empty, it simply runs the sampler once.

`last_id`: This tracks the last token for which logits have been computed. This goes in as the first token for the next batch.

`n_past`: This is the cumulative number of tokens for which logits have been computed.
