# llama.cpp/examples/predicted

Demonstration of predicted output generation with recovery.

```bash
build/bin/llama-predicted \
    -m  fast_models/qwen3-32b-q8_0.gguf \
    -md fast_models/qwen3-1.7b-q8_0.gguf \
    -c 0 -ngl 99 --color \
    --sampling-seq k --top-k 1 -fa --temp 0.0 \
    -ngld 99 --draft-max 16 --draft-min 5 \
    -p "What is the capital of France?"
    --text "The capital of France is Paris."
```
