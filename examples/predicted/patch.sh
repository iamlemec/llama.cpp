# test patch sampler

# take input code from stdin and use prompt specified as argument

MODEL_DIR="../../fast_models"
MODEL="$MODEL_DIR/gemma-3-12b-it-q8_0.gguf"
PROMPT="$1"
CODE=$(cat)

../../build/bin/llama-predicted -m $MODEL -c 0 -ngl 99 -fa --color -p "$PROMPT\n\n$CODE" --draft-text "$CODE" --draft-min 3
