# test patch sampler

# take input code from stdin and use prompt specified as argument

MODEL_DIR="../../fast_models"
MODEL="$MODEL_DIR/gemma-3-12b-it-q8_0.gguf"
SYSTEM="You are an assistant that makes changes to code. You are given a code snippet and a prompt. You need to make the changes to the code snippet to satisfy the prompt. You need to return the modified code snippet."
PROMPT="$1"
CODE=$(cat)

../../build/bin/llama-predicted -m $MODEL -c 0 -ngl 99 -fa --color --system-prompt "$SYSTEM" --prompt "$PROMPT\n\n$CODE" --draft-text "$CODE" --draft-min 5 --draft-max 32
