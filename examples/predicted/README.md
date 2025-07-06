# llama.cpp/examples/predicted

Demonstration of predicted output generation with recovery. See `patch.sh` for an example and `specul.sh` for comparison with draft models.

# Todo

Use `common_ngram_hash_function` to index the draft, mapping from n-grams to lists of positions. Then track `draft_pos` so that we can reject positions that have been dropped already.

# Algorithm

- `n_past`: cumulative number of tokens for which logits have been computed
- `use_draft`: whether we're using the draft or not
- `id_last`: the last token for which logits have been computed
- `batch_idx`: current index in the current batch (-1 means we need to decode a new batch)
