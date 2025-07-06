# llama.cpp/examples/predicted

Demonstration of predicted output generation with recovery. See `patch.sh` for an example and `lookup.sh` for comparison with lookup decoding.

# Todo

Use `common_ngram_hash_function` to index the draft, mapping from n-grams to lists of positions. Then track `draft_pos` so that we can reject positions that have been dropped already.

# Algorithm

- `n_past`: cumulative number of tokens sampled (including the prompt)
- `use_draft`: whether we're using the draft or not
- `id_last`: the last token that has been sampled
- `batch_idx`: current index in the current batch (-1 means we need to decode a new batch)
