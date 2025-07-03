# llama.cpp/examples/predicted

Demonstration of predicted output generation with recovery.

# Algorithm

`common_sampler_sample_and_accept_n`: Runs sampler on draft logits and accepts matches. If the full draft is accepted, it runs another sample to generate the last token. If the draft is empty, it simply runs the sampler once.

`last_id`: This tracks the last token for which logits have been computed. This goes in as the first token for the next batch.

`n_past`: This is the cumulative number of tokens for which logits have been computed.
