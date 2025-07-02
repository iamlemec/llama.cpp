#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

const int min_match_length = 3; // minimum number of tokens to match

std::pair<int64_t, int64_t> common_substring_match(
    const llama_tokens & draft_tokens,
    const llama_tokens & prefix_tokens,
    int64_t min_match_length
);

// TODO: this only accounts for insertions, but doesn't look for deletions yet
std::pair<int64_t, int64_t> common_substring_match(
    const llama_tokens & draft_tokens,
    const llama_tokens & prefix_tokens,
    int64_t min_match_length
) {
    // get array sizes
    int64_t draft_size = (int64_t) draft_tokens.size();
    int64_t prefix_size = (int64_t) prefix_tokens.size();

    // check for matches starting from beginning of draft
    for (int64_t i = 0; i < prefix_size - min_match_length + 1; ++i) {
        int64_t j = 0;
        for (; j < draft_size; ++j) {
            if (
                (i + j >= prefix_size) ||
                (prefix_tokens[i + j] != draft_tokens[j])
            ) {
                break;
            }
        }
        if (j >= min_match_length) {
            return std::pair<int64_t, int64_t>(j, i + j);
        }
    }

    return std::pair<int64_t, int64_t>(0, 0);
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = NULL;
    llama_context * ctx = NULL;

    // load the target model
    common_init_result llama_init = common_init_from_params(params);

    model = llama_init.model.get();
    ctx   = llama_init.context.get();

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx, params.prompt, true, true);

    if (llama_n_ctx(ctx) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx));

        return 1;
    }

    if (llama_n_batch(ctx) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx, id).c_str());
    }

    int n_predict = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool use_draft = true;
    bool has_eos = false;

    // draft text to use for prediction
    std::string draft_text = params.speculative.text;
    std::vector<llama_token> draft = common_tokenize(ctx, draft_text, true, true);
    const int n_draft = draft.size();

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);

    // eval the prompt (without the last token)
    llama_decode(ctx, llama_batch_get_one(inp.data(), inp.size() - 1));

    // tokens: all prompt + generated tokens
    llama_tokens tokens(inp);
    tokens.reserve(llama_n_ctx(ctx));

    // prefix: tokens since last match
    llama_tokens prefix;
    prefix.reserve(llama_n_ctx(ctx));

    // n_past is the number of tokens in the target context
    // note: we always need at least one token to evaluate from before
    llama_token id_last = inp.back();
    int n_past = inp.size() - 1;

    // batch for evaluating the model
    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);

    const auto t_enc_end = ggml_time_us();
    const auto t_dec_start = ggml_time_us();

    while (true) {
        // check if the draft matches the prefix and chop if needed
        std::pair<int64_t, int64_t> match = common_substring_match(draft, prefix, min_match_length);
        if (match.first >= 0) {
            draft.erase(draft.begin(), draft.begin() + match.first);
            prefix.erase(prefix.begin(), prefix.begin() + match.second);
            n_accept += match.first;
            use_draft = true;
        } else {
            use_draft = false;
        }

        //LOG_DBG("draft: %s\n", string_from(ctx_dft, draft).c_str());

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past++, { 0 }, true);

        // evaluate the target model on draft[0..n_batch-1]
        {
            if (use_draft) {
                const int n_batch1 = std::min((int) llama_n_batch(ctx), (int) draft.size());
                for (int i = 0; i < n_batch1; ++i) {
                    common_batch_add(batch, draft[i], n_past + i, { 0 }, true);
                }
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx, batch).c_str());

            llama_decode(ctx, batch);
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        n_past    += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            tokens.push_back(id_last);
            prefix.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_accept  = %d\n", n_accept);

    LOG_INF("\n");
    LOG_INF("performance:\n\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
