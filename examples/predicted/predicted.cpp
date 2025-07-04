#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "chat.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <deque>

int64_t draft_match_index(
    const std::deque<llama_token> & draft_tokens,
    const std::deque<llama_token> & diffs_tokens,
    int64_t match_length
);

std::string common_get_prompt(const common_params & params, const struct common_chat_templates * chat_templates);

// this looks for diffs that can start anywhere in draft
int64_t draft_match_index(
    const std::deque<llama_token> & draft_tokens,
    const std::deque<llama_token> & diffs_tokens,
    int64_t match_length
) {
    // get array sizes
    int64_t draft_size = (int64_t) draft_tokens.size();
    int64_t diffs_size = (int64_t) diffs_tokens.size();
    int64_t common_size = std::min(draft_size, diffs_size);

    // if the arrays are too short, return -1
    if (common_size < match_length) {
        return -1;
    }

    // check for matches starting from end
    for (int64_t i = 0; i < draft_size - match_length + 1; ++i) {
        int64_t j = 0;
        for (; j < match_length; ++j) {
            if (diffs_tokens[diffs_size - match_length + j] != draft_tokens[i + j]) {
                break;
            }
        }
        if (j == match_length) {
            return i + match_length;
        }
    }

    // no match found
    return -1;
}

// chat template support from main.cpp
std::string common_get_prompt(const common_params & params, const struct common_chat_templates * chat_templates) {
    std::vector<common_chat_msg> chat_msgs;

    if (params.enable_chat_template) {
        // format the system prompt (will use template default if empty)
        if (!params.system_prompt.empty()) {
            common_chat_msg new_msg;
            new_msg.role = "system";
            new_msg.content = params.system_prompt;
            common_chat_format_single(chat_templates, chat_msgs, new_msg, false, params.use_jinja);
            chat_msgs.push_back(new_msg);
        }

        // format and append the user prompt
        if (!params.prompt.empty()) {
            common_chat_msg new_msg;
            new_msg.role = "user";
            new_msg.content = params.prompt;
            common_chat_format_single(chat_templates, chat_msgs, new_msg, true, params.use_jinja);
            chat_msgs.push_back(new_msg);
        }

        // apply the chat template
        if (!params.system_prompt.empty() || !params.prompt.empty()) {
            common_chat_templates_inputs inputs;
            inputs.use_jinja = params.use_jinja;
            inputs.messages = chat_msgs;
            inputs.add_generation_prompt = !params.prompt.empty();
            return common_chat_templates_apply(chat_templates, inputs).prompt;
        }
    }

    // otherwise use the prompt as is
    return params.prompt;
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PREDICTED)) {
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

    // tokenize the prompt
    auto chat_templates = common_chat_templates_init(model, params.chat_template);
    std::string prompt = common_get_prompt(params, chat_templates.get());
    std::vector<llama_token> inp = common_tokenize(ctx, prompt, true, true);

    LOG_DBG("prompt: %s\n", prompt.c_str());

    if (inp.empty()) {
        LOG_ERR("%s: the prompt is empty\n", __func__);
        return 1;
    }

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

    const int draft_min = params.speculative.n_min;
    const int draft_max = params.speculative.n_max;

    // draft text to use for prediction
    std::string draft_text = params.speculative.text;
    std::vector<llama_token> draft_tokens = common_tokenize(ctx, draft_text, false, true);

    // use deque for front popping
    std::deque<llama_token> draft(draft_tokens.begin(), draft_tokens.end());
    const int n_draft = draft.size();

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);

    // eval the prompt
    const int n_prompt = inp.size();
    llama_decode(ctx, llama_batch_get_one(inp.data(), n_prompt - 1));

    const auto t_enc_end = ggml_time_us();
    const auto t_dec_start = ggml_time_us();

    // track stats
    int n_accept = 0;
    int n_logits = 0;

    // generation state
    int n_past = n_prompt - 1;
    bool use_draft = !draft.empty();
    llama_token id_last = inp.back();
    int batch_idx = -1;

    // tokens: all prompt + generated tokens
    std::vector<llama_token> tokens(inp);
    tokens.reserve(llama_n_ctx(ctx));

    // prefix: tokens since last match
    std::deque<llama_token> diffs;

    // batch for evaluating the model
    const int draft_size0 = std::min(draft_max, (int) llama_n_batch(ctx) - 1);
    llama_batch batch = llama_batch_init(draft_size0 + 1, 0, 1);

    // loop through (successfully) generated tokens
    for (;; n_past++) {
        // we've hit the generation limit
        if (params.n_predict >= 0 && n_past >= params.n_predict) {
            break;
        }

        std::vector<llama_token> draft_copy(draft.begin(), draft.end());
        std::vector<llama_token> diffs_copy(diffs.begin(), diffs.end());
        LOG_DBG("n_past: %d, use_draft: %d, batch_idx: %d\n", n_past, use_draft, batch_idx);
        LOG_DBG("id_last: %s\n", common_token_to_piece(ctx, id_last).c_str());
        LOG_DBG("Current draft: %s\n", string_from(ctx, draft_copy).c_str());
        LOG_DBG("Current diffs: %s\n", string_from(ctx, diffs_copy).c_str());

        // look for deletions in draft
        const int64_t match_idx = draft_match_index(draft, diffs, draft_min);
        if (match_idx >= 0) {
            draft.erase(draft.begin(), draft.begin() + match_idx);
            diffs.clear();
            use_draft = true;
            batch_idx = -1;
            n_accept += draft_min;
        }

        LOG_DBG("match_idx: %d\n", (int) match_idx);

        // generate new logits if needed
        if (use_draft) {
            if (batch_idx == -1 || batch_idx >= batch.n_tokens - 1) {
                const int draft_size = std::min(draft_size0, (int) draft.size());

                // decode from draft
                common_batch_clear(batch);
                common_batch_add(batch, id_last, n_past, { 0 }, true);
                for (int i = 0; i < draft_size; ++i) {
                    common_batch_add(batch, draft[i], n_past + 1 + i, { 0 }, true);
                }
                llama_decode(ctx, batch);

                // update generation state
                batch_idx = 0;
                n_logits += draft_size + 1;

                LOG_DBG("decoded batch [idx: %d]: %s\n", batch_idx, string_from(ctx, batch).c_str());
            } else {
                batch_idx++;
            }
        } else {
            // decode from sampled token
            common_batch_clear(batch);
            common_batch_add(batch, id_last, n_past, { 0 }, true);
            llama_decode(ctx, batch);

            // update generation state
            batch_idx = 0;
            n_logits++;
        }

        // sample from the target model
        id_last = common_sampler_sample(smpl, ctx, batch_idx);
        common_sampler_accept(smpl, id_last, true);
        tokens.push_back(id_last);

        LOG_DBG("sampled token: %s\n", common_token_to_piece(ctx, id_last).c_str());

        // check for EOS
        if (llama_vocab_is_eog(vocab, id_last)) {
            LOG_DBG("EOS\n");
            break;
        }

        // did the sampled token match the draft?
        if (use_draft) {
            if (id_last == draft.front()) {
                draft.pop_front();
                n_accept += 1;
            } else {
                use_draft = false;
                llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past + 1, -1);
            }
        }

        // process the accepted tokens and update contexts
        const std::string token_str = common_token_to_piece(ctx, id_last);
        if (params.use_color && use_draft) {
            LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
        } else {
            LOG("%s", token_str.c_str());
        }

        // if we're not in draft mode, update diffs tokens
        if (!use_draft) {
            diffs.push_back(id_last);
            if ((int) diffs.size() > draft_min) {
                diffs.pop_front();
            }
        }

        // are we out of draft tokens?
        if (draft.empty()) {
            use_draft = false;
            llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past + 1, -1);
        }

        LOG_DBG("\n\n");
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_prompt,   (t_enc_end - t_enc_start) / 1e6f, n_prompt / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_past, (t_dec_end - t_dec_start) / 1e6f, n_past  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_logits  = %d\n", n_logits);
    LOG_INF("n_accept  = %d\n", n_accept);

    LOG_INF("\n");
    LOG_INF("performance:\n\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
