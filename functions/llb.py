import stan
from prompts.llb import LLBPrompting
from llms import BaseLLM


def hybrid(data_point: dict, algo_n: int, algo_k: int, cache_folder: str, llb_prompting: LLBPrompting, llm: BaseLLM, **kwargs):
    # algorithm 9 the hybrid MCMC / VI / SNIS algorithm in "Large Language Bayes"
    assert 'id' in data_point
    assert 'data_path' in data_point
    assert 'instruction' in data_point
    assert 'target_variable' in data_point and len(data_point['target_variable']) == 1
    messages = llb_prompting.build_prompt(data_point['instruction'])
    # set n in kwargs to algo_n
    kwargs['n'] = algo_n
    completion = llm.chat_completion(prompt=messages, **kwargs)
    batched_text = llm.text_in_completion(completion=completion)    # (batch=1, algo_n)
    for text in batched_text[0]:    # the first batch, here we assume batch size is 1
        parsed_blocks, status = llb_prompting.parse_response(response=text)
        if not status:
            continue
        _stan_model_path = stan.save_model_file(cache_folder, 'sampled_model', parsed_blocks['stan_model'])
        fit = stan.fit_model(model_path=_stan_model_path, data_json_path=data_point['data_path'])
        assert algo_k <= len(fit.stan_variable(data_point['target_variable']))
        sampled_target_vals = fit.stan_variable(data_point['target_variable'])[:algo_k]
        q_mean = sampled_target_vals.mean(axis=0)
        q_cov = sampled_target_vals.var(axis=0)
        # TODO: implementation of IW-ELBO here

        pass
