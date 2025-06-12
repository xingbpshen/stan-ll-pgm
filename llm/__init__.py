from vllm import LLM, SamplingParams
from util import gpu_compute_capability
from prompt import PromptingMethod


def build_conversation(message: str):
    conversation = [
        {'role': 'user',
         'content': message}
    ]
    return conversation


class MyLLM:
    def __init__(self,
                 model: str,
                 gpu_mem_utilization: float,
                 num_gpu: int,
                 dtype: str = None,
                 allow_media_path: str = None,
                 use_vllm: bool = True):
        assert use_vllm
        if dtype is None:
            dtype = gpu_compute_capability()
        self.llm = LLM(model=model,
                       gpu_memory_utilization=gpu_mem_utilization,
                       tensor_parallel_size=num_gpu,
                       allow_media_path=allow_media_path,
                       dtype=dtype)

    def chat(self,
             prompting_method: PromptingMethod,
             query_instance: str,
             temperature: float,
             top_k: float,
             top_p: int,
             max_completion_tokens: int,
             num_completions: int = 1):
        sampling_params = SamplingParams(temperature=temperature,
                                         top_k=top_k,
                                         top_p=top_p,
                                         max_tokens=max_completion_tokens,
                                         n=num_completions)
        message = prompting_method.build_prompt(query_instance)
        conversation = build_conversation(message)
        outputs_batch = self.llm.chat(conversation,
                                      sampling_params=sampling_params,
                                      use_tqdm=False,
                                      chat_template_content_format='openai')
        assert len(outputs_batch) == 1  # batch size of 1, hard coding by design
        outputs = outputs_batch[0]  # take the first element in the batch
        list_gen_text = [output_n.text for output_n in outputs.outputs]
        return prompting_method.parse_response(response=list_gen_text)
