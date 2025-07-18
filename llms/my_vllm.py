from vllm import LLM, SamplingParams, RequestOutput
from utils import get_vllm_dtype
from llms import BaseLLM
from typing import List


class MyVLLM(BaseLLM):
    def __init__(self,
                 model: str,
                 gpu_mem_utilization: float,
                 num_gpu: int,
                 dtype: str = None,
                 allow_media_path: str = None):
        if dtype is None:
            dtype = get_vllm_dtype()
        self.llm = LLM(model=model,
                       gpu_memory_utilization=gpu_mem_utilization,
                       tensor_parallel_size=num_gpu,
                       allow_media_path=allow_media_path,
                       dtype=dtype)

    def chat_completion(self, prompt: List[dict] | List[List[dict]], **kwargs) -> list[RequestOutput]:
        sampling_params = SamplingParams(temperature=kwargs.get('temperature'),
                                         top_k=kwargs.get('top_k'),
                                         top_p=kwargs.get('top_p'),
                                         max_tokens=kwargs.get('max_completion_tokens'),
                                         n=kwargs.get('n', 1))
        outputs_batch = self.llm.chat(prompt,
                                      sampling_params=sampling_params,
                                      use_tqdm=False,
                                      chat_template_content_format='openai')
        assert len(outputs_batch) == 1  # batch size of 1, hard coding by design
        return outputs_batch
