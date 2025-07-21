from vllm import LLM, SamplingParams, RequestOutput
from utils import get_vllm_dtype
from typing import List
import llms


class MyVLLM(llms.BaseLLM):
    def __init__(self, model: str, gpu_mem_utilization: float, num_gpu: int, dtype: str = None, allow_media_path: str = None, **kwargs):

        if dtype is None:
            dtype = get_vllm_dtype()

        self.llm = LLM(model=model,
                       gpu_memory_utilization=gpu_mem_utilization,
                       tensor_parallel_size=num_gpu,
                       allow_media_path=allow_media_path,
                       dtype=dtype)

    def status(self) -> str:
        """
        Get the status of the LLM.

        Returns:
            str: The status of the LLM, which is always 'READY' for MyVLLM.
        """
        return llms.LLM_STATUS_READY

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
        return outputs_batch

    def text_in_completion(self, completion: any) -> List[List[str]]:
        """
        Extract text from the completion output.

        Args:
            completion (any): The completion output from the LLM.

        Returns:
            List[List[str]]: The generated text from the completion, (batch, n).
        """
        text = []
        for c in completion:
            text.append([output_n.text for output_n in c.outputs])
        return text
