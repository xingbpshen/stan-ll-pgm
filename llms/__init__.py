from abc import ABC, abstractmethod
from typing import List


LLM_STATUS_READY = 'READY'
LLM_STATUS_BUSY = 'BUSY'


class BaseLLM(ABC):
    """
    Abstract base class for all LLMs.
    """

    @abstractmethod
    def status(self) -> str:
        """
        Get the status of the LLM.

        Returns:
            str: The status of the LLM.
        """
        pass

    @abstractmethod
    def chat_completion(self, prompt: List[dict] | List[List[dict]], **kwargs) -> any:
        """
        Generate text based on the provided prompt.

        Args:
            prompt (List[dict]): The input conversation to generate text from.
            **kwargs: Additional parameters for the generation.

        Returns:
            any: The generated content.
        """
        pass

    @abstractmethod
    def text_in_completion(self, completion: any) -> any:
        """
        Extract text from the completion response.

        Args:
            completion (any): The completion response from the LLM.

        Returns:
            str | List[str]: The extracted text.
        """
        pass
