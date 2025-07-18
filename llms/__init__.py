from abc import ABC, abstractmethod
from typing import List


class BaseLLM(ABC):
    """
    Abstract base class for all LLMs.
    """

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
