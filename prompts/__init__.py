from abc import ABC, abstractmethod
from typing import Any


# an abstract class for all prompting methods (e.g., LLB, LL-PGM)
class PromptingMethod(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the prompting method.
        This method should set up any necessary attributes or configurations.
        """
        system_prompt = None  # str
        exemplars = None
        expected_response_blocks = None  # list of strs

    @abstractmethod
    def build_prompt(self, inst: str, **kwargs):
        pass

    @abstractmethod
    # parse to every blocks in self.expected_response_blocks and return a dict
    def parse_response(self, response: Any, **kwargs):
        pass

    @abstractmethod
    def handle_none_match(self, **kwargs):
        pass
