from abc import ABC, abstractmethod


# an abstract class for all prompting methods (e.g., LLB, LL-PGM)
class PromptingMethod(ABC):
    system_prompt = None
    exemplars = None
    expected_response_blocks = None

    @abstractmethod
    def build_prompt(self, inst, **kwargs):
        pass
