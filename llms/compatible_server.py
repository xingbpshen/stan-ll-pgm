import llms


class CompatibleServer(llms.BaseLLM):
    """
    CompatibleServer class that extends BaseLLM to provide compatibility with various LLM servers.
    """

    def __init__(self):
        """
        Initialize the CompatibleServer class.
        This constructor can be extended to include any necessary initialization logic.
        """
        super().__init__()
        pass

    def status(self) -> str:
        pass

    def chat_completion(self, prompt: List[dict], **kwargs) -> any:
        """
        Generate text based on the provided prompt.

        Args:
            prompt (List[dict]): The input content to generate a response for.

        Returns:
            any: The generated response.
        """
        pass
