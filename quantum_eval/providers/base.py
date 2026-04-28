from abc import ABC, abstractmethod


class Provider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model_id: str,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> str:
        """Generate a completion for prompt. Returns the model's reply as a string."""
        ...
