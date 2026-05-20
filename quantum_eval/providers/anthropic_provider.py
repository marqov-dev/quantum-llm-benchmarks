from anthropic import Anthropic, BadRequestError
from quantum_eval.providers.base import Provider


class AnthropicProvider(Provider):
    """Native Anthropic SDK adapter for Claude models."""

    def __init__(self, api_key: str) -> None:
        self._client = Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        *,
        model_id: str,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> str:
        kwargs: dict = dict(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if stop is not None:
            kwargs["stop_sequences"] = stop
        try:
            message = self._client.messages.create(**kwargs)
        except BadRequestError as e:
            if "temperature" in str(e) and "deprecated" in str(e):
                # Some models (e.g. Opus 4.7) don't accept a temperature parameter.
                kwargs.pop("temperature")
                message = self._client.messages.create(**kwargs)
            else:
                raise
        return message.content[0].text if message.content else ""
