from google import genai
from google.genai import types
from quantum_eval.providers.base import Provider


class GoogleProvider(Provider):
    """Native Google GenAI SDK adapter for Gemini models."""

    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        *,
        model_id: str,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> str:
        config_kwargs: dict = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if stop:
            config_kwargs["stop_sequences"] = stop
        config = types.GenerateContentConfig(**config_kwargs)
        response = self._client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config,
        )
        if not response.candidates:
            return ""
        candidate = response.candidates[0]
        if not getattr(candidate, "content", None) or not getattr(candidate.content, "parts", None):
            return ""
        return "".join(p.text for p in candidate.content.parts if hasattr(p, "text"))
