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
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop or [],
        )
        response = self._client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config,
        )
        return response.text
