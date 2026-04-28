from openai import OpenAI
from quantum_eval.providers.base import Provider


class CompatProvider(Provider):
    """Adapter for any OpenAI-compatible endpoint (Ollama, vLLM, DeepSeek, Groq, etc.)."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)

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
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if stop is not None:
            kwargs["stop"] = stop
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
