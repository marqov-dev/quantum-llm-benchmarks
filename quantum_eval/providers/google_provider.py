import google.generativeai as genai
from quantum_eval.providers.base import Provider


class GoogleProvider(Provider):
    """Native Google GenAI SDK adapter for Gemini models."""

    def __init__(self, api_key: str) -> None:
        genai.configure(api_key=api_key)

    def generate(
        self,
        prompt: str,
        *,
        model_id: str,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> str:
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop or [],
        )
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text
