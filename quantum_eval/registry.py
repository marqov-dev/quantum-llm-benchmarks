import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from quantum_eval.providers.base import Provider
from quantum_eval.providers.anthropic_provider import AnthropicProvider
from quantum_eval.providers.openai_provider import OpenAIProvider
from quantum_eval.providers.google_provider import GoogleProvider
from quantum_eval.providers.compat import CompatProvider

_DEFAULT_REGISTRY = Path(__file__).parent / "_data" / "models" / "registry.yaml"


@dataclass
class ModelConfig:
    id: str
    label: str
    provider_name: str
    base_url: str | None
    api_key_env: str
    temperature: float
    max_tokens: int
    max_retries: int

    def build_provider(self) -> Provider:
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise EnvironmentError(
                    f"Environment variable '{self.api_key_env}' is not set or empty. "
                    f"Export it before running model '{self.id}'."
                )
        else:
            api_key = "ollama"
        if self.provider_name == "anthropic":
            return AnthropicProvider(api_key=api_key)
        if self.provider_name == "openai":
            return OpenAIProvider(api_key=api_key)
        if self.provider_name == "google":
            return GoogleProvider(api_key=api_key)
        if self.provider_name == "compat":
            return CompatProvider(base_url=self.base_url or "", api_key=api_key)
        raise ValueError(f"Unknown provider: {self.provider_name!r}")


def load_registry(registry_path: Path | None = None) -> list[ModelConfig]:
    path = registry_path or _DEFAULT_REGISTRY
    with path.open() as f:
        data = yaml.safe_load(f)
    return [
        ModelConfig(
            id=entry["id"],
            label=entry["label"],
            provider_name=entry["provider"],
            base_url=entry.get("base_url"),
            api_key_env=entry.get("api_key_env", ""),
            temperature=entry.get("temperature", 0.2),
            max_tokens=entry.get("max_tokens", 4096),
            max_retries=entry.get("max_retries", 2),
        )
        for entry in data["models"]
    ]


def get_model(model_id: str, registry: list[ModelConfig] | None = None, registry_path: Path | None = None) -> ModelConfig:
    configs = registry if registry is not None else load_registry(registry_path)
    for config in configs:
        if config.id == model_id:
            return config
    raise KeyError(
        f"Model '{model_id}' not found in registry. "
        "Run `quantum-llm-benchmarks list` to see available models."
    )
