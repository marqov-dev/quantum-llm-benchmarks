import pytest
from unittest.mock import MagicMock, patch
from quantum_eval.providers.base import Provider
from quantum_eval.providers.compat import CompatProvider


def test_provider_is_abstract():
    """Provider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Provider()


def test_compat_provider_generate_returns_string():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)"

    with patch("quantum_eval.providers.compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        provider = CompatProvider(base_url="http://localhost:11434/v1", api_key="ollama")
        result = provider.generate(
            "Write a Bell state circuit",
            model_id="llama3",
            temperature=0.2,
            max_tokens=2048,
            stop=None,
        )

    assert isinstance(result, str)
    assert "QuantumCircuit" in result


def test_compat_provider_passes_stop_tokens():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "code here"

    with patch("quantum_eval.providers.compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        provider = CompatProvider(base_url="http://localhost:11434/v1", api_key="ollama")
        provider.generate(
            "prompt",
            model_id="llama3",
            temperature=0.2,
            max_tokens=2048,
            stop=["```"],
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stop"] == ["```"]


def test_compat_provider_omits_stop_when_none():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "code"

    with patch("quantum_eval.providers.compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        provider = CompatProvider(base_url="http://localhost:11434/v1", api_key="ollama")
        provider.generate("prompt", model_id="llama3", temperature=0.2, max_tokens=2048, stop=None)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "stop" not in call_kwargs
