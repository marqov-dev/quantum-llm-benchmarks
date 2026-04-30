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


from quantum_eval.providers.anthropic_provider import AnthropicProvider
from quantum_eval.providers.openai_provider import OpenAIProvider
from quantum_eval.providers.google_provider import GoogleProvider


def test_anthropic_provider_generate():
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="from qiskit import QuantumCircuit")]

    with patch("quantum_eval.providers.anthropic_provider.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(api_key="sk-test")
        result = provider.generate(
            "Write a Bell state",
            model_id="claude-sonnet-4-6",
            temperature=1.0,
            max_tokens=2048,
            stop=None,
        )

    assert "QuantumCircuit" in result


def test_openai_provider_generate():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "from qiskit import QuantumCircuit"

    with patch("quantum_eval.providers.openai_provider.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(api_key="sk-test")
        result = provider.generate(
            "Write a Bell state",
            model_id="gpt-4o",
            temperature=0.2,
            max_tokens=2048,
            stop=None,
        )

    assert "QuantumCircuit" in result


def test_google_provider_generate():
    mock_response = MagicMock()
    mock_response.text = "from qiskit import QuantumCircuit"

    with patch("quantum_eval.providers.google_provider.genai") as mock_genai:
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GoogleProvider(api_key="key-test")
        result = provider.generate(
            "Write a Bell state",
            model_id="gemini-2.5-flash",
            temperature=0.2,
            max_tokens=2048,
            stop=None,
        )

    assert "QuantumCircuit" in result
