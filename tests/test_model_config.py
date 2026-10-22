from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from open_deep_research.utils import get_model_config


def test_ollama_model_config_sets_num_predict(monkeypatch):
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_config = get_model_config("ollama:gemma3:4b", 123, {})
    configurable_model = init_chat_model(
        configurable_fields=("model", "max_tokens", "num_predict", "api_key")
    )

    ollama_model = configurable_model.with_config(model_config).bound
    request = ollama_model._chat_params([HumanMessage(content="Test")])

    assert model_config["num_predict"] == 123
    assert "max_tokens" not in model_config
    assert request["options"]["num_predict"] == 123


def test_other_providers_keep_max_tokens(monkeypatch):
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_config = get_model_config("openai:gpt-4.1", 456, {})

    assert model_config["max_tokens"] == 456
    assert "num_predict" not in model_config
