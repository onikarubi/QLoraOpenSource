import pytest

from qlora.dataset_manager import DatasetManager
from qlora.logging_formatter import get_logger
from qlora.models.model_registry import ModelRegistry
from qlora.prompt import (PromptDefaultFormatter, PromptLlama2Formatter,
                          PromptLlama3Formatter)
from qlora.tokenizer import Tokenizer

logger = get_logger(__name__)

dataset_path = "datasets/sample.jsonl"
registry = ModelRegistry()
models_to_test = [
    ("gemma", {}),
    ("llama", {"version": "llama3", "variant": "default"}),
    ("elyza", {"version": "llama2", "variant": "default"}),
    ("elyza", {"version": "llama2", "variant": "instruct"}),
    ("elyza", {"version": "llama3", "variant": "default"}),
]


def get_tokenizer(model_name, model_kwargs):
    model = registry.get_model(model_name, **model_kwargs)
    return Tokenizer(model)


@pytest.mark.parametrize("model_name, model_kwargs", models_to_test)
def test_select_formatter(model_name, model_kwargs):
    tokenizer = get_tokenizer(model_name, model_kwargs)
    manager = DatasetManager(
        dataset_path=dataset_path, data_format="chat_openai", tokenizer=tokenizer
    )
    formatter = manager._select_formatter(tokenizer.repo_id)

    if model_kwargs.get("version") == "llama3":
        assert isinstance(formatter, PromptLlama3Formatter)

    elif model_kwargs.get("version") == "llama2" and model_kwargs.get('variant') == "default":
        assert isinstance(formatter, PromptLlama2Formatter)

    elif model_kwargs.get("version") == "llama2" and model_kwargs.get('variant') == "instruct":
        assert isinstance(formatter, PromptLlama2Formatter)

    else:
        assert isinstance(formatter, PromptDefaultFormatter)


@pytest.mark.parametrize("model_name, model_kwargs", models_to_test)
def test_get_dataset(model_name, model_kwargs):
    tokenizer = get_tokenizer(model_name, model_kwargs)
    manager = DatasetManager(
        dataset_path=dataset_path, data_format="chat_openai", tokenizer=tokenizer
    )
    assert "text" in manager.dataset.column_names
