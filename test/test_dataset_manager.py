from qlora.dataset_manager import DatasetManager
from qlora.tokenizer import Tokenizer
from qlora.prompt import PromptDefaultFormatter
from qlora.models.model_registry import ModelRegistry
from qlora.tokenizer import Tokenizer
from qlora.logging_formatter import get_logger
from datasets import load_dataset
import pytest

logger = get_logger(__name__)

dataset_path = "datasets/sample.jsonl"
registry = ModelRegistry()
models_to_test = [
    ("gemma", {}),
    ("llama", {"version": "llama3"}),
    ("elyza", {"version": "llama2"}),
    ("elyza", {"version": "llama3"}),
]


def get_tokenizer(model_name, model_kwargs):
    model = registry.get_model(model_name, **model_kwargs)
    return Tokenizer(model)


@pytest.mark.parametrize("model_name, model_kwargs", models_to_test)
def test_get_dataset(model_name, model_kwargs):
    tokenizer = get_tokenizer(model_name, model_kwargs)
    manager = DatasetManager(
        dataset_path=dataset_path, data_format="chat_openai", tokenizer=tokenizer
    )
    assert "text" in manager.dataset.column_names

