import pytest

from qlora.logging_formatter import get_logger
from qlora.models.model_registry import ModelRegistry
from qlora.tokenizer import Tokenizer

logger = get_logger(__name__)
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
def test_create_tokenizer(model_name, model_kwargs):
    tokenizer = get_tokenizer(model_name, model_kwargs)
    special_tokens = tokenizer.hf_tokenizer.special_tokens_map
    if special_tokens.get('additional_special_tokens'):
        logger.info("Adding additional special tokens: %s", special_tokens['additional_special_tokens'])

    else:
        logger.info("No additional special tokens found.")

