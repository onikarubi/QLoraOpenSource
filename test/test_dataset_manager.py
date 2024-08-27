from qlora.dataset_manager import DatasetManager
from qlora.tokenizer import Tokenizer
from qlora.prompt import PromptDefaultFormatter, PromptLlamaFormatter
from qlora.models.model_registry import ModelRegistry
from qlora.tokenizer import Tokenizer
from qlora.logging_formatter import get_logger
from datasets import load_dataset
import os
import json

logger = get_logger(__name__)
dataset_path = 'datasets/sample.jsonl'
registry = ModelRegistry()
target_model_by_llama = registry.get_model('llama')
target_model_by_gemma = registry.get_model('gemma')

tokenizer_by_llama = Tokenizer(target_model_by_llama)
tokenizer_by_gemma = Tokenizer(target_model_by_gemma)

test_dataset = load_dataset('json', data_files=dataset_path, split='train')

def test_create_default_formatter():
    logger.debug('Creating dataset: %s' % test_dataset)
    formatter_by_default = PromptDefaultFormatter('chat_openai')
    logger.debug('formatter by default: %s' % formatter_by_default.format)
    new_dataset = test_dataset.map(formatter_by_default.create_text_field)
    logger.debug('new dataset: %s' % new_dataset)

def test_create_llama_formatter():
    formatter_by_llama = PromptLlamaFormatter('chat_openai', tokenizer_by_llama)
    new_dataset = test_dataset.map(formatter_by_llama.create_text_field)
    logger.debug('new dataset: %s' % new_dataset)

def test_create_dataset_by_default():
    manager = DatasetManager(dataset_path=dataset_path, data_format='chat_openai')
    dataset = manager.dataset
    logger.debug('dataset: %s' % dataset)
    assert 'text' in dataset.column_names

def test_create_dataset_by_llama():
    manager = DatasetManager(dataset_path=dataset_path, data_format='chat_openai', tokenizer=tokenizer_by_llama)
    dataset = manager.dataset
    logger.debug('dataset: %s' % dataset)
    assert 'text' in dataset.column_names
