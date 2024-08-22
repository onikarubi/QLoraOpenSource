from utils.dataset_generator import ChatOpenAIDatasetGenerator
from qlora.dataset_manager import DatasetManager
from qlora.llm import CausalLM
from qlora.tokenizer import Tokenizer
from qlora.train import ModelTrainer
import os

test_dataset_path = 'datasets/simple.csv'
test_output_path = 'datasets/simple.jsonl'

def test_chat_openai_dataset_generator():
    dataset_generator = ChatOpenAIDatasetGenerator(test_dataset_path, output_path=test_output_path)
    dataset_generator.generate()
    assert os.path.exists(test_dataset_path)
