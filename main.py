from llm import CausalLM
from train import ModelTrainer
from dataset_manager import DatasetManager
from tokenizer import Tokenizer
from dotenv import load_dotenv
from datasets import load_dataset
from transformers.utils import is_accelerate_available
import torch

load_dotenv()

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    data_path = './sample.jsonl'
    repo_id = 'google/gemma-2-2b-it'
    tokenizer = Tokenizer(repo_id)
    manager = DatasetManager(data_path, format='chat_openai')
    llm = CausalLM(repo_id)
    trainer = ModelTrainer(
        llm=llm,
        tokenizer=tokenizer,
        dataset_manager=manager,
        lora_config='default',
        training_args='default'
    )

    trainer.train('./saved_model')
