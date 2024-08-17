from llm import CausalLM
from runner import ModelRunner
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
    torch.cuda.empty_cache()
    device = torch.device("cuda" if use_cuda else "cpu")
    adapter_path = './ずんだもん_Adapter'
    repo_id = 'google/gemma-2-2b-it'

    runner = ModelRunner(repo_id, adapter_path)
    runner.run("自己紹介をしてくれますか？")
