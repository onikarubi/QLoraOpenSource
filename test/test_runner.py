from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from qlora.runner import ModelRunner
from qlora.train import ModelTrainer
from qlora.dataset_manager import DatasetManager
from qlora.tokenizer import Tokenizer
from qlora.llm import CausalLM
from qlora.logging_formatter import logger
import torch
import json
import bitsandbytes as bnb

repo_id = "google/gemma-2-2b-it"
adapter_path = "./test_output"

def test_runner():
    question = '自己紹介をしてくれますか？'
    llm = CausalLM(repo_id=adapter_path)
    tokenizer = Tokenizer(repo_id=repo_id)
    runner = ModelRunner(llm=llm, tokenizer=tokenizer)
    runner.run([question])
