from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer
from datasets import load_dataset, Dataset

from test_dataset_manager import create_test_dataset, create_dataset_manager
from test_llm import create_casual_llm
from test_tokenizer import create_tokenizer_instance
from qlora.train import ModelTrainer
from qlora.dataset_manager import DatasetManager
from qlora.tokenizer import Tokenizer
from qlora.llm import CausalLM
from qlora.logging_formatter import logger
import torch
import json
import bitsandbytes as bnb


def test_runner():
    repo_id = "google/gemma-2-2b-it"
    dataset_path = './sample.jsonl'
    manager = DatasetManager(
        dataset_path=dataset_path,
        format='chat_openai'
    )

    tokenizer = Tokenizer(repo_id=repo_id)
    llm = CausalLM(repo_id=repo_id)
    trainer = ModelTrainer(
        llm=llm,
        tokenizer=tokenizer,
        dataset_manager=manager
    )
    trainer.train('./output')
