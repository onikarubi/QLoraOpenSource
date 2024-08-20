import json

import torch
from datasets import Dataset
from dotenv import load_dotenv

from dataset_manager import DatasetManager
from logging_formatter import logger

load_dotenv()

dataset_path = "./sample.jsonl"

def load_dataset():
    json_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            json_data.append(json.loads(line))

    dataset = Dataset.from_list(json_data)
    return dataset

def create_validated_dataset():
    PROMPT_FORMAT = """<start_of_turn>user
{system}

{instruction}
<end_of_turn>
<start_of_turn>model
{output}
<end_of_turn>
"""

    json_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            json_data.append(json.loads(line))

    dataset = Dataset.from_list(json_data)

    def generate_text_field(data):
        messages = data["messages"]
        system = ""
        instruction = ""
        output = ""
        for message in messages:
            if message["role"] == "system":
                system = message["content"]
            elif message["role"] == "user":
                instruction = message["content"]
            elif message["role"] == "assistant":
                output = message["content"]
        full_prompt = PROMPT_FORMAT.format(
            system=system, instruction=instruction, output=output
        )
        return {"text": full_prompt}

    train_dataset = dataset.map(generate_text_field)

    return train_dataset

def create_test_dataset():
    manager = DatasetManager(dataset_path, "chat_openai")
    return manager.dataset

def create_dataset_manager():
    manager = DatasetManager(dataset_path, "chat_openai")
    return manager

def test_create_dataset():
    test_dataset = create_test_dataset()
    validated_dataset = create_validated_dataset()
    logger.info("Creating dataset %s", validated_dataset)
    logger.info("Creating dataset %s", test_dataset)
    assert validated_dataset["text"] == test_dataset["text"], "Datasets do not match"
    print("--------------------------------")
    logger.debug(test_dataset['text'])
    print("--------------------------------")
    logger.debug(validated_dataset["text"])
    print("--------------------------------")
