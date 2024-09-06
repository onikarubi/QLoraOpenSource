import bitsandbytes as bnb
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)

from qlora.llm import CausalLM

repo_id = "google/gemma-2-2b-it"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16, 
)

def create_validated_model():
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        device_map={"": "cuda"},
        quantization_config=quantization_config,
        attn_implementation="eager",
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model

def create_test_model():
    llm = CausalLM(
        repo_id=repo_id,
        quantization_config=quantization_config,
        attn_implementation="eager",
    )
    return llm.model

def create_casual_llm():
    llm = CausalLM(
        repo_id=repo_id,
        quantization_config=quantization_config,
        attn_implementation="eager",
    )
    return llm

def create_validated_linear_layer_names(model):
    target_class = bnb.nn.Linear4bit
    linear_layer_names = set()
    for name_list, module in model.named_modules():
        if isinstance(module, target_class):
            names = name_list.split(".")
            layer_name = names[-1] if len(names) > 1 else names[0]
            linear_layer_names.add(layer_name)
    if "lm_head" in linear_layer_names:
        linear_layer_names.remove("lm_head")
    return list(linear_layer_names)


def test_model_liner_layer_names():
    validated_model = create_validated_model()
    llm = CausalLM(
        repo_id=repo_id,
        quantization_config=quantization_config,
        attn_implementation="eager",
    )

    assert create_validated_linear_layer_names(validated_model) == llm.linear_layer_names, "Linear layer names do not match"
