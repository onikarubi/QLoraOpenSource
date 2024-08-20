from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from llm import CausalLM
import torch
import bitsandbytes as bnb

repo_id = "google/gemma-2-2b-it"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4ビット量子化を使用
    bnb_4bit_quant_type="nf4",  # 4ビット量子化の種類にnf4（NormalFloat4）を使用
    bnb_4bit_use_double_quant=True,  # 二重量子化を使用
    bnb_4bit_compute_dtype=torch.float16,  # 量子化のデータ型をfloat16に設定
)

def create_validated_model():
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=repo_id,  # モデルのリポジトリIDをセット
        device_map={"": "cuda"},  # 使用デバイスを設定
        quantization_config=quantization_config,  # 量子化のConfigをセット
        attn_implementation="eager",  # 注意機構に"eager"を設定（Gemma2モデルの学習で推奨されているため）
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
