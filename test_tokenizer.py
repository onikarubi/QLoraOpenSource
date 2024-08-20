from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from tokenizer import Tokenizer

repo_id = "google/gemma-2-2b-it"

def create_validated_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=repo_id,  # モデルのリポジトリIDをセット
        attn_implementation="eager",  # 注意機構に"eager"を設定（Gemma2モデルの学習で推奨されているため）
        add_eos_token=True,  # EOSトークンの追加を設定
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # パディングを右側に設定(fp16を使う際のオーバーフロー対策)
    tokenizer.padding_side = "right"
    return tokenizer

def create_test_tokenizer():
    tokenizer = Tokenizer(
        repo_id=repo_id,
    )
    return tokenizer.tokenizer

def create_tokenizer_instance():
    return Tokenizer(
        repo_id=repo_id,
    )

def test_tokenizer_padding():
    validated_tokenizer = create_validated_tokenizer()
    test_tokenizer = create_test_tokenizer()
    assert validated_tokenizer.pad_token == test_tokenizer.pad_token, "Pad tokens do not match"
    assert validated_tokenizer.padding_side == test_tokenizer.padding_side, "Padding sides do not match"
    print("--------------------------------")
    print(validated_tokenizer.pad_token)
    print("--------------------------------")
    print(test_tokenizer.pad_token)
    print("--------------------------------")
    print(validated_tokenizer.padding_side)
    print("--------------------------------")
    print(test_tokenizer.padding_side)
    print("--------------------------------")
    print(validated_tokenizer)
    print("--------------------------------")
    print(test_tokenizer)
    print("--------------------------------")

