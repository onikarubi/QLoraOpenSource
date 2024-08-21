from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

from test_dataset_manager import create_test_dataset, create_dataset_manager
from test_llm import create_casual_llm
from test_tokenizer import create_tokenizer_instance
from qlora.train import ModelTrainer
from qlora.logging_formatter import logger
import torch

training_arguments = TrainingArguments(
    output_dir="./train_logs",  # ログの出力ディレクトリ
    fp16=True,  # fp16を使用
    logging_strategy="epoch",  # 各エポックごとにログを保存（デフォルトは"steps"）
    save_strategy="epoch",  # 各エポックごとにチェックポイントを保存（デフォルトは"steps"）
    num_train_epochs=3,  # 学習するエポック数
    per_device_train_batch_size=1,  # （GPUごと）一度に処理するバッチサイズ
    gradient_accumulation_steps=4,  # 勾配を蓄積するステップ数
    optim="paged_adamw_32bit",  # 最適化アルゴリズム
    learning_rate=5e-4,  # 初期学習率
    lr_scheduler_type="cosine",  # 学習率スケジューラの種別
    max_grad_norm=0.3,  # 勾配の最大ノルムを制限（クリッピング）
    warmup_ratio=0.03,  # 学習を増加させるウォームアップ期間の比率
    weight_decay=0.001,  # 重み減衰率
    group_by_length=True,  # シーケンスの長さが近いものをまとめてバッチ化
    report_to="tensorboard",  # TensorBoard使用してログを生成（"./train_logs"に保存）
)

llm = create_casual_llm()
model = llm.model
tokenizer = create_tokenizer_instance()
manager = create_dataset_manager()
dataset = manager.dataset

target_modules = llm.linear_layer_names


def create_validated_model_trainer():
    Lora_config = LoraConfig(
        lora_alpha=8,  # LoRAによる学習の影響力を調整（スケーリング)
        lora_dropout=0.1,  # ドロップアウト率
        r=4,  # 低ランク行列の次元数
        bias="none",  # バイアスのパラメータ更新
        task_type="CAUSAL_LM",  # タスクの種別
        target_modules=target_modules,  # LoRAを適用するモジュールのリスト
    )

    trainer = SFTTrainer(
        model=model,  # モデルをセット
        tokenizer=tokenizer.tokenizer,  # トークナイザーをセット
        train_dataset=dataset,  # データセットをセット
        dataset_text_field="text",  # 学習に使用するデータセットのフィールド
        peft_config=Lora_config,  # LoRAのConfigをセット
        args=training_arguments,  # 学習パラメータをセット
        max_seq_length=512,  # 入力シーケンスの最大長を設定
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    return trainer

def create_test_trainer():
    trainer = ModelTrainer(
        llm=llm,  # CausalLMをセット
        tokenizer=tokenizer,  # Tokenizerをセット
        dataset_manager=manager,  # DatasetManagerをセット
    )

    return trainer.trainer

def test_trainer_model_named_modules():
    validated_trainer = create_validated_model_trainer()
    test_trainer = create_test_trainer()

    # validated_trainerの各モジュールが適切に変換されているか確認
    for name, module in validated_trainer.model.named_modules():
        if "norm" in name:
            assert (
                module.weight.dtype == torch.float32
            ), f"Module {name} is not in torch.float32 dtype."

    # test_trainerの各モジュールも同じように確認
    for name, module in test_trainer.model.named_modules():
        if "norm" in name:
            assert (
                module.weight.dtype == torch.float32
            ), f"Module {name} in test_trainer is not in torch.float32 dtype."

    logger.info("All modules are converted to torch.float32 dtype.")
