from transformers import (
    TrainingArguments,
)
from llm import CausalLM
from tokenizer import Tokenizer
from dataset_manager import DatasetManager
from peft import LoraConfig
from trl import SFTTrainer
import torch

class ModelTrainer:
    llm: CausalLM
    tokenizer: Tokenizer
    dataset_manager: DatasetManager
    training_args: TrainingArguments
    lora_config: LoraConfig

    def __init__(
        self,
        llm: CausalLM,
        tokenizer: Tokenizer,
        dataset_manager: DatasetManager,
        lora_config: LoraConfig | str = 'default',
        training_args: TrainingArguments | str = 'default'
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.dataset_manager = dataset_manager

        if lora_config == 'default':
            self.lora_config = self._create_default_lora_config()
        else:
            self.lora_config = lora_config

        if training_args == 'default':
            self.training_args = self._create_default_training_args()
        else:
            self.training_args = training_args

        self._convert_normalization_layer_to_float32()

    @property
    def trainer(self):
        return SFTTrainer(
            model=self.llm.model,
            tokenizer=self.tokenizer.tokenizer,
            train_dataset=self.dataset_manager.dataset,
            dataset_text_field='text',
            peft_config=self.lora_config,
            args=self.training_args,
            max_seq_length=512
        )

    def _convert_normalization_layer_to_float32(self):
        for name, module in self.llm.model.named_modules():
            if 'norm' in name:
                module = module.to(torch.float32)

    def _create_default_lora_config(self):
        return LoraConfig(
            lora_alpha=8,
            lora_dropout=0.1,
            r=4,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.llm.linear_layer_names
        )

    def _create_default_training_args(self):
        return TrainingArguments(
            output_dir="./train_logs",
            fp16=True,
            logging_strategy='epoch',
            save_strategy='epoch',
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            learning_rate=5e-4,
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            weight_decay=0.001,
            group_by_length=True,
            report_to="tensorboard"
        )

    def train(self, saved_in_path: str):
        try:
            self.trainer.train()
            self.trainer.model.save_pretrained(saved_in_path)

        except Exception as e:
            print('Failed to train the model')
            raise e

    def config_lora(
        self,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        r: int = 4,
        bias: str = "none",
    ) -> None:
        new_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=r,
            bias=bias,
            task_type="CAUSAL_LM",
            target_modules=self.llm.linear_layer_names
        )
        self.lora_config = new_config

    def config_train_args(
        self,
        output_dir: str = "./train_logs",
        fp16: bool = True,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        optim: str = "paged_adamw_32bit",
        learning_rate: float = 5e-4,
        lr_scheduler_type: str = "cosine",
        max_grad_norm: float = 0.3,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.001,
        group_by_length: bool = True,
    ):
        new_training_args = TrainingArguments(
            output_dir=output_dir,
            fp16=fp16,
            logging_strategy='epoch',
            save_strategy='epoch',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            group_by_length=group_by_length,
            report_to="tensorboard"
        )
        self.training_args = new_training_args
