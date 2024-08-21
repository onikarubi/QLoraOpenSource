from transformers import (
    TrainingArguments,
)
from llm import CausalLM
from tokenizer import Tokenizer
from dataset_manager import DatasetManager
from peft import LoraConfig
from trl import SFTTrainer
from logging_formatter import logger
from typing import Optional
from datasets import Dataset
import torch


class ModelTrainer:
    def __init__(
        self,
        llm: CausalLM,
        tokenizer: Tokenizer,
        dataset_manager: DatasetManager,
        lora_config: Optional[LoraConfig] = None,
        training_args: Optional[TrainingArguments] = None,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.dataset_manager = dataset_manager
        self.lora_config = lora_config or self._create_default_lora_config()
        self.training_args = training_args or self._create_default_training_args()

    def _create_default_training_args(self):
        return TrainingArguments(
            output_dir="./train_logs",
            fp16=True,
            logging_strategy="epoch",
            save_strategy="epoch",
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
            report_to="tensorboard",
        )

    def _create_default_lora_config(self):
        return LoraConfig(
            lora_alpha=8,
            lora_dropout=0.1,
            r=4,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.llm.linear_layer_names,
        )

    def create_trainer(self):
        return SFTTrainer(
            model=self.llm.model,
            tokenizer=self.tokenizer.tokenizer,
            train_dataset=self.dataset_manager.dataset,
            dataset_text_field="text",
            peft_config=self.lora_config,
            args=self.training_args,
            max_seq_length=512,
        )

    def _convert_normalization_layer_to_float32(self, trainer: SFTTrainer):
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module.to(torch.float32)

        return trainer

    def train(self, saved_in_path: str):
        try:
            logger.debug("Training model...")
            trainer = self.create_trainer()
            trainer = self._convert_normalization_layer_to_float32(trainer)
            logger.info('Trainer info:')
            logger.info('epochs: %s', trainer.args.num_train_epochs)
            logger.info('batch size: %s', trainer.args.per_device_train_batch_size)
            logger.info('gradient accumulation steps: %s', trainer.args.gradient_accumulation_steps)
            output = trainer.train()
            trainer.model.save_pretrained(saved_in_path)
            logger.info("Model saved to %s", saved_in_path)
            logger.info("Training output: %s", output)
        except Exception as e:
            logger.error("An error occurred during training: %s", e)
            raise e

    def config_lora(
        self,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        r: int = 4,
        bias: str = "none",
    ) -> None:
        self.lora_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=r,
            bias=bias,
            task_type="CAUSAL_LM",
            target_modules=self.llm.linear_layer_names,
        )

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
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            fp16=fp16,
            logging_strategy="epoch",
            save_strategy="epoch",
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
            report_to="tensorboard",
        )
