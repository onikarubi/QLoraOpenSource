from typing import Optional
import torch
import wandb
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
from .dataset_manager import DatasetManager
from .llm import CausalLM
from .logging_formatter import get_logger
from .tokenizer import Tokenizer

logger = get_logger(__name__)


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
        self._init_train_model_config()

    def _init_train_model_config(self):
        """Initialize the model configuration based on tokenizer."""
        additional_tokens = self.tokenizer.get_additional_special_tokens()
        if additional_tokens:
            current_vocab_size = self.llm.model.config.vocab_size
            new_vocab_size = len(self.tokenizer.hf_tokenizer)

            if new_vocab_size != current_vocab_size:
                self.llm.model.resize_token_embeddings(new_vocab_size)
            else:
                logger.info("Tokenizer and model have the same vocab size.")


    def _create_default_training_args(self) -> TrainingArguments:
        """Create default training arguments."""
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
            report_to="wandb",
        )

    def _create_default_lora_config(self) -> LoraConfig:
        """Create default LoRA configuration based on tokenizer."""
        base_config = {
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "r": 4,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": self.llm.linear_layer_names,
        }
        if self.tokenizer.get_additional_special_tokens():
            base_config["modules_to_save"] = ["lm_head", "embed_tokens"]

        return LoraConfig(**base_config)

    def create_sft_trainer(self) -> SFTTrainer:
        """Creates and returns an SFTTrainer for training."""
        return SFTTrainer(
            model=self.llm.model,
            tokenizer=self.tokenizer.hf_tokenizer,
            train_dataset=self.dataset_manager.dataset,
            peft_config=self.lora_config,
            args=self.training_args,
            dataset_text_field="text",
            max_seq_length=512,
        )

    def _convert_normalization_layer_to_float32(self, trainer: SFTTrainer) -> None:
        """Converts normalization layers in the model to float32 for better training stability."""
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module.to(torch.float32)

    def train(self, saved_in_path: str) -> None:
        """Trains the model and saves it to the specified path."""
        try:
            logger.debug("Starting model training...")
            trainer = self.create_sft_trainer()
            self._convert_normalization_layer_to_float32(trainer)
            self._log_trainer_info(trainer)
            trainer.train()
            trainer.model.save_pretrained(saved_in_path)
            logger.info("Model saved to %s", saved_in_path)
            logger.info("Training complete👏!!")
        except Exception as e:
            logger.error("An error occurred during training: %s", e)
            raise e
        finally:
            logger.info("Finishing wandb run...")
            wandb.finish()

    def _log_trainer_info(self, trainer: SFTTrainer) -> None:
        """Log information about the trainer."""
        logger.info("Trainer info:")
        logger.info("epochs: %s", trainer.args.num_train_epochs)
        logger.info("batch size: %s", trainer.args.per_device_train_batch_size)
        logger.info(
            "gradient accumulation steps: %s", trainer.args.gradient_accumulation_steps
        )

    def config_lora(self, **kwargs) -> None:
        """Configures the LoRA (Low-Rank Adaptation) settings."""
        self.lora_config = LoraConfig(
            task_type="CAUSAL_LM", target_modules=self.llm.linear_layer_names, **kwargs
        )

    def config_train_args(self, **kwargs) -> None:
        """Configures the training arguments for the model."""
        default_args = {
            "output_dir": "./train_logs",
            "fp16": True,
            "logging_strategy": "epoch",
            "save_strategy": "epoch",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "optim": "paged_adamw_32bit",
            "learning_rate": 5e-4,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": 0.3,
            "warmup_ratio": 0.03,
            "weight_decay": 0.001,
            "group_by_length": True,
            "report_to": "wandb",
        }
        default_args.update(kwargs)
        self.training_args = TrainingArguments(**default_args)
