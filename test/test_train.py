import torch
from transformers import TrainingArguments

from qlora.dataset_manager import DatasetManager
from qlora.llm import CausalLM
from qlora.logging_formatter import get_logger, logger
from qlora.models.model_registry import ModelRegistry
from qlora.tokenizer import Tokenizer
from qlora.train import ModelTrainer

torch.cuda.empty_cache()

logger = get_logger(__name__)

training_arguments = TrainingArguments(
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


registry = ModelRegistry()
gemma_model = registry.get_model("gemma")

tokenizer_gemma = Tokenizer(gemma_model)

dataset_path = "datasets/sample.jsonl"
manager = DatasetManager(dataset_path=dataset_path, data_format="chat_openai", tokenizer=tokenizer_gemma)
dataset = manager.dataset

test_llm = CausalLM(repo_id=gemma_model)
logger.info("target_modules first: %s", test_llm.linear_layer_names)

def _create_test_trainer():
    trainer = ModelTrainer(
        llm=test_llm,  
        tokenizer=tokenizer_gemma,  
        dataset_manager=manager, 
        lora_config=None,  
        training_args=None, 
    )

    return trainer

def test_trainer_model_named_modules():
    test_trainer = _create_test_trainer()
    logger.info("target_modules second: %s", test_trainer.llm.linear_layer_names)
    sft_trainer = test_trainer.create_sft_trainer()

    # test_trainerの各モジュールも同じように確認
    for name, module in sft_trainer.model.named_modules():
        if "norm" in name:
            assert (
                module.weight.dtype == torch.float32
            ), f"Module {name} in test_trainer is not in torch.float32 dtype."

    logger.info("All modules are converted to torch.float32 dtype.")

