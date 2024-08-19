import argparse
import yaml
from llm import CausalLM
from train import ModelTrainer
from dataset_manager import DatasetManager
from tokenizer import Tokenizer
from dotenv import load_dotenv
from logging_formatter import logger
import torch


def setup_environment():
    """Load environment variables and check for CUDA availability."""
    load_dotenv()

    use_cuda = torch.cuda.is_available()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if use_cuda else "cpu")

    if device.type != "cuda":
        logger.error("CUDA is not available. Switching to CPU")
        raise ValueError("CUDA is not available. Switching to CPU")
    else:
        logger.info("CUDA is available. Using GPU")

    return device


def initialize_components(dataset_path: str, repo_id: str, config_file: str = None):
    """Initialize and return the model, dataset manager, tokenizer, and trainer."""
    # If a config file is provided, load it
    config = {}
    if config_file:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")

    llm = CausalLM(
        repo_id=repo_id,
        quantization_config=config.get("quantization_config"),
        attn_implementation=config.get("attn_implementation", "eager"),
    )
    manager = DatasetManager(
        dataset_path=dataset_path, format=config.get("format", "chat_openai")
    )
    tokenizer = Tokenizer(
        repo_id=repo_id, attn_implementation=config.get("attn_implementation", "eager")
    )
    trainer = ModelTrainer(
        llm=llm,
        tokenizer=tokenizer,
        dataset_manager=manager,
        lora_config=config.get("lora_config", None),
        training_args=config.get("training_args", None),
    )

    return trainer


def train_model(trainer, output_dir: str):
    """Train the model and save it to the specified output directory."""
    trainer.train(saved_in_path=output_dir)


def run_model():
    """Placeholder function for running the model."""
    logger.info("Run task is not yet implemented.")


def main():
    """Main function to handle different tasks based on the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run tasks for model training or inference."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train", "run"],
        help="The task to perform: train or run",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help="The path to the dataset (required for training)",
    )
    parser.add_argument(
        "--repo_id", type=str, required=True, help="The repository ID of the model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="The directory to save the trained model (required for training)",
    )
    parser.add_argument(
        "--config_file", type=str, required=False, help="The configuration file to use"
    )

    args = parser.parse_args()

    # Setup environment
    device = setup_environment()
    logger.info("Device: %s", device)

    if args.task == "train":
        if not args.dataset_path or not args.output_dir:
            parser.error(
                "The --dataset_path and --output_dir arguments are required for the train task."
            )

        trainer = initialize_components(
            args.dataset_path, args.repo_id, args.config_file
        )
        train_model(trainer, args.output_dir)

    elif args.task == "run":
        run_model()


"""
!python main.py --task train --dataset_path ./sample.jsonl --repo_id google/gemma-2-2b-it --output_dir ./output
"""
if __name__ == "__main__":
    main()
