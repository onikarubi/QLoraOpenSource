import argparse
import yaml
from qlora.llm import CausalLM
from qlora.train import ModelTrainer
from qlora.runner import ModelRunner
from qlora.dataset_manager import DatasetManager
from qlora.tokenizer import Tokenizer
from dotenv import load_dotenv
from qlora.logging_formatter import logger
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


def initialize_components(
    dataset_path: str, repo_id: str, config_file: str = None, epochs: int = 3
):
    """Initialize and return the model, dataset manager, tokenizer, and trainer."""
    # If a config file is provided, load it
    config = {}
    if config_file:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")

    llm = CausalLM(
        repo_id=repo_id,
        quantization_config=config.get("quantization_config", None),
        attn_implementation=config.get("attn_implementation", "eager"),
    )
    tokenizer = Tokenizer(
        repo_id=repo_id, attn_implementation=config.get("attn_implementation", "eager")
    )
    manager = DatasetManager(
        dataset_path=dataset_path,
        data_format=config.get("format", "chat_openai"),
        tokenizer=tokenizer,
    )
    trainer = ModelTrainer(
        llm=llm,
        tokenizer=tokenizer,
        dataset_manager=manager,
        lora_config=config.get("lora_config", None),
        training_args=config.get("training_args", None),
    )

    # Configure training arguments to use the specified number of epochs
    trainer.config_train_args(num_train_epochs=epochs)

    return trainer


def train_model(trainer, output_dir: str):
    """Train the model and save it to the specified output directory."""
    trainer.train(saved_in_path=output_dir)


def run_model(repo_id: str, adapter_path: str, questions: list[str], system: str, max_tokens: int):
    """Run inference using the model and respond to the provided questions."""
    runner = ModelRunner(adapter_path=adapter_path, base_model=repo_id)

    responses = runner.run(questions=questions, system=system, max_tokens=max_tokens)
    for i, response in enumerate(responses, 1):
        logger.info(f"\nResponse {i}: {response}")


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
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=False,
        help="The path to the model adapter (required for running)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=False,
        default=500,
        help="The maximum number of tokens to generate (default: 500)",
    )
    parser.add_argument(
        "--questions",
        type=str,
        nargs="+",
        required=False,
        help="The questions to ask the model (required for running)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=3,
        help="The number of epochs for training (default: 3)",
    )
    parser.add_argument(
        "--system",
        type=str,
        required=False,
        default="",
        help="The system text to use when asking questions"
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
            args.dataset_path, args.repo_id, args.config_file, args.epochs
        )
        train_model(trainer, args.output_dir)

    elif args.task == "run":
        if not args.adapter_path or not args.questions:
            parser.error(
                "The --adapter_path and --questions arguments are required for the run task."
            )
        run_model(
            repo_id=args.repo_id,
            adapter_path=args.adapter_path,
            questions=args.questions,
            system=args.system,
            max_tokens=args.max_tokens,
        )

"""
!python main.py --task train --dataset_path datasets/simple.jsonl --repo_id google/gemma-2-2b-it --output_dir ./output --epochs 3

!python main.py --task run --repo_id google/gemma-2-2b-it --adapter_path ./output --max_tokens 500 --system "あなたはずんだもんです。嘘をつくのは苦手です。" --questions "自己紹介をしてくれますか？"

--task: このスクリプトの実行タスクを指定します。train または run のいずれかを指定します。（必須）
--dataset_path: データセットのパスを指定します。trainの場合は必須です。
--adapter_path: モデルのアダプターパスを指定します。runの場合は必須です。
--repo_id: モデルのリポジトリIDを指定します。（必須）
--output_dir: モデルの保存先ディレクトリを指定します。trainの場合は必須です。
--config_file: モデルの設定ファイルを指定します。（任意）もし指定しない場合はデフォルトの設定が使用されます。
--max_tokens: 生成されるトークンの最大数を指定します。（任意）デフォルトは500です。
--epochs: モデルのエポック数を指定します。（任意）デフォルトは3です。
--questions: モデルに対して質問をする際の質問文を指定します。（必須）複数の質問を指定する場合は空白で区切って指定します。
--system: モデルに対して質問をする際のシステム文を指定します。（任意）指定しない場合は空文字列が使用されます。
"""

if __name__ == "__main__":
    main()
