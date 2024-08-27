from typing import Dict, Literal, Optional
import pandas as pd
from datasets import Dataset, load_dataset
from .prompt import PromptLlamaFormatter, PromptDefaultFormatter, PromptFormatter
from .tokenizer import Tokenizer
from .logging_formatter import getLogger
from .models.model_registry import ModelRegistry

logger = getLogger(__name__)


class DatasetManager:
    def __init__(
        self,
        dataset_path: str,
        data_format: Literal["default", "chat_openai"],
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.format = data_format
        self.extension = dataset_path.split(".")[-1]
        self.tokenizer = tokenizer
        self.model_name = tokenizer.repo_id if tokenizer else None

        if not self.tokenizer:
            logger.info("Tokenizer not provided")

        self._dataset = self.get_dataset()

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def get_dataset(self) -> Dataset:
        """Loads and processes the dataset to add text fields."""
        init_dataset = self._load_dataset()
        dataset = init_dataset.map(self._create_text_field)
        return dataset

    def _create_text_field(self, example: Dict) -> Dict[str, str]:
        """
        Adds a text field to a dataset example.

        Uses PromptLlamaFormatter if the model is of type 'llama' or 'elyza'.
        Otherwise, uses PromptDefaultFormatter.
        """
        if self.model_name:
            model_registry = ModelRegistry()
            if self.model_name in model_registry.get_model(
                "elyza"
            ) or self.model_name in model_registry.get_model("llama"):
                logger.debug("model_name is in llama or elyza")
                formatter = PromptLlamaFormatter(
                    data_type=self.format, tokenizer=self.tokenizer
                )
            else:
                formatter = PromptDefaultFormatter(data_type=self.format)
        else:
            formatter = PromptDefaultFormatter(data_type=self.format)

        # Use the formatter to create a text field for each example
        return formatter.create_text_field(example)

    def _load_dataset(self) -> Dataset:
        """Loads dataset based on file extension."""
        if self.extension == "jsonl":
            return self._create_dataset_from_json()
        elif self.extension == "csv":
            return self._create_dataset_from_csv()
        else:
            raise ValueError("Dataset format not supported")

    def _create_dataset_from_csv(self) -> Dataset:
        """Creates a dataset from a CSV file."""
        csv_data = pd.read_csv(self.dataset_path)
        return Dataset.from_pandas(csv_data)

    def _create_dataset_from_json(self) -> Dataset:
        """Creates a dataset from a JSONL file."""
        dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        return dataset
