from datasets import load_dataset, Dataset
from prompt import PromptChatOpenAIFormatter, PromptDefaultFormatter
import pandas as pd

class DatasetManager:
    dataset_path: str
    extension: str
    default_dataset: Dataset
    format: str

    def __init__(self, dataset_path: str, format: str = 'default') -> None:
        self.dataset_path = dataset_path
        self.format = format
        self.extension = dataset_path.split('.')[-1]
        self.default_dataset = self.create_default_dataset()

    @property
    def dataset(self):
        return self.create_dataset()

    def create_dataset(self):
        if self.format == 'default':
            return self.default_dataset.map(PromptDefaultFormatter.create_text_format)

        elif self.format == 'chat_openai':
            return self.default_dataset.map(PromptChatOpenAIFormatter.create_text_format)

        else:
            raise ValueError('Dataset format not supported')

    def create_default_dataset(self) -> Dataset:
        if self.extension == 'jsonl':
            return self._create_dataset_from_json()

        elif self.extension == 'csv':
            return self._create_dataset_from_csv()

        else:
            raise ValueError('Dataset format not supported')

    def _create_dataset_from_csv(self) -> Dataset:
        csv_data = pd.read_csv(self.dataset_path)
        return Dataset.from_pandas(csv_data)

    def _create_dataset_from_json(self):
        datasets = load_dataset('json', data_files=self.dataset_path, split='train')
        return datasets
