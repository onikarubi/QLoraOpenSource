import abc
import csv
import json

from .logger import getLogger

logger = getLogger(__name__)

class FileFormatError(Exception):
    pass

class FileExtensionError(Exception):
    pass

class DatasetGenerator(metaclass=abc.ABCMeta):
    def __init__(
        self,
        file: str,
        output_path: str,
        system: str = "あなたは日本語で回答するAIアシスタントです。"
    ) -> None:
        self.file = file
        self.output_path = output_path
        self.system = system

    def generate(self) -> None:
        file_extension = self.file.split('.')[-1]
        try:
            if file_extension == 'csv':
                self.csv_to_jsonl(self.file, self.output_path)
                logger.info("Generating JSONL file is completed.")
                logger.info("Saving file to: " + self.output_path)
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                raise FileExtensionError("サポートされていないファイル形式です。")

        except FileFormatError as e:
            logger.error(f"File format error: {e}")

        except FileExtensionError as e:
            logger.error(f"File extension error: {e}")

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    @abc.abstractmethod
    def csv_to_jsonl(self, csv_file: str, output_path: str) -> None:
        pass


class ChatOpenAIDatasetGenerator(DatasetGenerator):
    def csv_to_jsonl(self, csv_file: str, output_path: str) -> None:
        with open(csv_file, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames != ['questions', 'answers']:
                raise FileFormatError("データのフォーマットが正しくありません。")

            with open(output_path, "w", encoding="utf-8") as jsonlfile:
                for row in reader:
                    jsonl_data = {
                        "messages": [
                            {"role": "system", "content": self.system},
                            {"role": "user", "content": row["questions"]},
                            {"role": "assistant", "content": row["answers"]},
                        ]
                    }
                    jsonlfile.write(json.dumps(jsonl_data, ensure_ascii=False) + "\n")
