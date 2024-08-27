import abc
from typing import Dict, Literal, Tuple

from datasets import Dataset

from .tokenizer import Tokenizer
from .models.model_registry import ModelRegistry
from transformers.tokenization_utils import PreTrainedTokenizer
from .logging_formatter import get_logger

logger = get_logger(__name__)


class PromptFormatter(metaclass=abc.ABCMeta):
    def __init__(self, data_type: Literal["default", "chat_openai"]) -> None:
        self.data_type = data_type

    def create_text_field(self, example: Dict) -> Dict[str, str]:
        """Generates text field based on the specified data type."""
        text = self._generate_prompt(example)
        return {"text": text}

    @abc.abstractmethod
    def _generate_prompt(self, example: Dict) -> str:
        """Generates the prompt based on the specified data type."""
        pass


class PromptDefaultFormatter(PromptFormatter):
    def __init__(self, data_type: Literal["default", "chat_openai"]) -> None:
        super().__init__(data_type)
        self._format = self._create_base_format()

    @property
    def format(self) -> str:
        return self._format

    def _create_base_format(self) -> str:
        """Defines the default format for prompts."""
        template = """<start_of_turn>user
{system}

{instruction}
<end_of_turn>
<start_of_turn>model
{output}
<end_of_turn>
"""
        return template

    def _generate_prompt(self, example: Dict) -> str:
        """Creates text format based on the specified type."""
        if self.data_type == "chat_openai":
            system, instruction, output = self._parse_messages(example["messages"])
        else:
            system = example["system"]
            instruction = example["instruction"]
            output = example["output"]

        return self.format.format(system=system, instruction=instruction, output=output)

    def _parse_messages(self, messages: list) -> Tuple[str, str, str]:
        """Parses messages to extract system, user, and assistant contents."""
        system, instruction, output = "", "", ""
        for message in messages:
            if message["role"] == "system":
                system += message["content"]
            elif message["role"] == "user":
                instruction += message["content"]
            elif message["role"] == "assistant":
                output += message["content"]
        return system, instruction, output


class PromptLlamaFormatter(PromptFormatter):
    def __init__(
        self, data_type: Literal["default", "chat_openai"], tokenizer: Tokenizer
    ) -> None:
        super().__init__(data_type)
        self.tokenizer = tokenizer
        self.model_registry = ModelRegistry()
        self._validate_model_type(tokenizer)

    def _generate_prompt(self, example: Dict) -> str:
        """Generates the prompt based on the messages in the example."""
        if self.data_type != "chat_openai":
            raise ValueError("Data type not supported")

        return self._create_base_format(example["messages"])

    def _validate_model_type(self, tokenizer: Tokenizer) -> None:
        """Validates if the model type is supported by the formatter."""
        if tokenizer.repo_id not in self.model_registry.get_model(
            "elyza"
        ) and tokenizer.repo_id not in self.model_registry.get_model("llama"):
            raise ValueError("Model not supported")

    def _create_base_format(self, messages: list) -> str:
        """Creates the format based on the tokenizer's chat template."""
        data = []

        for message in messages:
            if message["role"] in {"system", "user", "assistant"}:
                data.append({"role": message["role"], "content": message["content"]})
            else:
                raise ValueError("Role not supported")

        logger.debug("Creating data: %s", data)

        template = self.tokenizer.hf_tokenizer.apply_chat_template(data, tokenize=False)
        return template
