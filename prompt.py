from datasets import Dataset
import abc
from typing import Dict


class PromptFormatter(metaclass=abc.ABCMeta):
    PROMPT_FORMAT = """<start_of_turn>user
{system}

{instruction}
<end_of_turn>
<start_of_turn>model
{output}
<end_of_turn>
"""

    @abc.abstractmethod
    def create_text_format(cls, dataset: Dataset) -> Dict[str, str]:
        pass


class PromptDefaultFormatter(PromptFormatter):
    @classmethod
    def create_text_format(cls, dataset: Dataset) -> Dict[str, str]:
        system = dataset["system"]
        instruction = dataset["instruction"]
        output = dataset["output"]

        text = cls.PROMPT_FORMAT.format(
            system=system, instruction=instruction, output=output
        )
        return {"text": text}


class PromptChatOpenAIFormatter(PromptFormatter):
    @classmethod
    def create_text_format(cls, dataset: Dataset) -> Dict[str, str]:
        messages = dataset["messages"]
        system, instruction, output = cls._parse_messages(messages)
        text = cls.PROMPT_FORMAT.format(
            system=system, instruction=instruction, output=output
        )
        return {"text": text}

    @staticmethod
    def _parse_messages(messages: list) -> tuple:
        system, instruction, output = "", "", ""

        for message in messages:
            if message["role"] == "system":
                system += message["content"]
            elif message["role"] == "user":
                instruction += message["content"]
            elif message["role"] == "assistant":
                output += message["content"]

        return system, instruction, output
