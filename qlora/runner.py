import torch
from peft.peft_model import PeftModel

from .llm import CausalLM
from .logging_formatter import get_logger
from .models.model_registry import ModelRegistry
from .prompt import (PromptDefaultFormatter, PromptLlama2Formatter,
                     PromptLlama3Formatter)
from .tokenizer import Tokenizer

logger = get_logger(__name__)

class ModelRunner:
    """ A class to run the model with the given questions. """
    def __init__(self, adapter_path: str, base_model: str) -> None:
        self.base_llm = CausalLM(repo_id=base_model)
        self.tokenizer = Tokenizer(repo_id=base_model)
        self._init_model_config()
        self.adapter_model = PeftModel.from_pretrained(
            self.base_llm.model,
            adapter_path,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_model_config(self):
        """Initialize the model configuration based on tokenizer."""
        self.base_llm.model.resize_token_embeddings(len(self.tokenizer.hf_tokenizer))

    def _create_prompt(self, system: str, instruction: str) -> str:
        registry = ModelRegistry()
        model_name = self.tokenizer.repo_id

        if registry.is_llama3(model_name):
            formatter = PromptLlama3Formatter("chat_openai", self.tokenizer)

        elif registry.is_llama2(model_name):
            formatter = PromptLlama2Formatter("chat_openai", self.tokenizer)

        else:
            formatter = PromptDefaultFormatter("default")

        return formatter.create_input_prompt(system, instruction)

    def run(self, questions: list[str], system: str, max_tokens: int = 500):
        """ Run inference using the model and respond to the provided questions. """
        logger.debug("Starting model inference...")
        responses = []

        def _decode_token(token_ids: list[int]):
            return self.tokenizer.hf_tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            )

        for question in questions:
            try:
                prompt = self._create_prompt(system, question)

                model_inputs = self.tokenizer.hf_tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                with torch.no_grad():
                    output_ids = self.adapter_model.generate(
                        input_ids=model_inputs["input_ids"],
                        attention_mask=model_inputs["attention_mask"],
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.hf_tokenizer.unk_token_id,
                        eos_token_id=self.tokenizer.hf_tokenizer.eos_token_id,
                    )

                print('--------------------------------')
                print('output_ids: \n')
                output_ids_to_list: list[list[int]] = output_ids.tolist()
                logger.debug("output_ids_to_list = %s", output_ids_to_list)
                print('--------------------------------')
                print()
                print('--------------------------------')
                print('output_ids[0]: \n')
                output_ids_to_list_0: list[list[int]] = output_ids.tolist()[0]
                logger.debug("output_ids_to_list[0] = %s", output_ids_to_list[0])
                logger.debug("output_ids_to_list[0]:decode = %s", _decode_token(output_ids_to_list_0))
                print('--------------------------------')
                print()
                print('--------------------------------')
                print('model_inputs["input_ids"]: \n')
                model_input_ids = model_inputs["input_ids"]
                logger.debug("model_input_ids = %s", model_input_ids)
                print('--------------------------------')
                print()
                print('--------------------------------')
                print('input sequence length: \n')
                input_sequence_length = model_inputs["input_ids"].size(1)
                logger.debug("model_inputs_size = %s", input_sequence_length)
                print('--------------------------------')
                print()
                print('--------------------------------')
                print('input batch size: \n')
                input_batch_size = model_inputs["input_ids"].size(0)
                logger.debug("batch size = %s", input_batch_size)
                print("--------------------------------")
                print()
                print('--------------------------------')
                print('input_ids: \n')
                input_ids = model_inputs["input_ids"].tolist()[0]
                logger.debug("input_ids = %s", input_ids)
                logger.debug("input_ids:decode = %s", _decode_token(input_ids))
                print('--------------------------------')
                print()
                print('--------------------------------')
                print('token_ids: \n')
                token_ids = output_ids_to_list[0][input_sequence_length:]
                logger.debug("token_ids = %s", token_ids)
                logger.debug("token_ids:decode = %s", _decode_token(token_ids))
                print('--------------------------------')
                print()

                output = self.tokenizer.hf_tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                )
                responses.append(output)
                logger.info("Question: %s", question)
                logger.info("Answer: %s", output)

            except torch.cuda.OutOfMemoryError as e:
                logger.error("CUDA Out of Memory Error: %s", e)
                torch.cuda.empty_cache()
                raise e
            except RuntimeError as e:
                logger.error("Runtime Error: %s", e)
                raise e
            except Exception as e:
                logger.error(
                    "An unexpected error occurred during model generation: %s", e
                )
                raise e

        return responses
