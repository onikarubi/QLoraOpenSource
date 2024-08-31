import torch
from .llm import CausalLM
from .logging_formatter import get_logger
from .tokenizer import Tokenizer
from .models.model_registry import ModelRegistry
from .prompt import PromptDefaultFormatter, PromptLlama3Formatter, PromptLlama2Formatter

logger = get_logger(__name__)

class ModelRunner:
    def __init__(self, llm: CausalLM, tokenizer: Tokenizer) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

    def _create_prompt(self, system: str, instruction: str) -> str:
        registry = ModelRegistry()
        model_name = self.tokenizer.repo_id

        if model_name in registry.get_model(family="llama", version="llama3") or model_name in registry.get_model('elyza', version='llama3'):
            formatter = PromptLlama3Formatter("chat_openai", self.tokenizer)

        elif model_name in registry.get_model('elyza', version='llama2'):
            formatter = PromptLlama2Formatter("chat_openai", self.tokenizer)

        else:
            formatter = PromptDefaultFormatter("default")

        return formatter.create_input_prompt(system, instruction)

    def run(self, questions: list[str], system: str, max_tokens: int = 500):
        logger.debug("Starting model inference...")
        responses = []

        for question in questions:
            try:
                prompt = self._create_prompt(system, question)

                model_inputs = self.tokenizer.hf_tokenizer(
                    [prompt], return_tensors="pt"
                ).to(self.device)

                generated_ids = self.llm.model.generate(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=max_tokens,
                )

                trimmed_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]

                response = self.tokenizer.hf_tokenizer.batch_decode(
                    trimmed_ids, skip_special_tokens=True
                )[0]

                logger.info("Generated response: %s", response)
                responses.append(response)

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
