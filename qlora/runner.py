import torch
from .llm import CausalLM
from .logging_formatter import get_logger
from .tokenizer import Tokenizer

logger = get_logger(__name__)

class ModelRunner:
    def __init__(self, llm: CausalLM, tokenizer: Tokenizer) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

    def run(self, questions: list[str], max_tokens: int = 500):
        responses = []

        for question in questions:
            try:
                prompt = self.tokenizer.hf_tokenizer.apply_chat_template(
                    conversation=[{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

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
