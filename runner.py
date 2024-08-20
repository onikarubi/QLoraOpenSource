from llm import CausalLM
from tokenizer import Tokenizer
from logging_formatter import logger
import torch


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
                prompt = self.tokenizer.tokenizer.apply_chat_template(
                    conversation=[{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                model_inputs = self.tokenizer.tokenizer(
                    [prompt], return_tensors="pt"
                ).to(self.device)
                generated_ids = self.llm.model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=max_tokens,
                )
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]

                response = self.tokenizer.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                logger.info("Generated response: %s", response)
                responses.append(response)

            except torch.cuda.OutOfMemoryError as e:
                logger.error("CUDA Out of Memory Error: %s", e)
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
