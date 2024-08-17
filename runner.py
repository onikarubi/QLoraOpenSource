from llm import CausalLM
from tokenizer import Tokenizer
import torch


class ModelRunner:
    llm: CausalLM
    tokenizer: Tokenizer
    repo_id: str
    adapter_path: str

    def __init__(self, repo_id: str, adapter_path: str) -> None:
        self.repo_id = repo_id
        self.adapter_path = adapter_path
        self.llm = CausalLM(adapter_path)
        self.tokenizer = Tokenizer(repo_id)

    def run(self, question: str):
        prompt = self.tokenizer.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer.tokenizer([prompt], return_tensors="pt").to(
            "cuda"
        )
        generated_ids = self.llm.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=300,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        print(generated_ids)
        return generated_ids
