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
            conversation=[{'role': 'user', 'content': question}],
            tokenize=False,
            add_generation_prompt=True
        )

        token_ids = self.tokenizer.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors='pt'
        ).to('cuda')

        with torch.no_grad():
            output_ids = self.llm.model.generate(
                token_ids.input_ids,
                max_new_tokens=600,
                attention_mask=token_ids.attention_mask
            )

        output = self.tokenizer.tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1):],
            skip_special_tokens=True
        )
        print(output)

        return output