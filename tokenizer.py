from transformers import (
    AutoTokenizer, 
)

class Tokenizer:
    def __init__(self, repo_id: str, attn_implementation: str = "eager") -> None:
        self.repo_id = repo_id
        self.attn_implementation = attn_implementation
        self.tokenizer = self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.repo_id,
            attn_implementation=self.attn_implementation,
            add_eos_token=True,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"
        return tokenizer
