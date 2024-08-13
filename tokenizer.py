from transformers import (
    AutoTokenizer, 
)

class Tokenizer:
    repo_id: str
    attn_implementation: str

    def __init__(self, repo_id: str, attn_implementation: str = 'default') -> None:
        self.repo_id = repo_id
        self.attn_implementation = attn_implementation

    @property
    def tokenizer(self):
        if self.attn_implementation == 'default':
            self.attn_implementation = 'eager'

        return self._initialize_tokenizer(self.attn_implementation)

    def _initialize_tokenizer(self, attn_implementation: str):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.repo_id,
            attn_implementation=attn_implementation,
            add_eos_token=True
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = 'right'
        return tokenizer