from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Tokenizer:
    """
    A wrapper class for initializing and configuring a tokenizer from Hugging Face's Transformers library.

    Args:
        repo_id (str): The repository ID of the model for which the tokenizer is being initialized.
        attn_implementation (str): The attention implementation type. Default is "eager".
    """

    def __init__(self, repo_id: str, attn_implementation: str = "eager") -> None:
        self.repo_id = repo_id
        self.attn_implementation = attn_implementation
        self._hf_tokenizer = (
            self._initialize_tokenizer()
        )  # Internal attribute with a more descriptive name

    @property
    def hf_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Returns the initialized Hugging Face tokenizer.

        Returns:
            PreTrainedTokenizerBase: The initialized tokenizer instance.
        """
        return self._hf_tokenizer  # Property with a more descriptive name

    def _initialize_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Initializes the tokenizer with the specified settings.

        Returns:
            PreTrainedTokenizerBase: The initialized tokenizer instance.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.repo_id,
            attn_implementation=self.attn_implementation,
            add_eos_token=True,
        )

        # Ensure that the pad token is set to eos token if not already set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"
        return tokenizer

