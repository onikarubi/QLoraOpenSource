from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .logging_formatter import get_logger
from .models.model_registry import ModelRegistry

logger = get_logger(__name__)

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
        self._hf_tokenizer = self._initialize_tokenizer()
        self.registry = ModelRegistry()

    @property
    def hf_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Returns the initialized Hugging Face tokenizer.

        Returns:
            PreTrainedTokenizerBase: The initialized tokenizer instance.
        """
        return self._hf_tokenizer  # Property with a more descriptive name

    def get_additional_special_tokens(self) -> list[str] | None:
        """
        Returns the additional special tokens added to the tokenizer.

        Returns:
            list[str] | None: A list of special tokens or None if no special tokens were added.
        """
        special_tokens_dict = self.hf_tokenizer.special_tokens_map
        return special_tokens_dict.get("additional_special_tokens")

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

        special_tokens = self._create_special_tokens()
        if special_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        self._check_special_tokens(tokenizer, special_tokens)
        return tokenizer

    def _create_special_tokens(self) -> list[str] | None:
        """
        Creates a list of special tokens from the tokenizer.

        Returns:
            list[str]: A list of special tokens.
        """

        registry = ModelRegistry()

        if registry.is_llama2(self.repo_id):
            logger.debug("Creating special tokens: %s", self.repo_id)
            return ["[R_START]", "[R_END]"]

        return None

    def _check_special_tokens(
        self, tokenizer: PreTrainedTokenizerBase, special_tokens: list[str] | None
    ):
        """Check to see if any special tokens have been added to the tokenizer."""
        if special_tokens:
            added_special_tokens = set(tokenizer.all_special_tokens)
            for token in special_tokens:
                if token in added_special_tokens:
                    logger.info(f"特別なトークン '{token}' は正常に追加されました。")
                else:
                    logger.warning(f"特別なトークン '{token}' は追加されていません。")
        else:
            logger.info("追加する特別なトークンは提供されませんでした。")
