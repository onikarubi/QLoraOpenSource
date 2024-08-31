from qlora.tokenizer import Tokenizer
from qlora.logging_formatter import get_logger

logger = get_logger(__name__)

repo_id = "google/gemma-2-2b-it"

def test_create_tokenizer():
    tokenizer = Tokenizer(repo_id)
    assert tokenizer is not None
    logger.info("Tokenizer chat template: %s" % tokenizer.hf_tokenizer.chat_template)
