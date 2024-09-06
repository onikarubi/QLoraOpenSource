from qlora.runner import ModelRunner
from qlora.logging_formatter import get_logger

logger = get_logger(__name__)

repo_id = "google/gemma-2-2b-it"
adapter_path = "./test_output"

def test_create_input_prompt():
    question = ["自己紹介をしてくれますか？"]
    system = "あなたはずんだもんです。嘘をつくのは苦手です。"

    runner = ModelRunner(
        adapter_path=adapter_path,
        base_model=repo_id,
    )

    input_prompt = runner._create_prompt(system, question)
    assert isinstance(input_prompt, str)


def test_run():
    questions = ["自己紹介をしてくれますか？"]
    system = "あなたはずんだもんです。嘘をつくのは苦手です。"
    max_tokens = 500

    runner = ModelRunner(
        adapter_path=adapter_path,
        base_model=repo_id,
    )

    responses = runner.run(questions, system, max_tokens)
    logger.info(responses)