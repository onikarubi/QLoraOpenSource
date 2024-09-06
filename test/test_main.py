import subprocess
from logging import INFO
from unittest.mock import patch
from main import run_model

import pytest
from qlora.logging_formatter import get_logger

logger = get_logger(__name__, level=INFO)

test_data = [
    (
        "google/gemma-2-2b-it",
        "./test_output",
        ["自己紹介をしてくれますか？"],
        "あなたはずんだもんです。嘘をつくのは苦手です。",
        500,
    )
]


@pytest.mark.parametrize(
    "repo_id, adapter_path, questions, system, max_tokens", test_data
)
@patch("main.CausalLM")
@patch("main.Tokenizer")
@patch("main.ModelRunner.run")
def test_run_model_execute(
    mock_run,
    MockTokenizer,
    MockCausalLM,
    repo_id,
    adapter_path,
    questions,
    system,
    max_tokens,
):
    mock_run.return_value = ["私の名前はGemmaです。"]

    run_model(
        repo_id=repo_id,
        adapter_path=adapter_path,
        questions=questions,
        system=system,
        max_tokens=max_tokens,
    )

    mock_run.assert_called_once()
    mock_run.assert_called_with(
        questions=questions, system=system, max_tokens=max_tokens
    )


