import pytest

from src import config


@pytest.fixture(autouse=True)
def fast_llm(monkeypatch):
    """LLM trong test là đồ giả nên không cần giãn nhịp hay chờ giữa các lần thử lại."""
    monkeypatch.setattr(config, "LLM_REQUESTS_PER_MINUTE", 100_000)
    monkeypatch.setattr("src.llm_utils.time.sleep", lambda *_: None)
