"""Pytest configuration and global fixtures for Project Jane."""

import os
import pytest


@pytest.fixture(autouse=True)
def isolate_chat_history(tmp_path):
    """
    Autouse fixture that redirects all chat history logging during pytest
    runs to an isolated temporary directory, preventing test suite queries
    from ever polluting the production logs/chat_history.jsonl file.
    """
    isolated_path = str(tmp_path / "chat_history.jsonl")
    old_env = os.environ.get("JANE_CHAT_HISTORY_PATH")
    os.environ["JANE_CHAT_HISTORY_PATH"] = isolated_path
    yield isolated_path
    if old_env is not None:
        os.environ["JANE_CHAT_HISTORY_PATH"] = old_env
    else:
        os.environ.pop("JANE_CHAT_HISTORY_PATH", None)
