import os
from src.modules.retriever import Retriever


def test_retriever_has_ask_method(monkeypatch):
	monkeypatch.setenv("GEMINI_API_KEY", "dummy")
	retriever = Retriever()
	assert hasattr(retriever, "ask")

