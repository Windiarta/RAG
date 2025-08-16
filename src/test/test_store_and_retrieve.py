from typing import List

import pytest

from src.modules.chroma import VectorStore


class DummyEmbedder:
	def __call__(self, texts: List[str]) -> List[List[float]]:  # type: ignore[name-defined]
		# Simple deterministic embedding: length and first few chars
		return [[float(len(t))] + [float(ord(c)) for c in (t[:3] + "\0\0\0")[:3]] for t in texts]


def test_add_list_delete(tmp_path, monkeypatch):
	store = VectorStore(persist_path=str(tmp_path / "chroma"), embedder=DummyEmbedder())
	doc_id = store.add_document(None, "hello world\nthis is a test", metadata={"filename": "a.txt"}, chunk_size=10, overlap=2)
	listed = store.list_documents()
	assert any(did == doc_id for did, _ in listed)
	deleted = store.delete_document(doc_id)
	assert deleted >= 1


