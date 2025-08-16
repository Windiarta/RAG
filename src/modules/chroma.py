import os
import uuid
import logging
from typing import Callable, Dict, List, Optional, Tuple

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, IDs, Metadatas

from .ocr import Chunks
from .logging_config import configure_logging

logger = logging.getLogger(__name__)


class _CallableEmbeddingFn(EmbeddingFunction):
	def __init__(self, fn: Callable[[List[str]], List[List[float]]]):
		self.fn = fn

	def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
		return self.fn(list(input))


class VectorStore:
	"""ChromaDB wrapper with chunking and embeddings injection."""

	def __init__(self, persist_path: Optional[str] = None, collection_name: str = "rag_documents", embedder: Optional[Callable[[List[str]], List[List[float]]]] = None) -> None:
		configure_logging()
		self.persist_path = persist_path or os.getenv("CHROMA_PATH", os.path.join("data", "chroma"))
		os.makedirs(self.persist_path, exist_ok=True)
		self.collection_name = collection_name
		self._client = chromadb.PersistentClient(path=self.persist_path)
		self._embedder_fn = embedder
		self._collection = self._get_or_create_collection()
		logger.info("VectorStore ready at %s collection=%s", self.persist_path, self.collection_name)

	def _get_or_create_collection(self):
		embedding_function = _CallableEmbeddingFn(self._embedder_fn or self._default_embed)
		return self._client.get_or_create_collection(name=self.collection_name, embedding_function=embedding_function)

	def _default_embed(self, texts: List[str]) -> List[List[float]]:
		# Lazy import to avoid heavy dependency at test import time
		from .gemini import GeminiClient
		client = GeminiClient()
		return client.embed_texts(texts)

	def add_chunks(self, document_id: Optional[str], chunks: List[str], metadata: Optional[Dict] = None) -> str:
		"""Add precomputed chunks for a document."""
		if not chunks:
			raise ValueError("No chunks provided")
		document_id = document_id or str(uuid.uuid4())
		ids: IDs = [f"{document_id}:{i}" for i in range(len(chunks))]
		metas: Metadatas = [{"document_id": document_id, **(metadata or {})} for _ in chunks]
		self._collection.add(ids=ids, documents=chunks, metadatas=metas)
		logger.info("Indexed %d provided chunks for document_id=%s", len(ids), document_id)
		return document_id

	def delete_document(self, document_id: str) -> int:
		"""Delete all chunks for a document_id. Returns number of deleted items (best effort)."""
		# Chroma returns None; we approximate by counting before and after
		before = self._count_for_document(document_id)
		self._collection.delete(where={"document_id": document_id})
		after = self._count_for_document(document_id)
		logger.info("Deleted ~%d chunks for document_id=%s", max(0, before - after), document_id)
		return max(0, before - after)

	def _count_for_document(self, document_id: str) -> int:
		res = self._collection.get(where={"document_id": document_id}, include=["metadatas"], limit=1000000)
		return len(res.get("ids", []))

	def list_documents(self) -> List[Tuple[str, Dict]]:
		"""Return unique (document_id, sample_metadata) pairs."""
		res = self._collection.get(include=["metadatas"], limit=1000000)
		unique: Dict[str, Dict] = {}
		for meta in res.get("metadatas", []) or []:
			if not meta:
				continue
			doc_id = meta.get("document_id")
			if doc_id and doc_id not in unique:
				unique[doc_id] = meta
		return [(k, v) for k, v in unique.items()]

	def query(self, document_id: str, query_text: str, n_results: int = 4):
		"""Query within a document's chunks and return Chroma results."""
		logger.debug("Query: doc_id=%s n_results=%d question_len=%d", document_id, n_results, len(query_text or ""))
		# ids are always returned by Chroma; include only supported fields
		return self._collection.query(
			query_texts=[query_text],
			n_results=n_results,
			where={"document_id": document_id},
			include=["documents", "metadatas", "distances"],
		)


