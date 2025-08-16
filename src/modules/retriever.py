import logging
from typing import Dict, List, Optional

from .chroma import VectorStore
from .gemini import GeminiClient
from .logging_config import configure_logging

logger = logging.getLogger(__name__)


class Retriever:
	"""RAG pipeline: retrieve relevant chunks and generate an answer."""

	def __init__(self, store: Optional[VectorStore] = None, llm: Optional[GeminiClient] = None) -> None:
		configure_logging()
		self.store = store or VectorStore()
		self.llm = llm or GeminiClient()
		logger.info("Retriever initialized")

	def ask(self, question: str, document_id: str, top_k: int = 4, system_instruction: Optional[str] = None) -> Dict:
		logger.debug("Ask: doc_id=%s top_k=%d question_len=%d", document_id, top_k, len(question or ""))
		results = self.store.query(document_id=document_id, query_text=question, n_results=top_k)
		docs: List[str] = (results.get("documents") or [[]])[0]
		metas: List[Dict] = (results.get("metadatas") or [[]])[0]
		ids: List[str] = (results.get("ids") or [[]])[0]
		context = "\n---\n".join(d for d in docs if d)
		answer = self.llm.generate_answer(question=question, context=context, system_instruction=system_instruction)
		logger.info("Answer generated with %d sources", len(docs))
		return {
			"answer": answer,
			"sources": [
				{
					"id": ids[i] if i < len(ids) else None,
					"metadata": metas[i] if i < len(metas) else {},
					"chunk": docs[i] if i < len(docs) else "",
				}
				for i in range(len(docs))
			],
		}


