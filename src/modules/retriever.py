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

	def ask(self, question: str, document_id: Optional[str], top_k: int = 4, system_instruction: Optional[str] = None) -> Dict:
		if document_id is None:
			# Get all document IDs
			# Just get the unique document IDs from the first query result
			results = self.store.query(document_id=None, query_text=question, n_results=top_k)
			docs = (results.get("documents") or [[]])[0]
			metas = (results.get("metadatas") or [[]])[0]
			ids = (results.get("ids") or [[]])[0]
			# Find unique document_ids from metadatas
			unique_doc_ids = set()
			for meta in metas:
				if meta and "document_id" in meta:
					unique_doc_ids.add(meta["document_id"])
			all_docs = []
			for doc_id in unique_doc_ids:
				# Optionally, get meta for doc_id from list_documents
				meta = next((m for d, m in self.store.list_documents() if d == doc_id), {})
				all_docs.append((doc_id, meta))
			if not all_docs:
				logger.warning("No documents found in store.")
				return {"answer": "", "sources": []}

			answers = []
			for doc_id, _ in all_docs:
				results = self.store.query(document_id=doc_id, query_text=question, n_results=top_k)
				docs: List[str] = (results.get("documents") or [[]])[0]
				metas: List[Dict] = (results.get("metadatas") or [[]])[0]
				ids: List[str] = (results.get("ids") or [[]])[0]
				context = "\n---\n".join(d for d in docs if d)
				answers.append(self.llm.generate_answer(question=question, context=context, system_instruction=system_instruction))
			
			# Choose the best answer using llm.generate_answer
			# Use the LLM to select the best answer from the list of generated answers
			# by providing the question and all candidate answers as context.
			selection_prompt = (
				"You are given a question and several candidate answers generated from different documents. "
				"Choose the best answer to the question from the candidates below. "
				"Return only the best answer verbatim.\n\n"
				f"Question: {question}\n\n"
				"Candidate Answers:\n"
			)
			for idx, ans in enumerate(answers, 1):
				selection_prompt += f"{idx}. {ans}\n"
			# Ask the LLM to select the best answer
			answer = self.llm.generate_answer(
				question=question,
				context=selection_prompt,
				system_instruction="Select the best answer from the candidates above and return it verbatim."
			)
			
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
		else:
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


