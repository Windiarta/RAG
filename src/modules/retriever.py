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
			results = self.store.manager_query(query_text=question, n_results=top_k)
			docs = (results.get("documents") or [[]])[0]
			metas = (results.get("metadatas") or [[]])[0]
			ids = (results.get("ids") or [[]])[0]
			
			if not docs:
				logger.warning("No documents found in store.")
				return {"answer": "", "sources": []}

			selection_prompt = (
				"You are given a question and several candidate agent expertise in different documents. "
				"Choose the best agent to answer from the candidates below. "
				"The given agent contain all data about the document (filename). "
				"Return only a number of id of the best answer agent ID without any other character.\n\n"
				f"Question: {question}\n\n"
				"Candidate Agent:\n"
			)

			for idx, _ in enumerate(ids):
				selection_prompt += f"{idx+1}. {metas[idx]}\n"

			logger.info(selection_prompt)
			selected_document = self.llm.generate_answer(
				question=question, 
				context=selection_prompt,
				system_instruction="Given list of the agents, choose the most relevant agent to the question. Expected output: a number"
			)
			try:
				selected_document = int(selected_document)
				document_id = ids[selected_document-1][:-2]
			except:
				document_id = ids[0][:-2]

		logger.debug("Ask: doc_id=%s top_k=%d question_len=%d", document_id, top_k, len(question or ""))
		results = self.store.query(document_id=document_id, query_text=question, n_results=top_k)
		docs: List[str] = (results.get("documents") or [[]])[0]
		metas: List[Dict] = (results.get("metadatas") or [[]])[0]
		ids: List[str] = (results.get("ids") or [[]])[0]
		pairs = []
		for d, i in zip(docs, ids):
			if not (d and i):
				continue
			num = int(i.rsplit(":", 1)[-1].strip()) + 1
			pairs.append((num, d))

		pairs.sort(key=lambda x: x[0])

		result_blocks = []
		current_start = None
		current_end = None
		current_docs = []

		for num, d in pairs:
			if current_start is None:
				current_start = current_end = num
				current_docs = [d]
			elif num == current_end + 1:
				current_end = num
				current_docs.append(d)
			else: 
				result_blocks.append(f"{' '.join(current_docs)}")
				current_start = current_end = num
				current_docs = [d]

		if current_start is not None:
			result_blocks.append(f"{' '.join(current_docs)}")

		context = "\n---\n".join(result_blocks)
		context = (
			"Anda adalah seorang pakar hukum profesional. "
			"Anda bertugas untuk mencari informasi mengenai pertanyaan. "
			f"Untuk menjawabnya, anda diberikan dokumen referensi: {metas[0].get('filename')}. "
			f"Pertanyaan: {question}. \n"
			"Potongan dokumen referensi: \n\n"
		) + context

		answer = self.llm.generate_answer(
			question=question, 
			context=context, 
			system_instruction=system_instruction
		)
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


