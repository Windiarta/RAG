import os
import logging
from dotenv import load_dotenv
from typing import List, Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from .logging_config import configure_logging

logger = logging.getLogger(__name__)


class GeminiClient:
	"""Wrapper around Google Gemini for embeddings and generation."""

	def __init__(self, api_key: Optional[str] = None, embeddings_model: str = "models/gemini-embedding-001", generation_model: str = "gemini-2.0-flash-lite") -> None:
		configure_logging()
		load_dotenv()
		self.api_key = api_key or os.getenv("GEMINI_API_KEY")
		if not self.api_key:
			raise ValueError("GEMINI_API_KEY is not set in environment")
		genai.configure(api_key=self.api_key)
		self.embeddings_model = embeddings_model
		self.generation_model = generation_model
		self._gen_model = genai.GenerativeModel(self.generation_model)
		logger.info("GeminiClient initialized with model=%s, embeddings_model=%s", self.generation_model, self.embeddings_model)

	@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
	async def embed_texts(self, texts: List[str]) -> List[List[float]]:
		"""Return embeddings for a list of texts using Gemini embeddings model."""
		logger.debug("Embedding %d texts", len(texts))
		embeddings: List[List[float]] = []

		async def _embed_one(text):
			result = await genai.embed_content_async(model=self.embeddings_model, content=text)
			return result["embedding"] if isinstance(result, dict) else result.embedding

		async def _embed_all(texts):
			return [await _embed_one(text) for text in texts]

		embeddings = await _embed_all(texts)
		return embeddings

	@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
	def generate_answer(self, question: str, context: str, system_instruction: Optional[str] = None) -> str:
		"""Generate an answer grounded in provided context."""
		messages = []
		if system_instruction:
			messages.append({"role": "system", "content": system_instruction})
		# Provide explicit instruction for grounding
		prompt = (
			"You are a helpful assistant answering questions using the provided context only.\n"
			"If the answer isn't in the context, say you don't know.\n\n"
			f"Context:\n{context}\n\n"
			f"Question: {question}\n"
			"Answer:"
		)
		logger.debug("Generating answer; question_len=%d, context_len=%d", len(question or ""), len(context or ""))
		response = self._gen_model.generate_content(prompt)
		return (response.text or "").strip()


