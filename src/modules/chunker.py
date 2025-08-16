import logging
import re
from typing import List

from .logging_config import configure_logging

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
	"""Split text into overlapping chunks.

	Parameters
	----------
	text: Input text to split.
	chunk_size: Target size of each chunk.
	overlap: Number of overlapping characters between chunks.
	"""
	configure_logging()
	if not text:
		return []
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	chunks: List[str] = []
	start = 0
	length = len(text)
	logger.debug("Chunking text length=%d chunk_size=%d overlap=%d", length, chunk_size, overlap)
	while start < length:
		end = min(start + chunk_size, length)
		chunk = text[start:end].strip()
		if chunk:
			chunks.append(chunk)
		if end == length:
			break
		start = max(0, end - overlap)
	logger.debug("Produced %d chunks", len(chunks))
	return chunks


def chunk_by_sentence(text: str, max_chars: int = 1000) -> List[str]:
	configure_logging()
	if not text:
		return []
	# Naive sentence splitter by punctuation
	sentences = re.split(r"(?<=[.!?])\s+", text)
	chunks: List[str] = []
	current = []
	current_len = 0
	for s in sentences:
		s = s.strip()
		if not s:
			continue
		if current_len + len(s) + (1 if current else 0) > max_chars:
			chunks.append(" ".join(current))
			current = [s]
			current_len = len(s)
		else:
			current.append(s)
			current_len += len(s) + (1 if current_len else 0)
	if current:
		chunks.append(" ".join(current))
	return chunks


def chunk_by_paragraph(text: str, max_chars: int = 1200) -> List[str]:
	configure_logging()
	paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text or "") if p.strip()]
	chunks: List[str] = []
	current = []
	current_len = 0
	for p in paragraphs:
		if current_len + len(p) + (2 if current else 0) > max_chars:
			chunks.append("\n\n".join(current))
			current = [p]
			current_len = len(p)
		else:
			current.append(p)
			current_len += len(p) + (2 if current_len else 0)
	if current:
		chunks.append("\n\n".join(current))
	return chunks


def chunk_by_markdown(text: str, max_chars: int = 1500) -> List[str]:
	configure_logging()
	lines = (text or "").splitlines()
	sections: List[str] = []
	current: List[str] = []
	for line in lines:
		if line.strip().startswith("#") and current:
			sections.append("\n".join(current).strip())
			current = [line]
		else:
			current.append(line)
	if current:
		sections.append("\n".join(current).strip())
	# Merge small sections up to max_chars
	chunks: List[str] = []
	buf = []
	buf_len = 0
	for sec in sections:
		if buf_len + len(sec) + 1 > max_chars and buf:
			chunks.append("\n".join(buf))
			buf = [sec]
			buf_len = len(sec)
		else:
			buf.append(sec)
			buf_len += len(sec) + 1
	if buf:
		chunks.append("\n".join(buf))
	return [c for c in chunks if c.strip()]


def chunk_recursive(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
	# alias to default
	return chunk_text(text, chunk_size=chunk_size, overlap=overlap)


