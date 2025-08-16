import logging
from pathlib import Path
from typing import Optional

from PyPDF2 import PdfReader
from .logging_config import configure_logging

logger = logging.getLogger(__name__)


def extract_text(file_path: str) -> str:
    """Extract text from simple file types (PDF, TXT)."""
    configure_logging()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    text = ""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                logger.exception("Failed to extract text from a PDF page: %s", path.name)
                pages_text.append("")
        text = "\n\n".join(pages_text)
    elif path.suffix.lower() in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        # Unsupported type; return empty string to be handled by caller
        text = ""

    logger.info("DEBUG: Extracted text: %s", text)
    logger.info("Extracted %d chars from %s", len(text), path.name)
    return text.strip()


