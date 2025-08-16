# Architecture

- `src/modules/gemini.py`: wrapper for Gemini embeddings and generation
- `src/modules/chroma.py`: ChromaDB vector store wrapper
- `src/modules/retriever.py`: retrieval + generation
- `src/modules/ocr.py`: simple text extraction for PDF/TXT/MD
- `src/streamlit/*`: Streamlit pages (chatbot, documents)

# RAG Flow

1. User uploads file → text extraction → chunking → embeddings → stored in Chroma
2. User selects a document and asks question → top-k chunks retrieved → Gemini answers grounded on context
3. User can delete a document → all chunks removed from collection

# Notes

- For PDFs we use PyPDF2 text extraction; complex scanned PDFs may require OCR (not included).
- Embeddings model: `text-embedding-004`. Generation: `gemini-2.0-flash`.

