# RAG (Retrieval-Augmented Generation) Web UI

RAG is an application for document management, text extraction (PDF/TXT/MD), chunking, vector indexing with ChromaDB, and context-based question answering using Google Gemini. It features a modern WebUI (React) and a FastAPI backend.

## Features
- Upload documents (PDF, TXT, MD)
- Automatic text extraction (optional OCR for PDFs)
- Text chunking
- Vector storage in ChromaDB
- Contextual search & question answering (RAG)
- Document deletion & metadata management
- Modern WebUI (React + Vite)
- FastAPI endpoints (upload, retrieve, answer, document management)
- Docker support

## Architecture Overview
- **Backend**: FastAPI, ChromaDB, Google Gemini API
- **Frontend**: React (Vite), communicates with backend via REST API
- **Storage**: ChromaDB (persistent at `data/chroma`)
- **OCR**: Google Gemini Vision (optional)

## Installation & Running

### 1. Prepare `.env`
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_google_gemini_api_key
```

### 2. Run with Docker
```
docker compose up --build
```
Access the WebUI at: http://localhost:8000/web/

### 3. Run for Development
- Install Python dependencies:
  ```
  pip install -r requirements.txt
  ```
- Start the backend:
  ```
  uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
  ```
- Start the frontend (WebUI):
  ```
  cd src/webui
  npm install
  npm run dev
  # WebUI: http://localhost:5173 (proxy to backend)
  ```

## Usage
### WebUI
- Upload documents, optionally set a name
- Select a document and ask questions in the Chatbot
- View, search, and delete documents in the Document Manager

### API Endpoints (FastAPI)
- `/api/upload` : Upload documents
- `/api/retrieve` : Retrieve top-k relevant chunks
- `/api/answer` : Contextual Q&A (RAG)
- `/api/documents` : List documents
- `/api/documents/{document_id}` : Delete document

## Environment & Configuration
- Default config in `src/api/config.yaml`
- Data & vectors stored in the `data/` folder
- Key variables: `GEMINI_API_KEY`, `CHROMA_PATH`, `UPLOAD_DIR`

## Testing
```
pip install -r requirements.txt
pytest -q src/test/
```

## Notes
- ChromaDB persists automatically in `data/chroma`
- WebUI is built automatically during Docker build (if Node/NPM available)

---
Contributions & feedback are welcome!

