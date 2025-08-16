# RAG Library and Streamlit App

## Run with Docker

1. Create a `.env` file:

```
GEMINI_API_KEY=your_key_here
```

2. Start:

```
docker compose up --build
```

App will be on `http://localhost:8501`.

## Development

- Run tests:

```
pip install -r requirements.txt
pytest -q
```

- Run Streamlit locally:

```
streamlit run src/streamlit/chats.py
```

## Features

- Upload documents (PDF/TXT/MD)
- Store chunks in ChromaDB
- Ask questions per selected document
- Delete documents
- Gemini 2.0 Flash for generation; text-embedding-004 for embeddings

