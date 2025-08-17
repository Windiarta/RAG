from chunk import Chunk
import io
import os
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import yaml

from modules.logging_config import configure_logging
from modules.ocr import extract_text, Chunks, ChunkingMethod
from modules.chroma import VectorStore
from modules.retriever import Retriever


configure_logging()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = ROOT / "src" / "api" / "config.yaml"
if CONFIG_PATH.exists():
	CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
else:
	CONFIG = {
		"ui": {
			"default_ocr": False,
			"default_chunk_method": "recursive",
			"default_chunk_size": 1000,
			"default_overlap": 200,
			"default_max_retrieve": 4,
			"default_system_instruction": "You are a helpful assistant. Answer only using the retrieved context.",
		},
		"vector_store": {
			"collection_name": "rag_documents",
			"persist_path": str(DATA_DIR / "chroma"),
		},
	}

app = FastAPI(title="RAG Web UI")

templates = Jinja2Templates(directory=str(ROOT / "api" / "templates"))

# Mount built SPA (webui/dist) if present
# Serve built SPA from src/webui/dist
WEB_DIST = ROOT / "src" / "webui" / "dist"
if WEB_DIST.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIST)), name="web")

def spa_index() -> HTMLResponse:
    index_path = WEB_DIST / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    # Fallback to server-rendered template if SPA not built
    return templates.TemplateResponse("wizard.html", {"request": Request, "config": CONFIG["ui"]})


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if WEB_DIST.exists():
        return spa_index()
    return templates.TemplateResponse("wizard.html", {"request": request, "config": CONFIG["ui"]})


@app.get("/api/config")
async def api_config():
	return {"ui": CONFIG.get("ui", {}), "vector_store": CONFIG.get("vector_store", {})}


@app.post("/api/upload")
async def api_upload(
	files: List[UploadFile] = File(...),
	use_ocr: bool = Form(CONFIG["ui"].get("default_ocr", False)),
	method: str = Form(CONFIG["ui"].get("default_chunk_method", "recursive")),
	chunk_size: int = Form(CONFIG["ui"].get("default_chunk_size", 1000)),
	overlap: int = Form(CONFIG["ui"].get("default_overlap", 200)),
	metadata: str = Form("{}"),
):
	import json
	# 1) Simpan file
	saved_paths: List[str] = []
	for f in files:
		path = UPLOAD_DIR / f.filename
		content = await f.read()
		path.write_bytes(content)
		saved_paths.append(str(path))
	logger.info("Uploaded %d files", len(saved_paths))
	# 2) Ekstrak teks (OCR opsi), chunk per file, dan simpan per dokumen dengan metadata filename
	try:
		meta_base = json.loads(metadata or "{}")
		store = VectorStore(
			persist_path=CONFIG["vector_store"]["persist_path"],
			collection_name=CONFIG["vector_store"]["collection_name"],
		)
		documents = []
		total_chunks = 0
		for p in saved_paths:
			filename = Path(p).name
			try:
				text = await extract_text(p, use_ocr)
				chunks = Chunks(ChunkingMethod.PARAGRAPH).chunk_text(text)
				doc_id = store.add_chunks(None, chunks, metadata={"filename": filename, **(meta_base or {})})
				documents.append({"document_id": doc_id, "filename": filename, "num_chunks": len(chunks)})
				total_chunks += len(chunks)
			except Exception as e:
				logger.exception("Failed processing file %s", p)
				raise HTTPException(status_code=400, detail=f"Failed processing {p}: {e}")
	except HTTPException:
		raise
	except Exception as e:
		logger.exception("Failed saving chunks")
		raise HTTPException(status_code=500, detail=str(e))
	return {
		"files": saved_paths,
		"use_ocr": use_ocr,
		"method": method,
		"chunk_size": chunk_size,
		"overlap": overlap,
		"num_chunks": total_chunks,
		"document_id": documents[0]["document_id"] if documents else None,
		"documents": documents,
	}

@app.post("/api/save")
async def api_save(chunks: List[str] = Form(...), metadata: str = Form("{}")):
	import json
	meta = json.loads(metadata or "{}")
	store = VectorStore(persist_path=CONFIG["vector_store"]["persist_path"], collection_name=CONFIG["vector_store"]["collection_name"])
	doc_id = store.add_chunks(None, chunks, metadata=meta)
	return {"document_id": doc_id}


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui(request: Request):
    if WEB_DIST.exists():
        return spa_index()
    return templates.TemplateResponse("chat.html", {"request": request, "config": CONFIG["ui"]})


@app.post("/api/retrieve")
async def api_retrieve(question: str = Form(...), filters: str = Form("{}"), top_k: int = Form(4)):
	import json
	flt = json.loads(filters or "{}")
	store = VectorStore(persist_path=CONFIG["vector_store"]["persist_path"], collection_name=CONFIG["vector_store"]["collection_name"])
	# If filter contains document_id, use that; else query across all – we approximate by listing and merging
	if "document_id" in flt:
		doc_ids = [flt["document_id"]]
	else:
		doc_ids = [doc_id for doc_id, _ in store.list_documents()]
	all_results = []
	for did in doc_ids:
		res = store.query(document_id=did, query_text=question, n_results=top_k)
		docs = (res.get("documents") or [[]])[0]
		ids = (res.get("ids") or [[]])[0]
		metas = (res.get("metadatas") or [[]])[0]
		for i, d in enumerate(docs):
			all_results.append({"id": ids[i], "metadata": metas[i], "chunk": d})
	return {"results": all_results[:top_k]}


@app.post("/api/answer")
async def api_answer(question: str = Form(...), prompt: str = Form(""), filters: str = Form("{}"), top_k: int = Form(4)):
	import json
	flt = json.loads(filters or "{}")
	retriever = Retriever()
	# Limit to a single document if provided
	if "document_id" in flt:
		result = retriever.ask(question=question, document_id=flt["document_id"], top_k=top_k)
	else:
		# If none specified, pick first document
		store = retriever.store
		docs = store.list_documents()
		if not docs:
			raise HTTPException(status_code=400, detail="No documents available")
		result = retriever.ask(question=question, document_id=docs[0][0], top_k=top_k)
	return result



# Document management endpoints
@app.get("/api/documents")
async def api_list_documents(filename: Optional[str] = Query(None)):
	store = VectorStore(persist_path=CONFIG["vector_store"]["persist_path"], collection_name=CONFIG["vector_store"]["collection_name"])
	docs = store.list_documents()
	if filename:
		fl = filename.lower()
		docs = [(doc_id, meta) for doc_id, meta in docs if (meta or {}).get("filename", "").lower().find(fl) != -1]
	return {"documents": [{"document_id": d[0], "metadata": d[1]} for d in docs]}


@app.delete("/api/documents/{document_id}")
async def api_delete_document(document_id: str):
	store = VectorStore(persist_path=CONFIG["vector_store"]["persist_path"], collection_name=CONFIG["vector_store"]["collection_name"])
	deleted = store.delete_document(document_id)
	return {"deleted": deleted, "document_id": document_id}


# Catch-all to serve SPA client-side routes
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def spa_catch_all(full_path: str):
    if WEB_DIST.exists():
        return spa_index()
    # Fallback to home template when SPA not built
    return templates.TemplateResponse("wizard.html", {"request": Request, "config": CONFIG["ui"]})


