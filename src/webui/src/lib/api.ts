export type UploadResponse = { files: string[]; use_ocr: boolean }
export type ChunkResponse = { chunks: string[] }
export type SaveResponse = { document_id: string }
export type RetrieveResponse = { results: { id: string; metadata: any; chunk: string }[] }
export type AnswerResponse = { answer: string; sources: { id: string; metadata: any; chunk: string }[] }
export type DocumentsResponse = { documents: { document_id: string; metadata: any }[] }
export type UploadAllResponse = {
  files: string[]
  use_ocr: boolean
  method: string
  chunk_size: number
  overlap: number
  num_chunks: number
  chunks: string[]
  document_id: string
}
export type AppConfig = { ui: { default_ocr?: boolean }; vector_store: any }

const apiBase = '/api'

export async function uploadFiles(files: File[], useOcr: boolean): Promise<UploadResponse> {
  const form = new FormData()
  files.forEach(f => form.append('files', f))
  form.append('use_ocr', String(useOcr))
  const res = await fetch(`${apiBase}/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function chunkFile(params: { method: string; chunk_size: number; overlap: number; file_path: string; use_ocr: boolean }): Promise<ChunkResponse> {
  const form = new FormData()
  form.append('method', params.method)
  form.append('chunk_size', String(params.chunk_size))
  form.append('overlap', String(params.overlap))
  form.append('file_path', params.file_path)
  form.append('use_ocr', String(params.use_ocr))
  const res = await fetch(`${apiBase}/chunk`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function saveChunks(chunks: string[], metadata: Record<string, any>): Promise<SaveResponse> {
  const form = new FormData()
  form.append('chunks', JSON.stringify(chunks))
  form.append('metadata', JSON.stringify(metadata))
  const res = await fetch(`${apiBase}/save`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function uploadAll(params: {
  files: File[]
  use_ocr: boolean
  method?: string
  chunk_size?: number
  overlap?: number
  metadata?: Record<string, any>
}): Promise<UploadAllResponse> {
  const form = new FormData()
  params.files.forEach(f => form.append('files', f))
  form.append('use_ocr', String(params.use_ocr))
  if (params.method) form.append('method', params.method)
  if (params.chunk_size != null) form.append('chunk_size', String(params.chunk_size))
  if (params.overlap != null) form.append('overlap', String(params.overlap))
  if (params.metadata) form.append('metadata', JSON.stringify(params.metadata || {}))
  const res = await fetch(`${apiBase}/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function fetchConfig(): Promise<AppConfig> {
  const res = await fetch(`${apiBase}/config`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function listDocuments(): Promise<DocumentsResponse> {
  const res = await fetch(`${apiBase}/documents`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function searchDocumentsByFilename(q: string): Promise<DocumentsResponse> {
  const res = await fetch(`${apiBase}/documents?filename=${encodeURIComponent(q)}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function deleteDocument(documentId: string): Promise<{ deleted: number; document_id: string }> {
  const res = await fetch(`${apiBase}/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function deleteByFilename(filename: string): Promise<{ filename: string; deleted_chunks: number; documents: string[] }> {
  const res = await fetch(`${apiBase}/documents/by-filename/${encodeURIComponent(filename)}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function retrieve(question: string, top_k: number, filters: Record<string, any> = {}): Promise<RetrieveResponse> {
  const form = new FormData()
  form.append('question', question)
  form.append('filters', JSON.stringify(filters))
  form.append('top_k', String(top_k))
  const res = await fetch(`${apiBase}/retrieve`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function answer(question: string, top_k: number, filters: Record<string, any> = {}): Promise<AnswerResponse> {
  const form = new FormData()
  form.append('question', question)
  form.append('prompt', '')
  form.append('filters', JSON.stringify(filters))
  form.append('top_k', String(top_k))
  const res = await fetch(`${apiBase}/answer`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}


