import React, { useEffect, useState } from 'react'
import { deleteByFilename, deleteDocument, listDocuments, searchDocumentsByFilename } from '../lib/api'

export default function DocumentManager() {
  const [docs, setDocs] = useState<{ document_id: string; metadata: any }[]>([])
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState('')

  async function refresh() {
    setLoading(true)
    try {
      const res = query.trim() ? await searchDocumentsByFilename(query.trim()) : await listDocuments()
      setDocs(res.documents)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { refresh() }, [])

  async function onDelete(id: string) {
    if (!confirm(`Hapus dokumen ${id}?`)) return
    await deleteDocument(id)
    await refresh()
  }

  return (
    <div className="card">
      <div className="toolbar" style={{ justifyContent: 'space-between' }}>
        <h3 style={{ margin: 0 }}>Dokumen</h3>
        <div className="toolbar">
          <input className="input" placeholder="Cari berdasarkan filename..." value={query} onChange={e => setQuery(e.target.value)} />
          <button className="btn" onClick={refresh}>Cari / Refresh</button>
        </div>
      </div>
      {loading ? (
        <div className="muted" style={{ marginTop: 12 }}>Memuat...</div>
      ) : (
        <div className="scroll">
          <table className="table" style={{ marginTop: 8 }}>
            <thead>
              <tr>
                <th>Filename</th>
                <th>Metadata</th>
                <th>Aksi</th>
              </tr>
            </thead>
            <tbody>
              {docs.map(doc => (
                <tr key={doc.document_id}>
                  <td><h4>{doc.metadata?.filename || doc.document_id}</h4></td>
                  <td>
                    <pre style={{ margin: 0 }}>{JSON.stringify(doc.metadata || {}, null, 2)}</pre>
                  </td>
                  <td>
                    <button className="btn btn-danger" onClick={() => onDelete(doc.document_id)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}


