import React, { useEffect, useMemo, useState } from 'react'
import { fetchConfig, uploadAll } from '../lib/api'

export default function UploadWizard() {
  const [files, setFiles] = useState<File[]>([])
  const [useOcr, setUseOcr] = useState(false)
  const [busy, setBusy] = useState(false)
  const [savedId, setSavedId] = useState<string>('')
  const [numChunks, setNumChunks] = useState<number>(0)
  const [docName, setDocName] = useState<string>('')
  const pdfPreview = useMemo(() => files.find(f => f.type === 'application/pdf'), [files])

  useEffect(() => {
    fetchConfig()
      .then(cfg => setUseOcr(!!cfg.ui?.default_ocr))
      .catch(() => {})
  }, [])

  async function handleUpload() {
    try {
      setBusy(true)
      const metadata = docName.trim() ? { filename: docName.trim() } : undefined
      const res = await uploadAll({ files, use_ocr: useOcr, metadata })
      setSavedId(res.document_id)
      setNumChunks(res.num_chunks)
    } catch (e: any) {
      alert(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="grid grid-2">
      <div className="card">
        <h3>Upload Dokumen</h3>
        <div className="field">
          <label>Pilih file</label>
          <input className="input" type="file" multiple accept=".pdf,.txt,.md,.markdown,.json,.csv,.tsv" onChange={e => setFiles(Array.from(e.target.files || []))} />
        </div>
        <div className="field">
          <label>Nama dokumen (opsional)</label>
          <input className="input" placeholder="Masukkan nama untuk pencarian/penghapusan" value={docName} onChange={e => setDocName(e.target.value)} />
        </div>
        <div className="toolbar">
          <label className="muted" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input type="checkbox" checked={useOcr} onChange={e => setUseOcr(e.target.checked)} /> Gunakan OCR (default dari backend)
          </label>
          <button className="btn btn-primary" disabled={!files.length || busy} onClick={handleUpload}>Upload</button>
        </div>
        {savedId && (
          <div style={{ marginTop: 12 }}>
            <div className="muted">Berhasil tersimpan.</div>
            <div>document_id: <code>{savedId}</code></div>
            <div>Total chunk: {numChunks}</div>
          </div>
        )}
      </div>
      <div className="card">
        <h4>Preview PDF</h4>
        <div className="preview">
          {pdfPreview ? (
            <embed src={URL.createObjectURL(pdfPreview)} type="application/pdf" width="100%" height="100%" />
          ) : (
            <div className="muted">Tidak ada PDF terpilih</div>
          )}
        </div>
      </div>
    </section>
  )
}
