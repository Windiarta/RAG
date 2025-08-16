import React, { useEffect, useState } from 'react'
import { answer, listDocuments } from '../lib/api'

export default function Chatbot() {
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(4)
  const [documents, setDocuments] = useState<{ document_id: string; metadata: any }[]>([])
  const [selectedDoc, setSelectedDoc] = useState('')
  const [resp, setResp] = useState<{ answer: string; sources: { id: string; metadata: any; chunk: string }[] } | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    listDocuments().then(d => {
      setDocuments(d.documents)
      setSelectedDoc(d.documents[0]?.document_id || '')
    }).catch(err => console.error(err))
  }, [])

  async function ask() {
    try {
      setBusy(true)
      const filters = selectedDoc ? { document_id: selectedDoc } : {}
      const res = await answer(question, topK, filters)
      setResp(res)
    } catch (e: any) {
      alert(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid grid-2">
      <div className="card">
        <h3>Chatbot</h3>
        <div className="field">
          <label>Dokumen</label>
          <select className="select" value={selectedDoc} onChange={e => setSelectedDoc(e.target.value)}>
            <option value="">(Semua/auto)</option>
            {documents.map(d => (
              <option key={d.document_id} value={d.document_id}>{d.metadata?.filename || d.document_id}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Pertanyaan</label>
          <textarea className="textarea" rows={6} value={question} onChange={e => setQuestion(e.target.value)} />
        </div>
        <div className="field" style={{ maxWidth: 160 }}>
          <label>Top K</label>
          <input className="input" type="number" value={topK} onChange={e => setTopK(Number(e.target.value))} />
        </div>
        <div className="toolbar">
          <button className="btn btn-primary" disabled={!question.trim() || busy} onClick={ask}>Tanya</button>
        </div>
      </div>
      <div className="card">
        <h4>Jawaban</h4>
        {busy && <div className="muted">Menghasilkan jawaban...</div>}
        {resp && (
          <div className="scroll" style={{ padding: 10 }}>
            <div className="code">{resp.answer}</div>
            <h5 className="section-title">Sumber</h5>
            <ol>
              {resp.sources.map((s, i) => (
                <li key={i}>
                  <div className="muted" style={{ fontSize: 12 }}>{s.id}</div>
                  <div className="code" style={{ marginTop: 6 }}>{s.chunk}</div>
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    </div>
  )
}


