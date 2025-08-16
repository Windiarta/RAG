import React from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'

export default function App() {
  const { pathname } = useLocation()
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header className="header">
        <div className="container" style={{ display: 'flex', gap: 16, alignItems: 'center', padding: '12px 0' }}>
          <div className="brand">WIN RAG</div>
          <nav className="nav">
            <Link to="/" aria-current={pathname === '/' ? 'page' : undefined}>Upload</Link>
            <Link to="/chat" aria-current={pathname.startsWith('/chat') ? 'page' : undefined}>Chatbot</Link>
            <Link to="/documents" aria-current={pathname.startsWith('/documents') ? 'page' : undefined}>Manajer Dokumen</Link>
          </nav>
        </div>
      </header>
      <main className="container" style={{ flex: 1 }}>
        <Outlet />
      </main>
    </div>
  )
}


