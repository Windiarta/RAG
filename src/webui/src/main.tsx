import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import App from './pages/App'
import UploadWizard from './pages/UploadWizard'
import Chatbot from './pages/Chatbot'
import DocumentManager from './pages/DocumentManager'
import './styles.css'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <UploadWizard /> },
      { path: 'chat', element: <Chatbot /> },
      { path: 'documents', element: <DocumentManager /> },
    ],
  },
])

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)


