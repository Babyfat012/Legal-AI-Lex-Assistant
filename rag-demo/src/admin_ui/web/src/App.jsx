import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import LoginPage from './components/Auth/LoginPage';
import Sidebar from './components/Layout/Sidebar';
import UploadPage from './pages/UploadPage';
import DocumentsPage from './pages/DocumentsPage';
import CollectionPage from './pages/CollectionPage';

export default function App() {
  const [authenticated, setAuthenticated] = useState(() => {
    return localStorage.getItem('lex_admin_auth') === 'true';
  });

  const handleLogin = () => setAuthenticated(true);

  const handleLogout = () => {
    localStorage.removeItem('lex_admin_auth');
    setAuthenticated(false);
  };

  if (!authenticated) {
    return (
      <>
        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: '#1e293b',
              color: '#f1f5f9',
              border: '1px solid rgba(255,255,255,0.08)',
              fontSize: '14px',
              fontFamily: 'Inter, sans-serif',
            },
          }}
        />
        <LoginPage onLogin={handleLogin} />
      </>
    );
  }

  return (
    <BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1e293b',
            color: '#f1f5f9',
            border: '1px solid rgba(255,255,255,0.08)',
            fontSize: '14px',
            fontFamily: 'Inter, sans-serif',
          },
          success: {
            iconTheme: { primary: '#22c55e', secondary: '#1e293b' },
          },
          error: {
            iconTheme: { primary: '#ef4444', secondary: '#1e293b' },
          },
        }}
      />
      <div className="app-layout">
        <Sidebar onLogout={handleLogout} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="/collection" element={<CollectionPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
