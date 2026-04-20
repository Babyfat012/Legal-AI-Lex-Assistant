import { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import LoginPage from './components/Auth/LoginPage';
import ChatLayout from './components/Chat/ChatLayout';

export default function App() {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('lex_user');
    return saved ? JSON.parse(saved) : null;
  });
  const [lang, setLang] = useState('vi');

  const handleLogin = (userData, token) => {
    localStorage.setItem('lex_token', token);
    localStorage.setItem('lex_user', JSON.stringify(userData));
    setUser(userData);
  };

  const handleLogout = () => {
    localStorage.removeItem('lex_token');
    localStorage.removeItem('lex_user');
    setUser(null);
  };

  const toggleLang = () => setLang(l => l === 'vi' ? 'en' : 'vi');

  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: { background: '#1e293b', color: '#f1f5f9', border: '1px solid rgba(255,255,255,0.08)', fontSize: '14px', fontFamily: 'Inter, sans-serif' },
          success: { iconTheme: { primary: '#22c55e', secondary: '#1e293b' } },
          error: { iconTheme: { primary: '#ef4444', secondary: '#1e293b' } },
        }}
      />
      {user ? (
        <ChatLayout user={user} lang={lang} onToggleLang={toggleLang} onLogout={handleLogout} />
      ) : (
        <LoginPage lang={lang} onLogin={handleLogin} />
      )}
    </>
  );
}
