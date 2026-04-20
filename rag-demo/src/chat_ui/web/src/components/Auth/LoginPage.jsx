import { useState } from 'react';
import { login, register } from '../../api/auth';
import { strings } from '../../i18n/strings';

export default function LoginPage({ lang, onLogin }) {
  const t = strings[lang];
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      let data;
      if (isRegister) {
        data = await register(email, password, fullName);
      } else {
        data = await login(email, password);
      }
      onLogin(data.user, data.access_token);
    } catch (err) {
      setError(err.response?.data?.detail || 'Có lỗi xảy ra');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-icon">⚖️</div>
        <h1 className="login-title">Lex</h1>
        <p className="login-subtitle">Trợ lý pháp lý AI</p>
        <form className="login-form" onSubmit={handleSubmit}>
          {isRegister && (
            <div className="input-group">
              <label>{t.fullName}</label>
              <input className="input-field" value={fullName} onChange={e => setFullName(e.target.value)} placeholder="Nguyễn Văn A" />
            </div>
          )}
          <div className="input-group">
            <label>{t.email}</label>
            <input className="input-field" type="email" required value={email} onChange={e => setEmail(e.target.value)} placeholder="email@example.com" />
          </div>
          <div className="input-group">
            <label>{t.password}</label>
            <input className="input-field" type="password" required value={password} onChange={e => setPassword(e.target.value)} placeholder="••••••••" />
          </div>
          {error && <p className="login-error">{error}</p>}
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? '...' : isRegister ? t.register : t.login}
          </button>
        </form>
        <div className="login-switch">
          {isRegister ? t.hasAccount : t.noAccount}{' '}
          <button onClick={() => { setIsRegister(!isRegister); setError(''); }}>
            {isRegister ? t.login : t.register}
          </button>
        </div>
      </div>
    </div>
  );
}
