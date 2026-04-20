import { useState } from 'react';
import { Lock, Eye, EyeOff } from 'lucide-react';

const ADMIN_SECRET = 'admin123'; // Matches backend default

export default function LoginPage({ onLogin }) {
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    // Simulate small delay for UX
    setTimeout(() => {
      if (password === ADMIN_SECRET) {
        localStorage.setItem('lex_admin_auth', 'true');
        onLogin();
      } else {
        setError('Sai mật khẩu. Vui lòng thử lại.');
        setLoading(false);
      }
    }, 400);
  };

  return (
    <div className="login-container">
      <div className="login-card glass-card">
        <span className="login-icon">⚖️</span>
        <h2 className="login-title">Lex Admin</h2>
        <p className="login-subtitle">Legal AI Dashboard</p>

        <form className="login-form" onSubmit={handleSubmit}>
          <div className="input-group">
            <label htmlFor="admin-password">Mật khẩu quản trị</label>
            <div style={{ position: 'relative' }}>
              <input
                id="admin-password"
                type={showPassword ? 'text' : 'password'}
                className="input-field"
                placeholder="Nhập mật khẩu..."
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoFocus
                style={{ paddingRight: '44px' }}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{
                  position: 'absolute',
                  right: '12px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--text-muted)',
                  padding: '4px',
                  display: 'flex',
                }}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>

          {error && (
            <div style={{
              padding: '10px 14px',
              background: 'var(--error-bg)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--error)',
              fontSize: '13px',
              fontWeight: '500',
              textAlign: 'left',
              animation: 'slideUp 0.3s ease',
            }}>
              {error}
            </div>
          )}

          <button
            type="submit"
            className="btn btn-primary"
            disabled={!password || loading}
            style={{ width: '100%', marginTop: '8px' }}
          >
            {loading ? (
              <>
                <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} />
                Đang xác thực...
              </>
            ) : (
              <>
                <Lock size={16} />
                Đăng nhập
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
