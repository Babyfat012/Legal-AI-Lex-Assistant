import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'https://legal-ai-lex-assistant-production.up.railway.app/api/v1';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 min for large file uploads
});

// Auto-attach JWT token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('lex_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Auto-logout on 401
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('lex_token');
      localStorage.removeItem('lex_user');
      window.location.reload();
    }
    return Promise.reject(err);
  }
);

export default api;
