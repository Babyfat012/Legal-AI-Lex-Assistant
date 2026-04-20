import axios from 'axios';

const api = axios.create({ baseURL: '/api/v1' });

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
