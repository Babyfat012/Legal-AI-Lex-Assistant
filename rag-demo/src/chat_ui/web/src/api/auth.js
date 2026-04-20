import api from './client';

export const login = (email, password) =>
  api.post('/auth/login', { email, password }).then(r => r.data);

export const register = (email, password, full_name) =>
  api.post('/auth/register', { email, password, full_name }).then(r => r.data);

export const getMe = () => api.get('/auth/me').then(r => r.data);
