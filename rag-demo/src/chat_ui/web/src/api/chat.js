import api from './client';

export const sendMessage = (payload) =>
  api.post('/chat', payload).then(r => r.data);

export const detectIntent = (query) =>
  api.post('/docgen/detect-intent', { query }).then(r => r.data);
