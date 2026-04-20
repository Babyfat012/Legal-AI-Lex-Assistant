import api from './client';

export const listConversations = () =>
  api.get('/conversations').then(r => r.data);

export const createConversation = (title) =>
  api.post('/conversations', { title }).then(r => r.data);

export const getConversation = (id) =>
  api.get(`/conversations/${id}`).then(r => r.data);

export const deleteConversation = (id) =>
  api.delete(`/conversations/${id}`).then(r => r.data);
