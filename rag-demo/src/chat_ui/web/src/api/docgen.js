import api from './client';

export const getTemplates = () =>
  api.get('/docgen/templates').then(r => r.data);

export const getTemplate = (id) =>
  api.get(`/docgen/templates/${id}`).then(r => r.data);

export const generateDocument = (template_id, fields) =>
  api.post('/docgen/generate', { template_id, fields }).then(r => r.data);

export const getDocumentHistory = () =>
  api.get('/docgen/history').then(r => r.data);
