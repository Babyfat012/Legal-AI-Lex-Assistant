import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 min for large file uploads
});

// --- Health ---
export async function healthCheck() {
  const { data } = await api.get('/health');
  return data;
}

// --- Collection ---
export async function getCollectionInfo() {
  const { data } = await api.get('/collection/info');
  return data;
}

export async function deleteCollection() {
  const { data } = await api.delete('/collection');
  return data;
}

export async function deleteDocument(fileName) {
  const { data } = await api.delete(`/collection/documents/${encodeURIComponent(fileName)}`);
  return data;
}

// --- Ingest ---
export async function uploadFile(file, recreateCollection = false, sourceUrl = '', onProgress) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('recreate_collection', recreateCollection.toString());
  formData.append('source_url', sourceUrl);

  const { data } = await api.post('/ingest/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded * 100) / e.total));
      }
    },
  });
  return data;
}

export async function getIngestLogs() {
  const { data } = await api.get('/ingest/logs');
  return data;
}

export default api;
