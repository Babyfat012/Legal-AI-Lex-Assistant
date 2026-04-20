import { useState, useEffect } from 'react';
import { getDocumentHistory } from '../../api/docgen';
import { strings } from '../../i18n/strings';

export default function DocumentHistory({ lang, onClose }) {
  const t = strings[lang];
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const data = await getDocumentHistory();
        setDocuments(data);
      } catch (error) {
        console.error('Failed to load document history:', error);
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, []);

  const handleDownload = (doc) => {
    // Create blob from base64 content
    const byteCharacters = atob(doc.file_content);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: doc.mime_type });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = doc.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('vi-VN') + ' ' + date.toLocaleTimeString('vi-VN', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>📄 {t.docHistory || 'Document History'}</h2>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          {loading ? (
            <div className="typing-indicator" style={{ justifyContent: 'center' }}>
              <div className="typing-dot" /><div className="typing-dot" /><div className="typing-dot" />
            </div>
          ) : documents.length === 0 ? (
            <p style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
              {t.noDocuments || 'No documents generated yet'}
            </p>
          ) : (
            <div className="document-list">
              {documents.map(doc => (
                <div key={doc.id} className="document-item">
                  <div className="document-info">
                    <div className="document-name">{doc.filename}</div>
                    <div className="document-meta">
                      <span className="document-template">{doc.template_name}</span>
                      <span className="document-date">{formatDate(doc.created_at)}</span>
                    </div>
                  </div>
                  <button
                    className="btn btn-primary"
                    onClick={() => handleDownload(doc)}
                  >
                    📥 {t.download || 'Download'}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}