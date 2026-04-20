import { useState } from 'react';
import { strings } from '../../i18n/strings';

export default function DocumentCard({ document, lang }) {
  const t = strings[lang];
  const [preview, setPreview] = useState(null);

  const handleDownload = () => {
    // Create blob from base64 content
    const byteCharacters = atob(document.file_content);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: document.mime_type });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = document.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handlePreview = () => {
    setPreview(!preview);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('vi-VN') + ' ' + date.toLocaleTimeString('vi-VN', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="document-card">
      <div className="document-header">
        <div className="document-icon">📄</div>
        <div className="document-info">
          <div className="document-name">{document.filename}</div>
          <div className="document-meta">
            <span className="document-template">{document.template_name}</span>
            <span className="document-date">{formatDate(document.created_at)}</span>
          </div>
        </div>
      </div>

      <div className="document-actions">
        <button className="btn btn-primary" onClick={handleDownload}>
          📥 {t.download || 'Download'}
        </button>
        <button className="btn btn-ghost" onClick={handlePreview}>
          {preview ? '🔼 Hide Preview' : '👁️ Show Preview'}
        </button>
      </div>

      {preview && (
        <div className="document-preview">
          <h4>📋 {t.documentPreview || 'Document Preview'}</h4>
          <div className="preview-content">
            <p><strong>{t.templateName || 'Template'}:</strong> {document.template_name}</p>
            <p><strong>{t.generatedAt || 'Generated at'}:</strong> {formatDate(document.created_at)}</p>
            <div className="preview-fields">
              <h5>{t.fieldValues || 'Field Values'}:</h5>
              <pre>
                {JSON.stringify(document.field_values, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        .document-card {
          margin-top: 12px;
          padding: 16px;
          background: rgba(56, 189, 248, 0.06);
          border: 1px solid var(--border-accent);
          border-radius: var(--radius-md);
        }
        .document-header {
          display: flex;
          gap: 12px;
          margin-bottom: 12px;
        }
        .document-icon {
          font-size: 24px;
          opacity: 0.8;
        }
        .document-info {
          flex: 1;
        }
        .document-name {
          font-size: 14px;
          font-weight: 600;
          color: var(--text-primary);
          margin-bottom: 4px;
        }
        .document-meta {
          display: flex;
          gap: 12px;
          font-size: 12px;
          color: var(--text-muted);
        }
        .document-template {
          background: rgba(56, 189, 248, 0.1);
          color: var(--accent);
          padding: 2px 8px;
          border-radius: 4px;
          font-weight: 500;
        }
        .document-actions {
          display: flex;
          gap: 8px;
          margin-bottom: 12px;
        }
        .document-preview {
          background: var(--bg-input);
          border-radius: var(--radius-sm);
          padding: 12px;
          font-size: 13px;
        }
        .document-preview h4 {
          color: var(--accent);
          margin-bottom: 8px;
          font-size: 14px;
        }
        .document-preview p {
          margin-bottom: 8px;
          color: var(--text-secondary);
        }
        .preview-fields {
          margin-top: 8px;
        }
        .preview-fields h5 {
          color: var(--text-primary);
          margin-bottom: 4px;
          font-size: 12px;
        }
        .preview-fields pre {
          background: rgba(0, 0, 0, 0.2);
          padding: 8px;
          border-radius: 4px;
          font-size: 11px;
          overflow-x: auto;
          color: var(--text-primary);
        }
      `}</style>
    </div>
  );
}