import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, FileText, File, X, Link, AlertTriangle, Rocket, FolderUp, FileUp } from 'lucide-react';
import toast from 'react-hot-toast';
import { uploadFile } from '../api/client';

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function getFileIcon(name) {
  if (name.endsWith('.pdf')) return { cls: 'pdf', label: 'PDF' };
  if (name.endsWith('.docx') || name.endsWith('.doc')) return { cls: 'docx', label: 'DOCX' };
  return { cls: 'docx', label: 'FILE' };
}

export default function UploadPage() {
  // Upload mode: 'single' = one file at a time, 'batch' = up to 3 files
  const [mode, setMode] = useState('single');

  // Single mode: one file + one URL
  const [singleFile, setSingleFile] = useState(null);
  const [singleUrl, setSingleUrl] = useState('');

  // Batch mode: up to 3 entries, each { file, sourceUrl }
  const [batchEntries, setBatchEntries] = useState([]);

  const [recreate, setRecreate] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState({});
  const [results, setResults] = useState([]);

  // --- Single mode dropzone ---
  const onDropSingle = useCallback((accepted) => {
    if (accepted.length > 0) {
      setSingleFile(accepted[0]);
      setResults([]);
    }
  }, []);

  const singleDropzone = useDropzone({
    onDrop: onDropSingle,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    multiple: false,
  });

  // --- Batch mode dropzone ---
  const onDropBatch = useCallback((accepted) => {
    setBatchEntries((prev) => {
      const remaining = 3 - prev.length;
      if (remaining <= 0) {
        toast.error('Tối đa 3 tài liệu trong chế độ Batch');
        return prev;
      }
      const toAdd = accepted.slice(0, remaining);
      if (accepted.length > remaining) {
        toast(`Chỉ thêm ${remaining} file (tối đa 3)`, { icon: '⚠️' });
      }
      return [...prev, ...toAdd.map((f) => ({ file: f, sourceUrl: '' }))];
    });
    setResults([]);
  }, []);

  const batchDropzone = useDropzone({
    onDrop: onDropBatch,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    multiple: true,
  });

  const removeBatchEntry = (index) => {
    setBatchEntries((prev) => prev.filter((_, i) => i !== index));
  };

  const updateBatchUrl = (index, url) => {
    setBatchEntries((prev) =>
      prev.map((entry, i) => (i === index ? { ...entry, sourceUrl: url } : entry))
    );
  };

  // --- Validation ---
  const canUploadSingle = singleFile && singleUrl.trim().length > 0;
  const canUploadBatch =
    batchEntries.length > 0 &&
    batchEntries.every((e) => e.sourceUrl.trim().length > 0);

  // --- Upload handler ---
  const handleUpload = async () => {
    const entries =
      mode === 'single'
        ? [{ file: singleFile, sourceUrl: singleUrl }]
        : batchEntries;

    if (entries.length === 0) return;

    setUploading(true);
    setResults([]);
    setProgress({});

    const uploadResults = [];

    for (let i = 0; i < entries.length; i++) {
      const { file, sourceUrl } = entries[i];
      const shouldRecreate = recreate && i === 0;

      try {
        setProgress((p) => ({ ...p, [file.name]: 0 }));

        const result = await uploadFile(
          file,
          shouldRecreate,
          sourceUrl.trim(),
          (pct) => setProgress((p) => ({ ...p, [file.name]: pct }))
        );

        setProgress((p) => ({ ...p, [file.name]: 100 }));
        uploadResults.push({ fileName: file.name, success: true, info: result });
        toast.success(`${file.name}: ${result.chunks_stored} chunks đã lưu`);
      } catch (err) {
        const errMsg = err.response?.data?.detail || err.message;
        uploadResults.push({ fileName: file.name, success: false, info: errMsg });
        toast.error(`${file.name}: ${errMsg}`);
      }
    }

    setResults(uploadResults);
    setUploading(false);

    if (uploadResults.every((r) => r.success)) {
      setTimeout(() => {
        if (mode === 'single') {
          setSingleFile(null);
          setSingleUrl('');
        } else {
          setBatchEntries([]);
        }
      }, 1500);
    }
  };

  const activeEntries =
    mode === 'single'
      ? singleFile ? [{ file: singleFile }] : []
      : batchEntries;

  const totalProgress =
    activeEntries.length > 0
      ? Math.round(
          Object.values(progress).reduce((a, b) => a + b, 0) / activeEntries.length
        )
      : 0;

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1>📤 Upload Tài Liệu</h1>
        <p>Ingest tài liệu pháp luật vào knowledge base qua pipeline Agentic RAG</p>
      </div>

      {/* Mode Selector */}
      <div className="mode-selector" style={{
        display: 'flex',
        gap: '12px',
        marginBottom: '24px',
      }}>
        <button
          className={`mode-btn ${mode === 'single' ? 'active' : ''}`}
          onClick={() => { setMode('single'); setResults([]); }}
          id="mode-single"
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            padding: '16px 20px',
            borderRadius: 'var(--radius-lg)',
            border: mode === 'single' ? '1px solid var(--border-accent)' : '1px solid var(--border-subtle)',
            background: mode === 'single' ? 'rgba(56, 189, 248, 0.08)' : 'var(--bg-card)',
            color: mode === 'single' ? 'var(--accent)' : 'var(--text-secondary)',
            cursor: 'pointer',
            fontFamily: 'inherit',
            fontSize: '14px',
            fontWeight: 600,
            transition: 'all var(--transition-fast)',
            boxShadow: mode === 'single' ? '0 0 12px rgba(56, 189, 248, 0.08)' : 'none',
          }}
        >
          <FileUp size={20} />
          <div style={{ textAlign: 'left' }}>
            <div>Upload Từng File</div>
            <div style={{ fontSize: '11px', fontWeight: 400, color: 'var(--text-muted)', marginTop: '2px' }}>
              1 tài liệu + 1 source URL
            </div>
          </div>
        </button>

        <button
          className={`mode-btn ${mode === 'batch' ? 'active' : ''}`}
          onClick={() => { setMode('batch'); setResults([]); }}
          id="mode-batch"
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            padding: '16px 20px',
            borderRadius: 'var(--radius-lg)',
            border: mode === 'batch' ? '1px solid var(--border-accent)' : '1px solid var(--border-subtle)',
            background: mode === 'batch' ? 'rgba(56, 189, 248, 0.08)' : 'var(--bg-card)',
            color: mode === 'batch' ? 'var(--accent)' : 'var(--text-secondary)',
            cursor: 'pointer',
            fontFamily: 'inherit',
            fontSize: '14px',
            fontWeight: 600,
            transition: 'all var(--transition-fast)',
            boxShadow: mode === 'batch' ? '0 0 12px rgba(56, 189, 248, 0.08)' : 'none',
          }}
        >
          <FolderUp size={20} />
          <div style={{ textAlign: 'left' }}>
            <div>Upload Batch</div>
            <div style={{ fontSize: '11px', fontWeight: 400, color: 'var(--text-muted)', marginTop: '2px' }}>
              Tối đa 3 tài liệu, mỗi file 1 URL
            </div>
          </div>
        </button>
      </div>

      {/* ====================== SINGLE MODE ====================== */}
      {mode === 'single' && (
        <div className="animate-fade-in">
          {/* Dropzone */}
          <div
            {...singleDropzone.getRootProps()}
            className={`dropzone ${singleDropzone.isDragActive ? 'active' : ''}`}
            id="upload-dropzone-single"
          >
            <input {...singleDropzone.getInputProps()} />
            <UploadCloud className="dropzone-icon" />
            <h3>
              {singleDropzone.isDragActive
                ? 'Thả file vào đây...'
                : singleFile
                  ? 'Click hoặc kéo thả để thay đổi file'
                  : 'Kéo thả file hoặc click để chọn'}
            </h3>
            <p>Hỗ trợ định dạng .pdf và .docx — 1 file duy nhất</p>
          </div>

          {/* Selected file + URL */}
          {singleFile && (
            <div className="upload-config animate-slide-up" style={{ marginTop: '16px' }}>
              {/* File info */}
              <div className="file-item" style={{ background: 'transparent', border: 'none', padding: '0' }}>
                <div className={`file-item-icon ${getFileIcon(singleFile.name).cls}`}>
                  {getFileIcon(singleFile.name).cls === 'pdf' ? <File size={18} /> : <FileText size={18} />}
                </div>
                <div className="file-item-info">
                  <div className="file-item-name">{singleFile.name}</div>
                  <div className="file-item-size">{formatFileSize(singleFile.size)}</div>
                  {progress[singleFile.name] !== undefined && (
                    <div className="progress-bar" style={{ marginTop: '6px' }}>
                      <div className="progress-bar-fill" style={{ width: `${progress[singleFile.name]}%` }} />
                    </div>
                  )}
                </div>
                {!uploading && (
                  <button
                    className="file-item-remove"
                    onClick={() => setSingleFile(null)}
                    title="Xóa file"
                  >
                    <X size={16} />
                  </button>
                )}
              </div>

              {/* Source URL (required) */}
              <div className="input-group" style={{ marginTop: '16px' }}>
                <label htmlFor="source-url-single">
                  <Link size={14} style={{ display: 'inline', verticalAlign: '-2px', marginRight: '6px' }} />
                  URL Nguồn Gốc <span style={{ color: 'var(--error)' }}>*</span>
                </label>
                <input
                  id="source-url-single"
                  type="url"
                  className="input-field"
                  placeholder="https://thuvienphapluat.vn/van-ban/..."
                  value={singleUrl}
                  onChange={(e) => setSingleUrl(e.target.value)}
                  style={{
                    borderColor: singleUrl.trim() ? 'var(--border-subtle)' : 'rgba(239,68,68,0.3)',
                  }}
                />
                <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px', display: 'block' }}>
                  Bắt buộc — Chatbot sẽ tạo link highlight (#:~:text=) trực tiếp đến đoạn liên quan.
                </span>
              </div>

              {/* Actions */}
              <div className="config-row" style={{ marginTop: '12px' }}>
                <div className="toggle-wrapper" onClick={() => setRecreate(!recreate)}>
                  <div className={`toggle ${recreate ? 'active' : ''}`} />
                  <span className="toggle-label">
                    <AlertTriangle size={14} style={{ display: 'inline', verticalAlign: '-2px', marginRight: '4px', color: 'var(--warning)' }} />
                    <span className="warning">Recreate Collection</span> — Xóa toàn bộ KB trước khi ingest
                  </span>
                </div>

                <button
                  className="btn btn-primary"
                  onClick={handleUpload}
                  disabled={uploading || !canUploadSingle}
                  id="start-ingest-btn"
                  title={!canUploadSingle ? 'Cần chọn file và nhập Source URL' : ''}
                >
                  {uploading ? (
                    <>
                      <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} />
                      Đang xử lý... {totalProgress}%
                    </>
                  ) : (
                    <>
                      <Rocket size={16} />
                      Bắt Đầu Ingest
                    </>
                  )}
                </button>
              </div>

              {uploading && (
                <div className="progress-bar">
                  <div className="progress-bar-fill" style={{ width: `${totalProgress}%` }} />
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ====================== BATCH MODE ====================== */}
      {mode === 'batch' && (
        <div className="animate-fade-in">
          {/* Dropzone */}
          <div
            {...batchDropzone.getRootProps()}
            className={`dropzone ${batchDropzone.isDragActive ? 'active' : ''}`}
            id="upload-dropzone-batch"
            style={batchEntries.length >= 3 ? { opacity: 0.5, pointerEvents: 'none' } : {}}
          >
            <input {...batchDropzone.getInputProps()} />
            <FolderUp className="dropzone-icon" />
            <h3>
              {batchDropzone.isDragActive
                ? 'Thả files vào đây...'
                : batchEntries.length >= 3
                  ? 'Đã đạt giới hạn 3 files'
                  : `Kéo thả files hoặc click để chọn (${batchEntries.length}/3)`}
            </h3>
            <p>Hỗ trợ .pdf và .docx — Tối đa 3 tài liệu</p>
          </div>

          {/* Batch entries — each file with its own URL */}
          {batchEntries.length > 0 && (
            <div className="animate-slide-up" style={{ marginTop: '16px' }}>
              <div style={{
                fontSize: '12px',
                color: 'var(--text-muted)',
                textTransform: 'uppercase',
                letterSpacing: '1px',
                fontWeight: 600,
                marginBottom: '12px',
              }}>
                {batchEntries.length} / 3 tài liệu — Mỗi file cần 1 Source URL riêng
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {batchEntries.map((entry, i) => {
                  const icon = getFileIcon(entry.file.name);
                  const pct = progress[entry.file.name];
                  const hasUrl = entry.sourceUrl.trim().length > 0;

                  return (
                    <div
                      key={`${entry.file.name}-${i}`}
                      className="glass-card"
                      style={{
                        padding: '16px',
                        animation: `slideUp 0.3s ease forwards`,
                        animationDelay: `${i * 0.05}s`,
                      }}
                    >
                      {/* File info row */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                        <div
                          style={{
                            width: '24px',
                            height: '24px',
                            borderRadius: 'var(--radius-full)',
                            background: 'var(--accent-glow)',
                            color: 'var(--accent)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '12px',
                            fontWeight: 700,
                            flexShrink: 0,
                          }}
                        >
                          {i + 1}
                        </div>
                        <div className={`file-item-icon ${icon.cls}`} style={{ width: '32px', height: '32px' }}>
                          {icon.cls === 'pdf' ? <File size={16} /> : <FileText size={16} />}
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div className="file-item-name">{entry.file.name}</div>
                          <div className="file-item-size">{formatFileSize(entry.file.size)}</div>
                        </div>
                        {!uploading && (
                          <button
                            className="file-item-remove"
                            onClick={() => removeBatchEntry(i)}
                            title="Xóa file"
                          >
                            <X size={16} />
                          </button>
                        )}
                      </div>

                      {/* Source URL */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Link size={14} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                        <input
                          type="url"
                          className="input-field"
                          placeholder="https://thuvienphapluat.vn/van-ban/... *"
                          value={entry.sourceUrl}
                          onChange={(e) => updateBatchUrl(i, e.target.value)}
                          disabled={uploading}
                          style={{
                            padding: '10px 14px',
                            fontSize: '13px',
                            borderColor: hasUrl ? 'var(--border-subtle)' : 'rgba(239,68,68,0.3)',
                          }}
                        />
                      </div>

                      {/* Progress */}
                      {pct !== undefined && (
                        <div className="progress-bar" style={{ marginTop: '10px' }}>
                          <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Actions */}
              <div className="upload-config" style={{ marginTop: '16px' }}>
                <div className="config-row">
                  <div className="toggle-wrapper" onClick={() => setRecreate(!recreate)}>
                    <div className={`toggle ${recreate ? 'active' : ''}`} />
                    <span className="toggle-label">
                      <AlertTriangle size={14} style={{ display: 'inline', verticalAlign: '-2px', marginRight: '4px', color: 'var(--warning)' }} />
                      <span className="warning">Recreate Collection</span> — Xóa toàn bộ KB trước khi ingest
                    </span>
                  </div>

                  <button
                    className="btn btn-primary"
                    onClick={handleUpload}
                    disabled={uploading || !canUploadBatch}
                    id="start-ingest-batch-btn"
                    title={!canUploadBatch ? 'Mỗi file cần có Source URL' : ''}
                  >
                    {uploading ? (
                      <>
                        <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} />
                        Đang xử lý... {totalProgress}%
                      </>
                    ) : (
                      <>
                        <Rocket size={16} />
                        Bắt Đầu Ingest ({batchEntries.length} files)
                      </>
                    )}
                  </button>
                </div>

                {uploading && (
                  <div className="progress-bar">
                    <div className="progress-bar-fill" style={{ width: `${totalProgress}%` }} />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ====================== RESULTS ====================== */}
      {results.length > 0 && (
        <div style={{ marginTop: '20px' }} className="stagger-children">
          {results.map((r) => (
            <div
              key={r.fileName}
              className="file-item"
              style={{
                borderColor: r.success ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)',
                background: r.success ? 'var(--success-bg)' : 'var(--error-bg)',
              }}
            >
              <div className={`file-item-icon ${r.success ? 'docx' : 'pdf'}`}
                   style={{
                     background: r.success ? 'var(--success-bg)' : 'var(--error-bg)',
                     color: r.success ? 'var(--success)' : 'var(--error)',
                   }}>
                {r.success ? '✅' : '❌'}
              </div>
              <div className="file-item-info">
                <div className="file-item-name">{r.fileName}</div>
                <div className="file-item-size" style={{ color: r.success ? 'var(--success)' : 'var(--error)' }}>
                  {r.success
                    ? `${r.info.chunks_stored} chunks • ${r.info.documents_loaded} documents loaded`
                    : r.info
                  }
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
