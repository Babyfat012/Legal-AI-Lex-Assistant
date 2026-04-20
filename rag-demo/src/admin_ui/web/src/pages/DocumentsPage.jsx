import { useState, useEffect } from 'react';
import { Search, Trash2, FileText, AlertCircle, RefreshCw } from 'lucide-react';
import toast from 'react-hot-toast';
import { getIngestLogs, deleteDocument } from '../api/client';

function StatusBadge({ status }) {
  const map = {
    success: { cls: 'badge-success', label: 'Success' },
    failed: { cls: 'badge-error', label: 'Failed' },
    deleted: { cls: 'badge-muted', label: 'Deleted' },
  };
  const badge = map[status] || { cls: 'badge-info', label: status };
  return <span className={`badge ${badge.cls}`}>{badge.label}</span>;
}

export default function DocumentsPage() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const data = await getIngestLogs();
      setLogs(data);
    } catch (err) {
      toast.error('Không thể tải danh sách tài liệu');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await deleteDocument(deleteTarget);
      toast.success(`Đã xóa ${deleteTarget}`);
      setDeleteTarget(null);
      fetchLogs();
    } catch (err) {
      toast.error(`Lỗi khi xóa: ${err.response?.data?.detail || err.message}`);
    } finally {
      setDeleting(false);
    }
  };

  const filtered = search
    ? logs.filter((l) => l.file_name.toLowerCase().includes(search.toLowerCase()))
    : logs;

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1>📋 Quản Lý Tài Liệu</h1>
        <p>Xem và quản lý các tài liệu đã ingest vào knowledge base</p>
      </div>

      {/* Actions Row */}
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap' }}>
        <div className="search-bar" style={{ flex: 1 }}>
          <Search className="search-bar-icon" />
          <input
            id="search-documents"
            className="input-field"
            placeholder="Tìm kiếm theo tên file..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <button className="btn btn-ghost btn-sm" onClick={fetchLogs} disabled={loading}>
          <RefreshCw size={14} className={loading ? 'spinning' : ''} style={loading ? { animation: 'spin 1s linear infinite' } : {}} />
          Refresh
        </button>
      </div>

      {/* Table */}
      {loading ? (
        <div className="table-container" style={{ padding: '16px' }}>
          {[...Array(5)].map((_, i) => (
            <div key={i} className="skeleton skeleton-row" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty-state glass-card">
          <FileText className="empty-state-icon" />
          <h3>{search ? 'Không tìm thấy kết quả' : 'Chưa có tài liệu nào'}</h3>
          <p>{search ? 'Thử từ khóa khác' : 'Upload tài liệu ở tab Upload để bắt đầu'}</p>
        </div>
      ) : (
        <div className="table-container glass-card" style={{ padding: 0 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Tên File</th>
                <th>Ngày Upload</th>
                <th>Chunks</th>
                <th>Thời Gian</th>
                <th>Trạng Thái</th>
                <th>Lỗi</th>
                <th style={{ width: '80px' }}>Thao Tác</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((log) => (
                <tr key={log.id}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <FileText size={16} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                      <span style={{ fontWeight: 500 }}>{log.file_name}</span>
                    </div>
                  </td>
                  <td style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>
                    {log.upload_at
                      ? new Date(log.upload_at).toLocaleString('vi-VN', {
                          day: '2-digit', month: '2-digit', year: 'numeric',
                          hour: '2-digit', minute: '2-digit',
                        })
                      : '—'
                    }
                  </td>
                  <td>
                    <span style={{ fontWeight: 600, color: 'var(--accent)' }}>{log.chunk_count}</span>
                  </td>
                  <td style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>
                    {log.elapsed_secs ? `${log.elapsed_secs}s` : '—'}
                  </td>
                  <td><StatusBadge status={log.status} /></td>
                  <td style={{ maxWidth: '200px' }}>
                    {log.error_msg ? (
                      <span
                        style={{ color: 'var(--error)', fontSize: '12px', cursor: 'help' }}
                        title={log.error_msg}
                      >
                        <AlertCircle size={14} style={{ display: 'inline', verticalAlign: '-2px', marginRight: '4px' }} />
                        {log.error_msg.slice(0, 50)}{log.error_msg.length > 50 ? '...' : ''}
                      </span>
                    ) : '—'}
                  </td>
                  <td>
                    {log.status !== 'deleted' && (
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={() => setDeleteTarget(log.file_name)}
                        title="Xóa tài liệu"
                        id={`delete-${log.id}`}
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Summary */}
      {!loading && filtered.length > 0 && (
        <div style={{ marginTop: '12px', fontSize: '12px', color: 'var(--text-muted)' }}>
          Hiển thị {filtered.length} / {logs.length} tài liệu
          {search && ` • Đang lọc: "${search}"`}
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="modal-overlay" onClick={() => !deleting && setDeleteTarget(null)}>
          <div className="modal-content glass-card" onClick={(e) => e.stopPropagation()}>
            <h3 className="modal-title">Xác nhận xóa</h3>
            <div className="modal-body">
              Bạn có chắc chắn muốn xóa tất cả chunks của{' '}
              <strong style={{ color: 'var(--text-primary)' }}>&quot;{deleteTarget}&quot;</strong>{' '}
              khỏi knowledge base? Hành động này không thể hoàn tác.
            </div>
            <div className="modal-actions">
              <button
                className="btn btn-ghost"
                onClick={() => setDeleteTarget(null)}
                disabled={deleting}
              >
                Hủy
              </button>
              <button
                className="btn btn-danger"
                onClick={handleDelete}
                disabled={deleting}
              >
                {deleting ? (
                  <>
                    <div className="spinner" style={{ width: '14px', height: '14px', borderWidth: '2px' }} />
                    Đang xóa...
                  </>
                ) : (
                  <>
                    <Trash2 size={14} />
                    Xóa vĩnh viễn
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
