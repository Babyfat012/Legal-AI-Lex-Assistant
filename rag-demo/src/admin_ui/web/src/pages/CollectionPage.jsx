import { useState, useEffect, useCallback } from 'react';
import { Database, Ruler, Activity, Layers, Heart, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react';
import toast from 'react-hot-toast';
import StatCard from '../components/UI/StatCard';
import { getCollectionInfo, healthCheck } from '../api/client';

export default function CollectionPage() {
  const [info, setInfo] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showJson, setShowJson] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [collectionData, healthData] = await Promise.all([
        getCollectionInfo().catch(() => null),
        healthCheck().catch(() => null),
      ]);
      setInfo(collectionData);
      setHealth(healthData);
    } catch {
      toast.error('Không thể kết nối Backend');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    // Auto-refresh every 30s
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading && !info) {
    return (
      <div className="animate-fade-in">
        <div className="page-header">
          <h1>📊 Thống Kê Collection</h1>
          <p>Đang tải dữ liệu...</p>
        </div>
        <div className="stats-grid">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="skeleton" style={{ height: '140px', borderRadius: 'var(--radius-lg)' }} />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1>📊 Thống Kê Collection</h1>
          <p>Thông số Qdrant vector database và trạng thái hệ thống</p>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={fetchData} disabled={loading}>
          <RefreshCw size={14} style={loading ? { animation: 'spin 1s linear infinite' } : {}} />
          Refresh
        </button>
      </div>

      {/* Health Status */}
      {health && (
        <div
          className="glass-card animate-slide-up"
          style={{ padding: '16px 20px', marginBottom: '24px', display: 'flex', gap: '24px', alignItems: 'center', flexWrap: 'wrap' }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Heart size={16} style={{ color: health.status === 'ok' ? 'var(--success)' : 'var(--error)' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>API Server</span>
            <span className={`badge ${health.status === 'ok' ? 'badge-success' : 'badge-error'}`}>
              {health.status === 'ok' ? 'Online' : 'Offline'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Database size={16} style={{ color: health.qdrant?.startsWith('ok') ? 'var(--success)' : 'var(--error)' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>Qdrant</span>
            <span className={`badge ${health.qdrant?.startsWith('ok') ? 'badge-success' : 'badge-error'}`}>
              {health.qdrant}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={16} style={{ color: health.openai_key_set ? 'var(--success)' : 'var(--warning)' }} />
            <span style={{ fontSize: '13px', fontWeight: 600 }}>OpenAI Key</span>
            <span className={`badge ${health.openai_key_set ? 'badge-success' : 'badge-warning'}`}>
              {health.openai_key_set ? 'Configured' : 'Missing'}
            </span>
          </div>
        </div>
      )}

      {/* Stat Cards */}
      {info ? (
        <>
          <div className="stats-grid stagger-children">
            <StatCard
              icon={Database}
              value={info.points_count?.toLocaleString() || '0'}
              label="Tổng Points"
              color="56, 189, 248"
            />
            <StatCard
              icon={Ruler}
              value={info.dimension || '—'}
              label="Dimension"
              color="139, 92, 246"
            />
            <StatCard
              icon={Activity}
              value={info.status?.toUpperCase() || '—'}
              label="Status"
              color="34, 197, 94"
            />
            <StatCard
              icon={Layers}
              value={info.segments_count || '0'}
              label="Segments"
              color="245, 158, 11"
            />
          </div>

          {/* JSON Viewer Toggle */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <button
              onClick={() => setShowJson(!showJson)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                background: 'none',
                border: 'none',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: 600,
                fontFamily: 'inherit',
                width: '100%',
                padding: 0,
              }}
              id="toggle-json-viewer"
            >
              {showJson ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              Raw JSON Response
            </button>
            {showJson && (
              <div className="json-viewer animate-slide-up">
                {JSON.stringify(info, null, 2)}
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="empty-state glass-card">
          <Database className="empty-state-icon" />
          <h3>Không thể kết nối đến Qdrant</h3>
          <p>Kiểm tra rằng Backend đang chạy tại http://localhost:8000</p>
        </div>
      )}
    </div>
  );
}
