export default function StatCard({ icon: Icon, value, label, color }) {
  const iconBg = color ? `rgba(${color}, 0.12)` : 'var(--accent-glow)';
  const iconColor = color ? `rgb(${color})` : 'var(--accent)';

  return (
    <div className="stat-card glass-card">
      <div className="stat-card-header">
        <div
          className="stat-card-icon"
          style={{ background: iconBg, color: iconColor }}
        >
          <Icon size={20} />
        </div>
      </div>
      <div className="stat-card-value">{value}</div>
      <div className="stat-card-label">{label}</div>
    </div>
  );
}
