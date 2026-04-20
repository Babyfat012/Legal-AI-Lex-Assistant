import { NavLink } from 'react-router-dom';
import { Upload, FileText, BarChart3, LogOut } from 'lucide-react';

const navItems = [
  { to: '/', icon: Upload, label: 'Upload Tài Liệu' },
  { to: '/documents', icon: FileText, label: 'Quản Lý Tài Liệu' },
  { to: '/collection', icon: BarChart3, label: 'Thống Kê Collection' },
];

export default function Sidebar({ onLogout }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="sidebar-logo">⚖️</span>
        <div className="sidebar-brand">
          <h1>Lex Admin</h1>
          <span>Legal AI Platform</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `sidebar-link ${isActive ? 'active' : ''}`
            }
          >
            <Icon className="sidebar-link-icon" size={20} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button className="sidebar-logout" onClick={onLogout}>
          <LogOut size={18} />
          Đăng xuất
        </button>
      </div>
    </aside>
  );
}
