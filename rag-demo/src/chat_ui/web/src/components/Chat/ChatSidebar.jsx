import { strings } from '../../i18n/strings';
import { MessageSquare, Plus, LogOut, Trash2 } from 'lucide-react';

export default function ChatSidebar({ conversations, activeConvId, onNewChat, onSelect, onDelete, onLogout, lang, user }) {
  const t = strings[lang];
  return (
    <aside className="chat-sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <span>⚖️</span>
          <h1>Lex</h1>
        </div>
        <button className="sidebar-new-btn" onClick={onNewChat}>
          <Plus size={16} /> {t.newChat}
        </button>
      </div>
      <div className="sidebar-list">
        {conversations.map(c => (
          <div key={c.id} className={`sidebar-item ${c.id === activeConvId ? 'active' : ''}`} onClick={() => onSelect(c.id)}>
            <MessageSquare size={15} />
            <span className="sidebar-item-title">{c.title}</span>
            <button className="sidebar-item-delete" onClick={e => { e.stopPropagation(); onDelete(c.id); }} title={t.deleteConv}>
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
      <div className="sidebar-footer">
        <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: '0 12px 8px', overflow: 'hidden', textOverflow: 'ellipsis' }}>{user?.email}</div>
        <button className="sidebar-logout" onClick={onLogout}>
          <LogOut size={16} />{t.logout}
        </button>
      </div>
    </aside>
  );
}
