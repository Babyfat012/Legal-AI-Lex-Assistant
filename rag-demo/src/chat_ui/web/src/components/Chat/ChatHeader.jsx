import { useState } from 'react';
import { strings } from '../../i18n/strings';
import { getDocumentHistory } from '../../api/docgen';
import DocumentHistory from '../DocumentHistory/DocumentHistory';

export default function ChatHeader({ lang, onToggleLang, reasoning, onToggleReasoning }) {
  const t = strings[lang];
  const [showDocuments, setShowDocuments] = useState(false);

  const handleDocumentsClick = async () => {
    setShowDocuments(!showDocuments);
  };

  return (
    <div className="chat-header">
      <button className={`chat-header-btn ${reasoning ? 'active' : ''}`} onClick={onToggleReasoning}>
        🧠 Reasoning {reasoning ? 'ON' : 'OFF'}
      </button>
      <button className="chat-header-btn" onClick={handleDocumentsClick}>
        📄 Documents
      </button>
      <button className="chat-header-btn" onClick={onToggleLang}>{t.langSwitch}</button>

      {showDocuments && (
        <DocumentHistory
          lang={lang}
          onClose={() => setShowDocuments(false)}
        />
      )}
    </div>
  );
}
