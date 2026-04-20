import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { strings } from '../../i18n/strings';
import SourceCard from './SourceCard';
import TemplateSelector from '../DocGen/TemplateSelector';
import DocumentCard from './DocumentCard';

export default function MessageBubble({ message, lang, onOpenDocgen, onSelectTemplate }) {
  const t = strings[lang];
  const isUser = message.role === 'user';

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-bot'}`}>
      <div className="message-bubble">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>

        {/* DocGen trigger button */}
        {message.docgen && (
          <div className="docgen-trigger">
            <button className="docgen-trigger-btn" onClick={() => {
              console.log('Opening docgen form for template:', message.docgen.template_id);
              onOpenDocgen(message.docgen.template_id);
            }}>
              {t.openForm}
            </button>
          </div>
        )}

        {/* Template selector */}
        {message.showTemplateSelector && (
          <div style={{ marginTop: 12 }}>
            <TemplateSelector lang={lang} onSelect={onSelectTemplate} />
          </div>
        )}

        {/* RAG sources */}
        {message.sources && message.sources.length > 0 && message.toolUsed === 'rag' && (
          <SourceCard label={t.ragSources} sources={message.sources} type="rag" />
        )}

        {/* Web sources */}
        {message.webSources && message.webSources.length > 0 && (
          <SourceCard label={t.webSources} sources={message.webSources} type="web" />
        )}

        {/* Generated document card */}
        {message.document && (
          <DocumentCard document={message.document} lang={lang} />
        )}
      </div>
    </div>
  );
}
