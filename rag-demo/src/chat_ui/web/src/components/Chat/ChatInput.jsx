import { useState } from 'react';
import { strings } from '../../i18n/strings';
import { Send } from 'lucide-react';

export default function ChatInput({ lang, onSend, disabled }) {
  const t = strings[lang];
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim() || disabled) return;
    onSend(text.trim());
    setText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-input-container">
      <form className="chat-input-inner" onSubmit={handleSubmit}>
        <textarea
          className="chat-input"
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t.typeMessage}
          rows={1}
          disabled={disabled}
        />
        <button className="chat-send-btn" type="submit" disabled={disabled || !text.trim()}>
          <Send size={20} />
        </button>
      </form>
      <p className="chat-disclaimer">Lex có thể mắc sai sót. Hãy xác minh thông tin quan trọng.</p>
    </div>
  );
}
