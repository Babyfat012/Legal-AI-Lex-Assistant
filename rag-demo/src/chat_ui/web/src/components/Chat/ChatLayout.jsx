import { useState, useEffect, useRef, useCallback } from 'react';
import { strings } from '../../i18n/strings';
import { sendMessage, detectIntent } from '../../api/chat';
import { listConversations, createConversation, getConversation, deleteConversation } from '../../api/conversations';
import { getDocumentHistory } from '../../api/docgen';
import ChatSidebar from './ChatSidebar';
import ChatHeader from './ChatHeader';
import MessageBubble from './MessageBubble';
import ChatInput from './ChatInput';
import DocGenFormModal from '../DocGen/DocGenFormModal';
import TemplateSelector from '../DocGen/TemplateSelector';
import toast from 'react-hot-toast';

export default function ChatLayout({ user, lang, onToggleLang, onLogout }) {
  const t = strings[lang];
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [reasoning, setReasoning] = useState(false);
  const [docgenModal, setDocgenModal] = useState(null); // template_id
  const [templateSelector, setTemplateSelector] = useState(false);
  const messagesEndRef = useRef(null);

  // Load conversations on mount
  useEffect(() => { loadConversations(); }, []);

  const loadConversations = async () => {
    try {
      const data = await listConversations();
      setConversations(data);
    } catch { /* ignore */ }
  };

  const handleNewChat = async () => {
    setActiveConvId(null);
    setMessages([]);
  };

  const handleSelectConv = async (convId) => {
    setActiveConvId(convId);
    try {
      const data = await getConversation(convId);
      setMessages(data.messages.map(m => ({
        role: m.role,
        content: m.content,
        sources: m.sources,
      })));
    } catch (err) {
      toast.error('Failed to load conversation');
    }
  };

  const handleDeleteConv = async (convId) => {
    try {
      await deleteConversation(convId);
      setConversations(prev => prev.filter(c => c.id !== convId));
      if (activeConvId === convId) { setActiveConvId(null); setMessages([]); }
      toast.success('Đã xóa');
    } catch { toast.error('Lỗi khi xóa'); }
  };

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, loading, scrollToBottom]);

  const handleSend = async (text) => {
    if (!text.trim() || loading) return;

    // Add user message
    const userMsg = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      // If no active conversation, create one
      let convId = activeConvId;
      if (!convId) {
        const title = text.length > 50 ? text.slice(0, 50) + '…' : text;
        const conv = await createConversation(title);
        convId = conv.id;
        setActiveConvId(convId);
        setConversations(prev => [conv, ...prev]);
      }

      // Detect intent first
      const intentResult = await detectIntent(text);
      console.log('Intent detection result:', intentResult);

      if (intentResult.intent === 'document_gen' && intentResult.confidence > 0.7) {
        // Document generation flow
        if (intentResult.template_id) {
          const botMsg = {
            role: 'assistant',
            content: t.docgenIntro.replace('{name}', intentResult.template_name || intentResult.template_id),
            docgen: { template_id: intentResult.template_id, template_name: intentResult.template_name },
          };
          setMessages(prev => [...prev, botMsg]);

          // Also send to chat API to persist
          await sendMessage({
            query: text,
            session_id: convId,
            top_k: 5,
            use_reranker: true,
            reasoning_mode: false,
            language: lang,
          }).catch(() => {});
        } else {
          // Unknown template — show selector
          const botMsg = {
            role: 'assistant',
            content: t.selectTemplate,
            showTemplateSelector: true,
          };
          setMessages(prev => [...prev, botMsg]);
        }
        setLoading(false);
        return;
      }

      // Normal legal QA flow
      const data = await sendMessage({
        query: text,
        session_id: convId,
        top_k: 5,
        use_reranker: true,
        reasoning_mode: reasoning,
        language: lang,
      });

      const botMsg = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        webSources: data.web_sources,
        toolUsed: data.tool_used,
        reasoning_steps: data.reasoning_steps,
      };
      setMessages(prev => [...prev, botMsg]);

      // Update conversation title if it was the first message
      loadConversations();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Lỗi kết nối API');
      setMessages(prev => [...prev, { role: 'assistant', content: '❌ Lỗi kết nối. Vui lòng thử lại.' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleTemplateSelect = (templateId) => {
    console.log('Template selected:', templateId);
    setDocgenModal(templateId);
  };

  const handleDocGenComplete = async () => {
    setDocgenModal(null);

    // Get the most recently generated document from the database
    try {
      const history = await getDocumentHistory();
      if (history.length > 0) {
        const latestDoc = history[0];
        const docMsg = {
          role: 'assistant',
          content: t.docgenSuccess + '\n\n' + t.docgenDisclaimer,
          document: latestDoc
        };
        setMessages(prev => [...prev, docMsg]);
      } else {
        const successMsg = { role: 'assistant', content: t.docgenSuccess + '\n\n' + t.docgenDisclaimer };
        setMessages(prev => [...prev, successMsg]);
      }
    } catch (error) {
      console.error('Failed to fetch generated document:', error);
      const successMsg = { role: 'assistant', content: t.docgenSuccess + '\n\n' + t.docgenDisclaimer };
      setMessages(prev => [...prev, successMsg]);
    }
  };

  return (
    <div className="chat-layout">
      <ChatSidebar
        conversations={conversations}
        activeConvId={activeConvId}
        onNewChat={handleNewChat}
        onSelect={handleSelectConv}
        onDelete={handleDeleteConv}
        onLogout={onLogout}
        lang={lang}
        user={user}
      />

      <div className="chat-main">
        <ChatHeader lang={lang} onToggleLang={onToggleLang} reasoning={reasoning} onToggleReasoning={() => setReasoning(!reasoning)} />

        <div className="chat-messages">
          <div className="chat-messages-inner">
            {messages.length === 0 && !loading && (
              <div className="chat-welcome">
                <div className="chat-welcome-icon">⚖️</div>
                <h2>Lex — Trợ lý Pháp lý AI</h2>
                <p>{lang === 'vi' ? 'Hãy đặt câu hỏi pháp luật hoặc yêu cầu tạo đơn để bắt đầu.' : 'Ask a legal question or request to create a document to get started.'}</p>
              </div>
            )}

            {messages.map((msg, i) => (
              <MessageBubble
                key={i}
                message={msg}
                lang={lang}
                onOpenDocgen={(tid) => setDocgenModal(tid)}
                onSelectTemplate={handleTemplateSelect}
              />
            ))}

            {loading && (
              <div className="message message-bot">
                <div className="message-bubble">
                  <div className="typing-indicator">
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <ChatInput lang={lang} onSend={handleSend} disabled={loading} />
      </div>

      {docgenModal && (
        <DocGenFormModal
          templateId={docgenModal}
          lang={lang}
          onClose={() => {
            console.log('Closing modal for template:', docgenModal);
            setDocgenModal(null);
          }}
          onComplete={handleDocGenComplete}
        />
      )}
    </div>
  );
}
